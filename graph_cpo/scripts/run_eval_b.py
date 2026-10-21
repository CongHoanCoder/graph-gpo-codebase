"""Protocol B evaluation: downstream fine-tuning utility.

Usage:
  python -m graph_cpo.scripts.run_eval_b configs/dev_qwen_0.5b.yaml \
      --seeds data/train.jsonl --benchmarks data/test.jsonl --limit 50
"""
from __future__ import annotations

from pathlib import Path

import torch

from ._common import cfg_from_args, device, load_jsonl
from graph_cpo.eval.protocol_b import run_protocol_b, save_result
from graph_cpo.models.policy import GraphAugmentedPolicy
from graph_cpo.models.rollout import HFRolloutEngine
from graph_cpo.rewards.bundle import build_reward_bundle
from graph_cpo.sdg.gnn import GraphEncoderBundle
from graph_cpo.sdg.schema import SDG
from graph_cpo.sdg.store import load_graph
from graph_cpo.trajectory import sample_trajectories
from transformers import AutoTokenizer


def _attach_sdgs(rows, cache):
    out = []
    for r in rows:
        pid = r.get("id", r["question"][:40])
        sdg = load_graph(cache, pid)
        if sdg is not None:
            r = dict(r); r["id"] = pid; r["sdg"] = sdg
            out.append(r)
    return out


def main() -> None:
    cfg, argv = cfg_from_args()
    seeds_path = argv[argv.index("--seeds") + 1] if "--seeds" in argv else "data/train.jsonl"
    bench_path = argv[argv.index("--benchmarks") + 1] if "--benchmarks" in argv else "data/test.jsonl"
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 50
    dev = device()

    seeds = _attach_sdgs(load_jsonl(seeds_path)[:limit], cfg.sdg.graph_cache)
    benchmarks = []
    for r in load_jsonl(bench_path)[:limit]:
        benchmarks.append({"question": r["question"],
                           "answer": r.get("answer"),
                           "benchmark": Path(bench_path).stem})
    if not seeds:
        raise SystemExit("No seed problems with cached graphs; run build_graphs first.")

    rl_dirs = list(Path(cfg.output_dir).glob("rl_*/checkpoint"))
    if rl_dirs:
        policy = GraphAugmentedPolicy.load_with_adapters(
            str(rl_dirs[0]), cfg.model.backbone, graph_prefix_dim=cfg.sdg.prefix_dim,
            dtype=cfg.model.dtype)
    else:
        policy = GraphAugmentedPolicy(cfg.model.backbone, cfg.model.tokenizer,
                                      lora_enabled=False,
                                      graph_prefix_dim=cfg.sdg.prefix_dim,
                                      dtype=cfg.model.dtype)
    policy.to(dev).eval()
    engine = HFRolloutEngine(policy, temperature=0.7, device=dev)

    span_tok = AutoTokenizer.from_pretrained(cfg.sdg.tagger_model)
    gnn = GraphEncoderBundle(cfg.sdg.tagger_model, hidden=cfg.sdg.gnn.hidden,
                             num_layers=cfg.sdg.gnn.layers).to(dev).eval()
    node_emb = {s["sdg"].problem_id: gnn.encode_graph(span_tok, s["sdg"], device=dev)
                for s in seeds}
    rewards = build_reward_bundle(cfg.rewards, cfg.planning, device=dev)

    def generator(sdg: SDG):
        gate = rewards.make_gate(sdg)
        trajs = sample_trajectories(engine, sdg, cfg.planning,
                                    node_emb=node_emb.get(sdg.problem_id),
                                    gate=gate, num_rollouts=1,
                                    max_new_tokens=cfg.training.max_new_tokens)
        return trajs[0]

    out_dir = Path(cfg.output_dir) / "protocol_b"
    result = run_protocol_b(
        seeds, benchmarks, generator, cfg.eval.solver_backbone, str(out_dir),
        method_name="graph_cpo", variants_per_seed=cfg.eval.protocol_b_variants,
        num_runs=cfg.eval.num_decoding_runs, device=dev)
    save_result(result, str(out_dir / "result.json"))
    print(f"[protocol_b] mean_accuracy={result.mean_accuracy:.3f} "
          f"per_benchmark={result.per_benchmark}")

    baseline = run_protocol_b(
        seeds, benchmarks, generator, cfg.eval.solver_backbone,
        str(out_dir / "baseline"), method_name="qwen3_baseline",
        variants_per_seed=1, num_runs=cfg.eval.num_decoding_runs,
        baseline=True, device=dev)
    save_result(baseline, str(out_dir / "baseline_result.json"))
    print(f"[protocol_b] baseline mean_accuracy={baseline.mean_accuracy:.3f}")


if __name__ == "__main__":
    main()
