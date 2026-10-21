"""Sequence validity evaluation (generator validity rates).

Usage:
  python -m graph_cpo.scripts.run_validity configs/dev_qwen_0.5b.yaml \
      --input data/test.jsonl --limit 20
"""
from __future__ import annotations

import json
from pathlib import Path

from ._common import cfg_from_args, device, load_jsonl
from graph_cpo.eval.solver import SolverHarness
from graph_cpo.eval.validity import sequence_validity
from graph_cpo.models.policy import GraphAugmentedPolicy
from graph_cpo.models.rollout import HFRolloutEngine
from graph_cpo.rewards.bundle import build_reward_bundle
from graph_cpo.sdg.gnn import GraphEncoderBundle
from graph_cpo.sdg.schema import SDG
from graph_cpo.sdg.store import load_graph
from graph_cpo.trajectory import sample_trajectories
from transformers import AutoTokenizer


def main() -> None:
    cfg, argv = cfg_from_args()
    inp = argv[argv.index("--input") + 1] if "--input" in argv else "data/test.jsonl"
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 20
    dev = device()

    problems = []
    for r in load_jsonl(inp)[:limit]:
        pid = r.get("id", r["question"][:40])
        sdg = load_graph(cfg.sdg.graph_cache, pid)
        if sdg is not None:
            r = dict(r); r["id"] = pid; r["sdg"] = sdg
            problems.append(r)
    if not problems:
        raise SystemExit("No problems with cached graphs; run build_graphs first.")

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
    node_emb = {p["sdg"].problem_id: gnn.encode_graph(span_tok, p["sdg"], device=dev)
                for p in problems}
    rewards = build_reward_bundle(cfg.rewards, cfg.planning, device=dev)

    def generator(sdg: SDG):
        gate = rewards.make_gate(sdg)
        trajs = sample_trajectories(engine, sdg, cfg.planning,
                                    node_emb=node_emb.get(sdg.problem_id),
                                    gate=gate, num_rollouts=1,
                                    max_new_tokens=cfg.training.max_new_tokens)
        return trajs[0]

    teacher = SolverHarness(cfg.eval.solver_backbone,
                            max_tokens=cfg.eval.solver_max_tokens,
                            timeout_s=cfg.eval.solver_timeout_s, device=dev)
    result = sequence_validity(problems, teacher, generator)
    out = Path(cfg.output_dir) / "validity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"validity_rate": result.validity_rate, "num": result.num_sequences,
         "per_problem": result.per_problem}, indent=2))
    print(f"[validity] rate={result.validity_rate:.3f} n={result.num_sequences} -> {out}")


if __name__ == "__main__":
    main()
