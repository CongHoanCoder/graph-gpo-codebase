"""Protocol A evaluation: direct augmentation.

Usage:
  python -m graph_cpo.scripts.run_eval_a configs/dev_qwen_0.5b.yaml \
      --input data/test.jsonl --limit 20 \
      [--rl_checkpoint runs/dev/rl_graph_cpo/checkpoint]
"""
from __future__ import annotations

from pathlib import Path

import torch

from ._common import cfg_from_args, device, load_jsonl
from graph_cpo.eval.protocol_a import run_protocol_a
from graph_cpo.eval.solver import SolverHarness
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

    rows = load_jsonl(inp)[:limit]
    problems = []
    for r in rows:
        pid = r.get("id", r["question"][:40])
        sdg = load_graph(cfg.sdg.graph_cache, pid)
        if sdg is None:
            continue
        r = dict(r); r["id"] = pid; r["sdg"] = sdg
        problems.append(r)
    if not problems:
        raise SystemExit("No problems with cached graphs; run build_graphs on the split.")

    # generator: RL checkpoint if present, else base model
    ckpt = None
    for cand in argv:
        if cand.startswith("--rl_checkpoint="):
            ckpt = cand.split("=", 1)[1]
    rl_dirs = list((Path(cfg.output_dir)).glob("rl_*/checkpoint"))
    use_ckpt = ckpt or (str(rl_dirs[0]) if rl_dirs else None)
    if use_ckpt:
        policy = GraphAugmentedPolicy.load_with_adapters(
            use_ckpt, cfg.model.backbone, graph_prefix_dim=cfg.sdg.prefix_dim,
            dtype=cfg.model.dtype)
    else:
        policy = GraphAugmentedPolicy(cfg.model.backbone, cfg.model.tokenizer,
                                      lora_enabled=False,
                                      graph_prefix_dim=cfg.sdg.prefix_dim,
                                      dtype=cfg.model.dtype)
    policy.to(dev).eval()
    engine = HFRolloutEngine(policy, temperature=0.7, device=dev)

    span_tok = AutoTokenizer.from_pretrained(cfg.sdg.tagger_model)
    bundle_gnn = GraphEncoderBundle(cfg.sdg.tagger_model, hidden=cfg.sdg.gnn.hidden,
                                    num_layers=cfg.sdg.gnn.layers).to(dev)
    bundle_gnn.eval()
    node_emb = {p["sdg"].problem_id: bundle_gnn.encode_graph(span_tok, p["sdg"], device=dev)
                for p in problems}
    rewards = build_reward_bundle(cfg.rewards, cfg.planning, device=dev)

    def generator(sdg: SDG):
        gate = rewards.make_gate(sdg)
        trajs = sample_trajectories(engine, sdg, cfg.planning,
                                    node_emb=node_emb.get(sdg.problem_id),
                                    gate=gate, num_rollouts=1,
                                    max_new_tokens=cfg.training.max_new_tokens)
        return trajs[0]

    solver = SolverHarness(cfg.eval.solver_backbone, max_tokens=cfg.eval.solver_max_tokens,
                           timeout_s=cfg.eval.solver_timeout_s, device=dev)
    result = run_protocol_a(problems, solver, generator=generator,
                            nli_model=cfg.rewards.nli_model, device=dev,
                            save_path=str(Path(cfg.output_dir) / "protocol_a.json"))
    print(f"[protocol_a] n={result.num_problems} "
          f"treatment={result.success_treatment:.3f} control={result.success_control:.3f} "
          f"random={result.success_random:.3f} "
          f"PO={result.purpose_orientation:.3f} targeted={result.targeted:.3f}")


if __name__ == "__main__":
    main()
