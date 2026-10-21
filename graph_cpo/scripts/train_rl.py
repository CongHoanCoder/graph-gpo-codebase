"""Graph-CPO reinforcement learning training.

Loads the SFT checkpoint as pi_theta and pi_ref, then optimizes with the
decoupled per-dimension advantage + curriculum weighting + clipped CPO loss.
Run with --baseline scalar for the scalar-GRPO comparison (reward-collapse
figure + ablation row).

Usage:
  python -m graph_cpo.scripts.train_rl configs/dev_qwen_0.5b.yaml \
      [--limit 200] [--epochs 2] [--baseline scalar]
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ._common import cfg_from_args, device, load_jsonl
from graph_cpo.eval.report import training_curves
from graph_cpo.models.policy import GraphAugmentedPolicy
from graph_cpo.models.rollout import build_rollout_engine
from graph_cpo.planning.prefix import GraphPrefixProjector
from graph_cpo.planning.teacher import TeacherStep, TeacherTrajectory
from graph_cpo.rewards.bundle import build_reward_bundle
from graph_cpo.sdg.gnn import GraphEncoderBundle
from graph_cpo.sdg.schema import SDG
from graph_cpo.sdg.store import load_graph
from graph_cpo.trainer.grpo_scalar import make_scalar_grpo_trainer
from graph_cpo.trainer.rl import GraphCPOTrainer
from transformers import AutoTokenizer


def main() -> None:
    cfg, argv = cfg_from_args()
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 200
    epochs = int(argv[argv.index("--epochs") + 1]) if "--epochs" in argv else cfg.training.rl.epochs
    baseline = argv[argv.index("--baseline") + 1] if "--baseline" in argv else "cpo"
    dev = device()

    # cached SDGs
    problems: list[SDG] = []
    cache = Path(cfg.sdg.graph_cache)
    for f in sorted(cache.glob("*.jsonl")) + sorted(cache.glob("*.json")):
        if f.suffix == ".json":
            sdg = SDG.from_dict(json.loads(f.read_text()))
            problems.append(sdg)
        if len(problems) >= limit:
            break
    if not problems:
        raise SystemExit(f"No cached graphs in {cache}; run build_graphs first.")

    # node embeddings for the graph prefix and rewards
    span_tok = AutoTokenizer.from_pretrained(cfg.sdg.tagger_model)
    bundle = GraphEncoderBundle(cfg.sdg.tagger_model, hidden=cfg.sdg.gnn.hidden,
                                num_layers=cfg.sdg.gnn.layers).to(dev)
    relf = Path(cfg.output_dir) / "sdg_relf" / "gnn.pt"
    if relf.exists():
        bundle.gnn.load_state_dict(torch.load(relf))
    bundle.eval()
    node_emb = {sdg.problem_id: bundle.encode_graph(span_tok, sdg, device=dev)
                for sdg in problems}

    # policy + reference from the SFT checkpoint
    sft_ckpt = Path(cfg.output_dir) / "sft_checkpoint"
    if sft_ckpt.exists():
        policy = GraphAugmentedPolicy.load_with_adapters(
            str(sft_ckpt), cfg.model.backbone,
            graph_prefix_dim=cfg.sdg.prefix_dim, dtype=cfg.model.dtype)
        ref_policy = GraphAugmentedPolicy.load_with_adapters(
            str(sft_ckpt), cfg.model.backbone,
            graph_prefix_dim=cfg.sdg.prefix_dim, dtype=cfg.model.dtype)
    else:
        print("[rl] no SFT checkpoint found; initializing from base model (cold start)")
        kwargs = dict(graph_prefix_dim=cfg.sdg.prefix_dim, dtype=cfg.model.dtype)
        policy = GraphAugmentedPolicy(cfg.model.backbone, cfg.model.tokenizer, **kwargs)
        ref_policy = GraphAugmentedPolicy(cfg.model.backbone, cfg.model.tokenizer, **kwargs)
    prefix_proj = GraphPrefixProjector(cfg.sdg.gnn.hidden, cfg.sdg.prefix_dim,
                                       num_prefix_tokens=policy.num_prefix_tokens)
    prefix_path = Path(cfg.output_dir) / "sft_checkpoint" / "graph_prefix_projector.pt"
    if prefix_path.exists():
        prefix_proj.load_state_dict(torch.load(prefix_path))
    prefix_proj.to(dev)

    bundle_rw = build_reward_bundle(cfg.rewards, cfg.planning, device=dev)
    engine = build_rollout_engine(cfg.training.rollout_engine, policy,
                                  cfg.model.backbone, device=dev)

    teachers = {}
    for f in sorted(Path(cfg.planning.teacher_dir).glob("*_teachers.jsonl")):
        for row in load_jsonl(f):
            teachers[row["problem_id"]] = TeacherTrajectory(
                problem_id=row["problem_id"],
                steps=[TeacherStep(**s) for s in row["steps"]])

    common = dict(policy=policy, ref_policy=ref_policy, engine=engine,
                  bundle=bundle_rw, cfg=cfg.training.rl,
                  planning_cfg=cfg.planning, prefix_projector=prefix_proj,
                  teacher_by_problem=teachers, node_embeddings=node_emb,
                  device=dev)
    if baseline == "scalar":
        trainer = make_scalar_grpo_trainer(**common)
        run_name = "grpo_scalar"
    else:
        trainer = GraphCPOTrainer(**common)
        run_name = "graph_cpo"

    for epoch in range(epochs):
        print(f"[rl] === epoch {epoch+1}/{epochs} ({run_name}) ===")
        for sdg in problems:
            trainer.update_on_problem(sdg)

    out = Path(cfg.output_dir) / f"rl_{run_name}"
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out / "checkpoint")
    torch.save(prefix_proj.state_dict(), out / "graph_prefix_projector.pt")
    curves = trainer.plot_ready()
    (out / "training_log.json").write_text(json.dumps(
        {"history": [vars(h) for h in trainer.history]}, indent=2, default=str))
    training_curves({run_name: curves}, str(out / "training_curves.png"))
    print(f"[rl] saved to {out}")


if __name__ == "__main__":
    main()
