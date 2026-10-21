"""Targeted supervised warm start.

Loads cached SDGs + teacher trajectories, builds teacher-forced graph-
conditioned trajectories, and fine-tunes the Graph-Augmented Policy (LoRA)
by NLL. The resulting checkpoint is both the RL initialization (pi_theta) and
the KL reference (pi_ref).

Usage:
  python -m graph_cpo.scripts.train_sft configs/dev_qwen_0.5b.yaml
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from ._common import cfg_from_args, device, load_jsonl
from graph_cpo.models.policy import GraphAugmentedPolicy
from graph_cpo.planning.prefix import GraphPrefixProjector
from graph_cpo.sdg.gnn import GraphEncoderBundle
from graph_cpo.sdg.schema import SDG
from graph_cpo.sdg.store import load_graph
from graph_cpo.planning.teacher import TeacherStep, TeacherTrajectory
from graph_cpo.trainer.sft import SFTTrainer
from graph_cpo.trajectory import rollout_with_teacher


def main() -> None:
    cfg, argv = cfg_from_args()
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 1000
    dev = device()

    # teacher trajectories
    teachers: dict[str, TeacherTrajectory] = {}
    for f in sorted(Path(cfg.planning.teacher_dir).glob("*_teachers.jsonl")):
        for row in load_jsonl(f):
            teachers[row["problem_id"]] = TeacherTrajectory(
                problem_id=row["problem_id"],
                steps=[TeacherStep(**s) for s in row["steps"]])
    if not teachers:
        raise SystemExit(f"No teachers in {cfg.planning.teacher_dir}; "
                         "run build_teachers first.")

    # teacher-forced trajectories conditioned on cached graphs
    trajectories, node_embeddings = [], {}
    span_tok = AutoTokenizer.from_pretrained(cfg.sdg.tagger_model)
    bundle = GraphEncoderBundle(cfg.sdg.tagger_model, hidden=cfg.sdg.gnn.hidden,
                                num_layers=cfg.sdg.gnn.layers).to(dev)
    relf = Path(cfg.output_dir) / "sdg_relf" / "gnn.pt"
    if relf.exists():
        bundle.gnn.load_state_dict(torch.load(relf))
    bundle.eval()
    for pid, teacher in teachers.items():
        sdg = load_graph(cfg.sdg.graph_cache, pid)
        if sdg is None:
            continue
        traj = rollout_with_teacher(sdg, teacher, cfg.planning)
        if traj.timesteps:
            trajectories.append(traj)
            node_embeddings[pid] = bundle.encode_graph(span_tok, sdg, device=dev)
        if len(trajectories) >= limit:
            break
    if not trajectories:
        raise SystemExit("No teacher trajectories matched cached graphs; "
                         "run build_graphs first.")

    policy = GraphAugmentedPolicy(
        cfg.model.backbone, cfg.model.tokenizer,
        lora_enabled=cfg.model.lora.enabled, lora_r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha, lora_dropout=cfg.model.lora.dropout,
        lora_targets=cfg.model.lora.target_modules,
        graph_prefix_dim=cfg.sdg.prefix_dim, dtype=cfg.model.dtype,
        gradient_checkpointing=cfg.model.gradient_checkpointing)
    print(f"[sft] trainable params: {policy.trainable_parameter_count():,}")

    prefix_proj = GraphPrefixProjector(cfg.sdg.gnn.hidden, cfg.sdg.prefix_dim,
                                       num_prefix_tokens=policy.num_prefix_tokens)
    trainer = SFTTrainer(policy, cfg.training.sft, prefix_projector=prefix_proj,
                         device=dev)
    trainer.train(trajectories, node_embeddings=node_embeddings)

    out = Path(cfg.output_dir) / "sft_checkpoint"
    trainer.save(out)
    torch.save(prefix_proj.state_dict(), Path(out) / "graph_prefix_projector.pt")


if __name__ == "__main__":
    main()
