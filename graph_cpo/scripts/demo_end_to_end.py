"""End-to-end smoke demo on a handful of GSM8K problems, CPU-friendly.

Pipeline: silver SDG -> cached graphs -> teacher sequences -> targeted SFT
(warm start) -> Graph-CPO RL updates -> Protocol A evaluation. Uses the dev
config's small models so it completes on a laptop-class machine; every stage
scales via the config file (configs/paper_qwen32b.yaml for the full-scale run).

Usage:
  python -m graph_cpo.scripts.demo_end_to_end graph_cpo/configs/dev_qwen_0.5b.yaml \
      --n_problems 5
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ._common import cfg_from_args, device, load_jsonl, save_jsonl
from graph_cpo.eval.protocol_a import run_protocol_a
from graph_cpo.eval.solver import SolverHarness
from graph_cpo.models.policy import GraphAugmentedPolicy
from graph_cpo.models.rollout import HFRolloutEngine
from graph_cpo.planning.prefix import GraphPrefixProjector
from graph_cpo.planning.teacher import build_teacher_from_silver
from graph_cpo.rewards.bundle import build_reward_bundle
from graph_cpo.sdg.gnn import GraphEncoderBundle
from graph_cpo.sdg.silver_labels import extract_silver_sdg
from graph_cpo.sdg.store import save_graph
from graph_cpo.trainer.rl import GraphCPOTrainer
from graph_cpo.trainer.sft import SFTTrainer
from graph_cpo.trajectory import rollout_with_teacher, sample_trajectories


def main() -> None:
    cfg, argv = cfg_from_args()
    n = int(argv[argv.index("--n_problems") + 1]) if "--n_problems" in argv else 5
    data_path = argv[argv.index("--input") + 1] if "--input" in argv else "data/train.jsonl"
    dev = device()
    print(f"[demo] device={dev} backbone={cfg.model.backbone}")

    rows = load_jsonl(data_path)[:n]
    for i, r in enumerate(rows):
        r["id"] = r.get("id", f"demo{i}")

    # ---- Phase 1: SDGs (rule-based silver extractor) + cache
    sdgs = {}
    for r in rows:
        sdg = extract_silver_sdg(r["id"], r["question"], r.get("answer", ""))
        save_graph(sdg, cfg.sdg.graph_cache)
        sdgs[r["id"]] = sdg
        print(f"[demo] {r['id']}: {len(sdg.nodes)} nodes, {len(sdg.edges)} edges, "
              f"frontier={sdgs[r['id']].frontier() if hasattr(sdg, 'frontier') else 'n/a'}")

    # ---- node embeddings (GNN)
    span_tok = AutoTokenizer.from_pretrained(cfg.sdg.tagger_model)
    gnn = GraphEncoderBundle(cfg.sdg.tagger_model, hidden=cfg.sdg.gnn.hidden,
                             num_layers=cfg.sdg.gnn.layers).to(dev).eval()
    node_emb = {pid: gnn.encode_graph(span_tok, sdg, device=dev)
                for pid, sdg in sdgs.items()}

    # ---- teacher trajectories
    teachers = {}
    for r in rows:
        t = build_teacher_from_silver(r["id"], r["question"], r.get("answer", ""))
        if t is not None and t.steps:
            teachers[r["id"]] = t
    trajectories = [rollout_with_teacher(sdgs[pid], teachers[pid], cfg.planning)
                    for pid in teachers]

    # ---- Phase 2: targeted SFT warm start
    policy = GraphAugmentedPolicy(
        cfg.model.backbone, cfg.model.tokenizer,
        lora_enabled=cfg.model.lora.enabled, lora_r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha, lora_dropout=cfg.model.lora.dropout,
        lora_targets=cfg.model.lora.target_modules,
        graph_prefix_dim=cfg.sdg.prefix_dim, dtype=cfg.model.dtype)
    prefix_proj = GraphPrefixProjector(cfg.sdg.gnn.hidden, cfg.sdg.prefix_dim,
                                       num_prefix_tokens=policy.num_prefix_tokens).to(dev)
    sft = SFTTrainer(policy, cfg.training.sft, prefix_projector=prefix_proj, device=dev)
    sft.train(trajectories, node_embeddings=node_emb)

    # reference policy = warm-start snapshot
    ref_policy = copy.deepcopy(policy).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    # ---- Phase 3: Graph-CPO RL
    engine = HFRolloutEngine(policy, temperature=0.8, device=dev)
    rewards = build_reward_bundle(cfg.rewards, cfg.planning, device=dev)
    trainer = GraphCPOTrainer(policy, ref_policy, engine, rewards, cfg.training.rl,
                              cfg.planning, prefix_projector=prefix_proj,
                              teacher_by_problem=teachers,
                              node_embeddings=node_emb, device=dev)
    for sdg in sdgs.values():
        trainer.update_on_problem(sdg)

    # ---- generation demo: one trajectory per problem
    print("\n[demo] generated Socratic sequences:")
    for pid, sdg in sdgs.items():
        gate = rewards.make_gate(sdg)
        trajs = sample_trajectories(engine, sdg, cfg.planning,
                                    node_emb=node_emb[pid], gate=gate,
                                    num_rollouts=1, max_new_tokens=64,
                                    prefix_projector=prefix_proj)
        print(f"\n[demo] {pid}: {sdg.text[:80]}...")
        for i, q in enumerate(trajs[0].questions, 1):
            print(f"   Q{i}: {q}")

    # ---- Protocol A smoke eval
    problems = []
    for r in rows:
        if r["id"] in sdgs:
            p = dict(r); p["sdg"] = sdgs[r["id"]]
            problems.append(p)
    solver = SolverHarness(cfg.eval.solver_backbone,
                           max_tokens=cfg.eval.solver_max_tokens,
                           timeout_s=cfg.eval.solver_timeout_s, device=dev)

    def generator(sdg):
        gate = rewards.make_gate(sdg)
        return sample_trajectories(engine, sdg, cfg.planning,
                                   node_emb=node_emb[sdg.problem_id], gate=gate,
                                   num_rollouts=1, max_new_tokens=64,
                                   prefix_projector=prefix_proj)[0]

    res = run_protocol_a(problems, solver, generator=generator,
                         nli_model=cfg.rewards.nli_model, device=dev,
                         save_path=str(Path(cfg.output_dir) / "demo_protocol_a.json"))
    print(f"\n[demo] Protocol A: treatment={res.success_treatment:.2f} "
          f"control={res.success_control:.2f} PO={res.purpose_orientation:.2f} "
          f"targeted={res.targeted:.2f}")
    print("[demo] done — full pipeline runs end to end; scale up via config.")


if __name__ == "__main__":
    main()
