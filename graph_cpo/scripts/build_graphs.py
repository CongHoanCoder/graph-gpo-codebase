"""Build and cache SDGs for a dataset split.

Priority:
  1. Trained tagger + GNN + biaffine pipeline (if checkpoints exist in
     <output_dir>/sdg_tagger and <output_dir>/sdg_relf).
  2. Rule-based silver extractor (works out of the box; same code that
     produced the silver training labels).

Usage:
  python -m graph_cpo.scripts.build_graphs configs/dev_qwen_0.5b.yaml \
      --input data/train.jsonl --split train [--limit 2000]
"""
from __future__ import annotations

from pathlib import Path

import torch

from ._common import cfg_from_args, device, load_jsonl
from graph_cpo.sdg.dag import to_dag
from graph_cpo.sdg.gnn import GraphEncoderBundle
from graph_cpo.sdg.rel_classifier import BiaffineClassifier, predict_edges
from graph_cpo.sdg.schema import SDG, SDGNode
from graph_cpo.sdg.silver_labels import extract_silver_sdg
from graph_cpo.sdg.store import graph_exists, save_graph
from graph_cpo.sdg.tagger import SDGTagger, predict_spans, spans_to_nodes
from transformers import AutoTokenizer


def build_one(problem: dict, cfg, tagger, tagger_tok, bundle, classifier, dev: str) -> SDG:
    pid = problem.get("id", problem["question"][:40])
    text = problem["question"]
    if tagger is None:
        return extract_silver_sdg(pid, text, problem.get("answer", ""))
    spans = predict_spans(tagger, tagger_tok, text, device=dev)
    nodes = spans_to_nodes(spans)
    if not nodes:
        return extract_silver_sdg(pid, text, problem.get("answer", ""))
    sdg = SDG(problem_id=pid, text=text, gold_answer=problem.get("answer"))
    sdg.nodes = nodes
    node_emb = bundle.encode_graph(tagger_tok, sdg, device=dev)
    sdg.edges = predict_edges(classifier, node_emb, sdg,
                              min_conf=cfg.sdg.min_edge_conf, device=dev)
    # anchor the goal: connect the last variable/operation to the goal if none
    if sdg.goal_nodes() and not any(e.type == "goal_dep" for e in sdg.edges):
        tail = [n for n in sdg.nodes if n.type in ("variable", "operation")]
        if tail:
            from graph_cpo.sdg.schema import SDGEdge
            sdg.edges.append(SDGEdge(src=tail[-1].id, dst=sdg.goal_nodes()[0].id,
                                     type="goal_dep", conf=0.9))
    return to_dag(sdg)


def main() -> None:
    cfg, argv = cfg_from_args()
    inp = argv[argv.index("--input") + 1] if "--input" in argv else "data/train.jsonl"
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 2000
    dev = device()

    tagger = tagger_tok = bundle = classifier = None
    tagger_dir = Path(cfg.output_dir) / "sdg_tagger"
    relf_dir = Path(cfg.output_dir) / "sdg_relf"
    if (tagger_dir / "tagger.pt").exists() and (relf_dir / "gnn.pt").exists():
        tagger_tok = AutoTokenizer.from_pretrained(cfg.sdg.tagger_model)
        tagger = SDGTagger(cfg.sdg.tagger_model)
        tagger.load_state_dict(torch.load(tagger_dir / "tagger.pt"))
        tagger.to(dev).eval()
        bundle = GraphEncoderBundle(cfg.sdg.tagger_model, hidden=cfg.sdg.gnn.hidden,
                                    num_layers=cfg.sdg.gnn.layers)
        bundle.gnn.load_state_dict(torch.load(relf_dir / "gnn.pt"))
        bundle.to(dev).eval()
        classifier = BiaffineClassifier(cfg.sdg.rel_classifier_hidden)
        classifier.load_state_dict(torch.load(relf_dir / "biaffine.pt"))
        classifier.to(dev).eval()
        print("[build_graphs] using trained tagger + GNN + biaffine pipeline")
    else:
        print("[build_graphs] no trained SDG checkpoints; using rule-based extractor")

    rows = load_jsonl(inp)[:limit]
    built = skipped = 0
    for row in rows:
        pid = row.get("id", row["question"][:40])
        row["id"] = pid
        if graph_exists(cfg.sdg.graph_cache, pid):
            skipped += 1
            continue
        sdg = build_one(row, cfg, tagger, tagger_tok, bundle, classifier, dev)
        save_graph(sdg, cfg.sdg.graph_cache)
        built += 1
    print(f"[build_graphs] built={built} cached(skip)={skipped} -> {cfg.sdg.graph_cache}")


if __name__ == "__main__":
    main()
