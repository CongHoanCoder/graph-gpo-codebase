"""Evaluate SDG construction against expert-annotated graphs.

Usage:
  python -m graph_cpo.scripts.eval_graphs configs/dev_qwen_0.5b.yaml \
      --gold data/sdg/expert_graphs.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from ._common import cfg_from_args, load_jsonl
from graph_cpo.eval.graph_quality import evaluate_against_expert
from graph_cpo.sdg.schema import SDG
from graph_cpo.sdg.store import load_graph


def main() -> None:
    cfg, argv = cfg_from_args()
    gold_path = argv[argv.index("--gold") + 1] if "--gold" in argv \
        else "data/sdg/expert_graphs.jsonl"
    golds = [SDG.from_dict(r) for r in load_jsonl(gold_path)]
    preds = []
    for g in golds:
        p = load_graph(cfg.sdg.graph_cache, g.problem_id)
        if p is not None:
            preds.append(p)
    if not preds:
        raise SystemExit("No predicted graphs matched the expert set; "
                         "run build_graphs on the annotated problems first.")
    metrics = evaluate_against_expert(preds, golds)
    out = Path(cfg.output_dir) / "graph_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    for k, v in metrics.items():
        print(f"[graph_quality] {k}: {v}")
    print(f"[graph_quality] saved to {out}")


if __name__ == "__main__":
    main()
