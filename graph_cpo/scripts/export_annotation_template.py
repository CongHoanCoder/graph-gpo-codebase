"""Export an annotation template for expert SDG evaluation.

Samples MWPs and writes a JSONL template that expert annotators fill with node
spans/types and typed edges (target: 1,000 graphs; held out from all
training). The resulting annotated file is consumed by
graph_cpo/eval/graph_quality.py-style metrics in scripts/eval_graphs usage
inside tests and eval tooling.

Usage:
  python -m graph_cpo.scripts.export_annotation_template \
      --input data/test.jsonl --output data/sdg/expert_graphs_template.jsonl \
      --n 1000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/test.jsonl")
    ap.add_argument("--output", default="data/sdg/expert_graphs_template.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    random.Random(args.seed).sample(rows, min(args.n, len(rows)))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for i, r in enumerate(rows[: args.n]):
            out.write(json.dumps({
                "problem_id": r.get("id", f"ann{i}"),
                "question": r["question"],
                "annotator": None,
                # fill these:
                "nodes": [],   # [{"id", "type", "text", "start", "end"}]
                "edges": [],   # [{"src", "dst", "type"}]
                "topological_order": [],
            }, ensure_ascii=False) + "\n")
    print(f"[annotation] {min(args.n, len(rows))} templates -> {args.output}")
    print("[annotation] node types: quantity, entity, variable, operation, condition, goal")
    print("[annotation] edge types: quantity_of, op_input, derive, condition_edge, "
          "prerequisite, goal_dep")


if __name__ == "__main__":
    main()
