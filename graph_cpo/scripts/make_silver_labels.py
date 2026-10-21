"""Build silver SDG labels from GSM8K (and optionally MathQA) solution steps.

Usage:
  python -m graph_cpo.scripts.make_silver_labels configs/dev_qwen_0.5b.yaml \
      --input data/train.jsonl --limit 12000

Writes one JSON SDG per problem into config.sdg.silver_dir. These silver labels
train the tagger and biaffine classifier; the 1,000 expert-annotated graphs are
held out separately (see export_annotation_template.py).
"""
from __future__ import annotations

from pathlib import Path

from ._common import cfg_from_args, load_jsonl, save_jsonl
from graph_cpo.sdg.silver_labels import extract_silver_sdg


def main() -> None:
    cfg, argv = cfg_from_args()
    inp = argv[argv.index("--input") + 1] if "--input" in argv else "data/train.jsonl"
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 12000
    out_dir = Path(cfg.sdg.silver_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(inp)[:limit]
    records = []
    for i, r in enumerate(rows):
        pid = r.get("id", f"row{i}")
        sdg = extract_silver_sdg(pid, r["question"], r.get("answer", ""))
        d = sdg.to_dict()
        d["source"] = Path(inp).name
        records.append(d)
    out_file = out_dir / f"{Path(inp).stem}_silver.jsonl"
    save_jsonl(records, out_file)
    n_edges = sum(len(r["edges"]) for r in records)
    print(f"[silver] {len(records)} graphs -> {out_file} "
          f"(avg {n_edges / max(len(records), 1):.1f} edges/graph)")


if __name__ == "__main__":
    main()
