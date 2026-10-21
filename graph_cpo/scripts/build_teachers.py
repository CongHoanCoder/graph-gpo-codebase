"""Build teacher Socratic sequences for the targeted-SFT warm start.

Each gold solution step is verbalized as one sub-question aligned to a graph
node (roughly half GSM8K, half MathQA at full scale).

Usage:
  python -m graph_cpo.scripts.build_teachers configs/dev_qwen_0.5b.yaml \
      --input data/train.jsonl --limit 8000
"""
from __future__ import annotations

from pathlib import Path

from ._common import cfg_from_args, load_jsonl
from graph_cpo.planning.teacher import build_teacher_from_silver, write_teachers


def main() -> None:
    cfg, argv = cfg_from_args()
    inp = argv[argv.index("--input") + 1] if "--input" in argv else "data/train.jsonl"
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 8000
    out_dir = Path(cfg.planning.teacher_dir)
    out_file = out_dir / f"{Path(inp).stem}_teachers.jsonl"
    if out_file.exists():
        out_file.unlink()      # rewrite cleanly

    rows = load_jsonl(inp)[:limit]
    written = 0
    for i, r in enumerate(rows):
        pid = r.get("id", f"row{i}")
        traj = build_teacher_from_silver(pid, r["question"], r.get("answer", ""))
        if traj is not None and traj.steps:
            write_teachers(traj, out_file)
            written += 1
    print(f"[teachers] {written} teacher trajectories -> {out_file}")


if __name__ == "__main__":
    main()
