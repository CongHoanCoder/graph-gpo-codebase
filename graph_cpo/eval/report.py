"""Render result tables from evaluation JSON results: the headline Protocol B
benchmark comparison, the RQ1 Socratic comparison, the reward ablation, the
generator-validity table, and the cost table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def _fmt(x: Optional[float], pct: bool = False) -> str:
    if x is None:
        return "--"
    return f"{x*100:.2f}" if pct else f"{x:.2f}"


def benchmark_table(results: Dict[str, dict], out_path: Optional[str] = None) -> str:
    """Protocol B comparison table.

    results: {method_name: {"per_benchmark": {...}, "mean_accuracy": ...}}
    Superscript deltas are computed vs. the backbone baseline (first entry).
    """
    benchmarks: List[str] = []
    for r in results.values():
        for b in r.get("per_benchmark", {}):
            if b not in benchmarks:
                benchmarks.append(b)
    names = list(results.keys())
    base = results[names[0]].get("per_benchmark", {})
    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * (len(benchmarks) + 1) + "}",
        r"\toprule",
        " & " + " & ".join(f"\\textbf{{{b}}}" for b in benchmarks) + r" & \textbf{Avg.} \\",
        r"\midrule",
    ]
    for name in names:
        r = results[name]
        cells = []
        for b in benchmarks:
            v = r["per_benchmark"].get(b)
            delta = None
            if name != names[0] and v is not None and base.get(b) is not None:
                delta = (v - base[b]) * 100
            cell = _fmt(v)
            if delta is not None and abs(delta) >= 0.005:
                cell += rf"\improve{{{delta:.2f}}}" if delta > 0 else rf"\decline{{{abs(delta):.2f}}}"
            cells.append(cell)
        avg = r.get("mean_accuracy")
        avg_delta = None
        if name != names[0] and avg is not None and results[names[0]].get("mean_accuracy") is not None:
            avg_delta = (avg - results[names[0]]["mean_accuracy"]) * 100
        avg_cell = _fmt(avg)
        if avg_delta is not None:
            avg_cell += rf"\improve{{{avg_delta:.2f}}}" if avg_delta > 0 else rf"\decline{{{abs(avg_delta):.2f}}}"
        bold = r"\textbf{" + name + "}" if "Graph-CPO" in name or "graph_cpo" in name else name
        lines.append(f"{bold} & " + " & ".join(cells) + f" & {avg_cell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    tex = "\n".join(lines)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(tex)
    return tex


def reward_ablation_table(rows: List[dict], out_path: Optional[str] = None) -> str:
    """Ablation of reward dimensions."""
    lines = [
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"\textbf{Variant} & $R_{\mathrm{rel}}$ & $R_{\mathrm{coh}}$ & "
        r"$R_{\mathrm{form}}$ & DW & \textbf{MathQA} \\",
        r"\midrule",
    ]
    for r in rows:
        mark = lambda b: r"\checkmark" if b else "--"
        name = r["variant"]
        if r.get("full"):
            name = r"\textbf{" + name + "}"
        lines.append(f"{name} & {mark(r['rel'])} & {mark(r['coh'])} & "
                     f"{mark(r['form'])} & {mark(r['dw'])} & {_fmt(r['accuracy'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    if out_path:
        Path(out_path).write_text(tex)
    return tex


def validity_table(rows: Dict[str, float], out_path: Optional[str] = None) -> str:
    """Generator sequence validity rates."""
    lines = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lc}",
             r"\toprule", r"\textbf{Generator} & \textbf{Validity (\%)} \\", r"\midrule"]
    for name, v in rows.items():
        bold = r"\textbf{" + name + "}" if "Graph-CPO" in name or "graph_cpo" in name else name
        lines.append(f"{bold} & {v:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    tex = "\n".join(lines)
    if out_path:
        Path(out_path).write_text(tex)
    return tex


def cost_table(rows: List[dict], out_path: Optional[str] = None) -> str:
    """Computational cost comparison."""
    lines = [
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"\textbf{Method} & \textbf{Time/ep.} & \textbf{GPU-h} & "
        r"\textbf{Lat. (s)} & \textbf{Mem. (GB)} \\", r"\midrule",
    ]
    for r in rows:
        lines.append(f"{r['method']} & {r['time_per_epoch_h']}h & {r['gpu_h']} & "
                     f"{r['latency_s']} & {r['peak_mem_gb']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    if out_path:
        Path(out_path).write_text(tex)
    return tex


def training_curves(data: Dict[str, List[float]], out_path: str) -> None:
    """Per-dimension reward curves for the reward-collapse analysis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] matplotlib unavailable; dumping JSON instead")
        Path(out_path).with_suffix(".json").write_text(json.dumps(data))
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key in zip(axes, ["rel", "coh", "form"]):
        for run_name, series in data.items():
            if key in series:
                ax.plot(series.get("step", range(len(series[key]))), series[key],
                        label=run_name)
        ax.set_title(key)
        ax.set_xlabel("update step")
        ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
