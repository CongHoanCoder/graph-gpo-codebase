"""Compute-cost accounting: training time, GPU-hours, latency, peak memory.

GPU-h = wall-clock time x number of GPUs; Lat. = average generation time per
problem; Mem. = peak allocated GPU memory.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class CostMeter:
    num_gpus: int = 1
    epochs: List[float] = field(default_factory=list)      # hours per epoch
    latencies_ms: List[float] = field(default_factory=list)
    peak_mem_gb: float = 0.0

    @contextmanager
    def epoch(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        yield self
        self.epochs.append((time.time() - start) / 3600.0)
        if torch.cuda.is_available():
            self.peak_mem_gb = max(self.peak_mem_gb,
                                   torch.cuda.max_memory_allocated() / 2**30)

    def record_latency(self, seconds: float) -> None:
        self.latencies_ms.append(seconds * 1000.0)

    @property
    def gpu_hours(self) -> float:
        return sum(self.epochs) * self.num_gpus

    @property
    def mean_epoch_hours(self) -> float:
        return sum(self.epochs) / len(self.epochs) if self.epochs else 0.0

    @property
    def mean_latency_s(self) -> float:
        return (sum(self.latencies_ms) / len(self.latencies_ms) / 1000.0
                if self.latencies_ms else 0.0)

    def summary(self, method: str) -> dict:
        return {
            "method": method,
            "time_per_epoch_h": round(self.mean_epoch_hours, 3),
            "gpu_h": round(self.gpu_hours, 1),
            "latency_s": round(self.mean_latency_s, 3),
            "peak_mem_gb": round(self.peak_mem_gb, 2),
            "num_gpus": self.num_gpus,
        }


def compare(rows: List[dict]) -> str:
    """Render a cost-comparison markdown table."""
    header = "| Method | Time/ep. | GPU-h | Lat. (s) | Mem. (GB) |\n|---|---|---|---|---|"
    body = "\n".join(
        f"| {r['method']} | {r['time_per_epoch_h']}h | {r['gpu_h']} | "
        f"{r['latency_s']} | {r['peak_mem_gb']} |" for r in rows)
    return header + "\n" + body
