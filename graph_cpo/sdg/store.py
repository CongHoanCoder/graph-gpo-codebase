"""Offline graph cache: graphs are built ahead of time and cached on disk."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .schema import SDG


def _graph_path(cache_dir: str | Path, problem_id: str) -> Path:
    h = hashlib.sha1(problem_id.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{h}_{problem_id}.json"


def save_graph(sdg: SDG, cache_dir: str | Path) -> Path:
    p = _graph_path(cache_dir, sdg.problem_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(sdg.to_dict(), f, ensure_ascii=False)
    return p


def load_graph(cache_dir: str | Path, problem_id: str) -> Optional[SDG]:
    p = _graph_path(cache_dir, problem_id)
    if not p.exists():
        return None
    with open(p) as f:
        return SDG.from_dict(json.load(f))


def graph_exists(cache_dir: str | Path, problem_id: str) -> bool:
    return _graph_path(cache_dir, problem_id).exists()
