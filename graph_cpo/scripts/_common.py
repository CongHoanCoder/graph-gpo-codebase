"""Shared script helpers: config, data loading, component construction."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# make graph_cpo importable when running scripts directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graph_cpo.config import Config, load_config  # noqa: E402


def cfg_from_args(argv: Optional[List[str]] = None) -> tuple[Config, List[str]]:
    """First CLI arg is the config path; remaining args are script-specific."""
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = argv[0] if argv and argv[0].endswith((".yaml", ".yml")) \
        else "graph_cpo/configs/dev_qwen_0.5b.yaml"
    if argv and argv[0].endswith((".yaml", ".yml")):
        argv = argv[1:]
    return load_config(config_path), argv


def load_jsonl(path: str | Path) -> List[Dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(rows: List[Dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def attach_sdg_to_problems(problems: List[Dict], cache_dir: str) -> List[Dict]:
    """Attach cached SDGs (keyed by problem id) to problem dicts, skipping misses."""
    from graph_cpo.sdg.store import load_graph
    out = []
    for p in problems:
        pid = p.get("id") or p["question"][:40]
        sdg = load_graph(cache_dir, pid)
        if sdg is not None:
            p = dict(p)
            p["id"] = pid
            p["sdg"] = sdg
            out.append(p)
    return out


def compute_node_embeddings(sdg, tagger_tokenizer, bundle, device_str: str
                            ) -> Dict[str, torch.Tensor]:
    """Node embeddings h^(L) via the GNN bundle (or span-only if bundle is None)."""
    if bundle is None:
        return {}
    bundle = bundle.to(device_str)
    return bundle.encode_graph(tagger_tokenizer, sdg, device=device_str)
