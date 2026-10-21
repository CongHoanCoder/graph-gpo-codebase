"""Train the relation-aware GNN + biaffine relation classifier on silver graphs.

Jointly learns (a) GNN node embeddings h^(L) and (b) edge-type prediction, so
the same GNN supplies graph prefixes and reward embeddings downstream.

Usage:
  python -m graph_cpo.scripts.train_relf configs/dev_qwen_0.5b.yaml [--epochs 3]
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ._common import cfg_from_args, device, load_jsonl
from graph_cpo.sdg.gnn import GraphEncoderBundle, numeric_features, RelationGNN
from graph_cpo.sdg.rel_classifier import LABELS, L2I, BiaffineClassifier, candidate_pairs
from graph_cpo.sdg.schema import EDGE_TYPES, SDG
from transformers import AutoTokenizer


def _pair_features(bundle: GraphEncoderBundle, tokenizer, sdg: SDG, dev: str):
    """Node embeddings for one silver graph via the span encoder + GNN."""
    return bundle.encode_graph(tokenizer, sdg, device=dev)


def main() -> None:
    cfg, argv = cfg_from_args()
    epochs = int(argv[argv.index("--epochs") + 1]) if "--epochs" in argv else 3
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 8000
    dev = device()

    silver_files = sorted(Path(cfg.sdg.silver_dir).glob("*_silver.jsonl"))
    if not silver_files:
        raise SystemExit(f"No silver graphs in {cfg.sdg.silver_dir}; "
                         "run make_silver_labels first.")
    graphs = []
    for f in silver_files:
        for row in load_jsonl(f):
            graphs.append(SDG.from_dict(row))
    graphs = graphs[:limit]

    tokenizer = AutoTokenizer.from_pretrained(cfg.sdg.tagger_model)
    bundle = GraphEncoderBundle(
        cfg.sdg.tagger_model, hidden=cfg.sdg.gnn.hidden,
        num_layers=cfg.sdg.gnn.layers).to(dev)
    classifier = BiaffineClassifier(cfg.sdg.rel_classifier_hidden).to(dev)

    params = list(bundle.gnn.parameters()) + list(classifier.parameters())
    opt = torch.optim.AdamW(params, lr=2e-5)

    bundle.train(); classifier.train()
    for epoch in range(epochs):
        total, correct, count = 0.0, 0, 0
        for sdg in graphs:
            if not sdg.nodes or not sdg.edges:
                continue
            node_emb = _pair_features(bundle, tokenizer, sdg, dev)
            pairs = candidate_pairs(sdg)
            if not pairs:
                continue
            gold_edge = {(e.src, e.dst): e.type for e in sdg.edges}
            h_src = torch.stack([node_emb[a] for a, _ in pairs])
            h_dst = torch.stack([node_emb[b] for _, b in pairs])
            scores = classifier(h_src, h_dst)
            labels = torch.tensor(
                [L2I[gold_edge.get((a, b), "null")] for a, b in pairs]).to(dev)
            loss = F.cross_entropy(scores, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            correct += int((scores.argmax(-1) == labels).sum())
            count += len(pairs)
        print(f"[relf] epoch {epoch+1}/{epochs} loss={total / max(count, 1):.4f} "
              f"pair_acc={correct / max(count, 1):.3f}")

    out_dir = Path(cfg.output_dir) / "sdg_relf"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.gnn.state_dict(), out_dir / "gnn.pt")
    torch.save(classifier.state_dict(), out_dir / "biaffine.pt")
    print(f"[relf] saved to {out_dir}")


if __name__ == "__main__":
    main()
