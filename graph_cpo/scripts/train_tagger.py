"""Train the BERT span tagger on silver SDG labels.

Usage:
  python -m graph_cpo.scripts.train_tagger configs/dev_qwen_0.5b.yaml \
      [--epochs 3] [--limit 8000]
"""
from __future__ import annotations

from pathlib import Path

import torch

from ._common import cfg_from_args, device, load_jsonl
from graph_cpo.sdg.schema import SDG
from graph_cpo.sdg.tagger import (BIO2ID, SDGTagger, bio_tags_for_spans,
                                  silver_spans_from_sdg, _word_spans)
from transformers import AutoTokenizer


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
    model = SDGTagger(cfg.sdg.tagger_model).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)

    model.train()
    for epoch in range(epochs):
        total, count = 0.0, 0
        for sdg in graphs:
            words = _word_spans(sdg.text)
            if not words:
                continue
            spans = silver_spans_from_sdg(sdg)
            tags = bio_tags_for_spans(len(words), spans, words)
            # word -> wordpiece ids with word index mapping
            word_ids, input_ids = [], []
            for wi, (ws, we) in enumerate(words):
                sub = tokenizer(sdg.text[ws:we], add_special_tokens=False)["input_ids"]
                input_ids.extend(sub)
                word_ids.extend([wi] * len(sub))
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            word_ids = [-1] + word_ids + [-1]
            labels = [BIO2ID["O"] if wi < 0 else BIO2ID[tags[wi]] for wi in word_ids]
            ids = torch.tensor([input_ids]).to(dev)
            lab = torch.tensor([labels]).to(dev)
            mask = torch.ones_like(ids)
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            total += float(out["loss"].detach())
            count += 1
        print(f"[tagger] epoch {epoch+1}/{epochs} loss={total / max(count, 1):.4f}")

    out_dir = Path(cfg.output_dir) / "sdg_tagger"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "tagger.pt")
    tokenizer.save_pretrained(out_dir)
    print(f"[tagger] saved to {out_dir}")


if __name__ == "__main__":
    main()
