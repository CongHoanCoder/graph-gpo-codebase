"""BERT sequence tagger for SDG node extraction.

Fine-tunes a BERT token-classification head on silver labels to extract typed
spans (Quantity / Entity / Variable / Operation / Condition / Goal). BIO tagging
over character-offset-mapped wordpieces.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .schema import NODE_TYPES, SDGNode

BIO_LABELS = ["O"] + [f"B-{t}" for t in NODE_TYPES] + [f"I-{t}" for t in NODE_TYPES]
BIO2ID = {l: i for i, l in enumerate(BIO_LABELS)}


@dataclass
class Span:
    start: int          # char offset in original text
    end: int
    type: str
    text: str


class SDGTagger(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", freeze_encoder: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, len(BIO_LABELS))

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}


def _word_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def bio_tags_for_spans(n_words: int, spans: List[Span], words: List[Tuple[int, int]]) -> List[str]:
    tags = ["O"] * n_words
    for sp in spans:
        for i, (ws, we) in enumerate(words):
            if sp.start < we and ws < sp.end:      # overlap
                tags[i] = f"B-{sp.type}" if tags[i] == "O" or not tags[i].endswith(sp.type) else f"I-{sp.type}"
    return tags


def silver_spans_from_sdg(sdg) -> List[Span]:
    """Derive gold char spans for tagger training from a silver SDG."""
    spans = []
    for n in sdg.nodes:
        idx = sdg.text.find(n.text[:20] if len(n.text) > 20 else n.text)
        if n.start >= 0 and n.end > n.start:
            spans.append(Span(n.start, n.end, n.type, n.text))
        elif idx >= 0:
            spans.append(Span(idx, idx + len(n.text[:20]), n.type, n.text[:20]))
    return spans


@torch.no_grad()
def predict_spans(tagger: SDGTagger, tokenizer, text: str, device="cpu",
                  max_length: int = 256) -> List[Span]:
    """Run BIO decoding and return char-offset spans."""
    words = _word_spans(text)
    word_tokens, word_ids = [], []
    for wi, (ws, we) in enumerate(words):
        sub = tokenizer(text[ws:we], add_special_tokens=False)["input_ids"]
        word_tokens.extend(sub)
        word_ids.extend([wi] * len(sub))
    word_tokens = word_tokens[: max_length - 2]
    word_ids = word_ids[: max_length - 2]
    ids = tokenizer.build_inputs_with_special_tokens(word_tokens)
    wid = [-1] + word_ids + [-1]
    mask = [1] * len(ids)
    out = tagger(input_ids=torch.tensor([ids]).to(device),
                 attention_mask=torch.tensor([mask]).to(device))
    pred = out["logits"].argmax(-1)[0].tolist()
    spans: List[Span] = []
    cur_type, cur_start, cur_end = None, -1, -1
    for tok_i, wi in enumerate(wid):
        label = BIO_LABELS[pred[tok_i]] if wi >= 0 else "O"
        if label.startswith("B-"):
            if cur_type is not None:
                spans.append(_mk_span(text, words, cur_start, cur_end, cur_type))
            cur_type, cur_start, cur_end = label[2:], wi, wi
        elif label.startswith("I-") and cur_type == label[2:]:
            cur_end = wi
        elif cur_type is not None:
            spans.append(_mk_span(text, words, cur_start, cur_end, cur_type))
            cur_type = None
    if cur_type is not None:
        spans.append(_mk_span(text, words, cur_start, cur_end, cur_type))
    return spans


def _mk_span(text, words, wi_start, wi_end, ntype) -> Span:
    start = words[wi_start][0]
    end = words[wi_end][1]
    return Span(start, end, ntype, text[start:end])


def spans_to_nodes(spans: List[Span], id_prefix: str = "n") -> List[SDGNode]:
    nodes = []
    for i, sp in enumerate(spans):
        num = None
        if sp.type == "quantity":
            m = re.search(r"[\d,]+\.?\d*", sp.text)
            if m:
                num = float(m.group(0).replace(",", ""))
        nodes.append(SDGNode(id=f"{id_prefix}{i}", type=sp.type, text=sp.text,
                             start=sp.start, end=sp.end, num_value=num))
    return nodes
