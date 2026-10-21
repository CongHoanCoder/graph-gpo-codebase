"""Logical coherence reward (NLI-based):

R_coh(q_t, v_t) = P_NLI(entail | psi_t, d(v_t)) - P_NLI(contradict | psi_t, d(v_t))
psi_t = [P; d(Pre(v_t)); d_hat(q_t)]

d_hat(q_t) is the declarative intent of the generated question (extracted by a
small instruction prompt to a cheap LM, with a regex fallback). The judge is
DeBERTa-v3-large-mnli at full scale (architecturally distinct from the Qwen
generator, reducing self-rewarding bias); the dev config uses a smaller NLI
model.
"""
from __future__ import annotations

import re
from typing import List, Optional

import torch

from ..planning.action_frame import verbalize_node  # noqa: F401 (d(v) helper available)


class DeclarativeIntentExtractor:
    """Extract d_hat(q): what the question asks for, as a declarative statement."""

    _PATTERN = re.compile(
        r"(?:what|how many|how much|which|who|when|where)\b[^?]*", re.IGNORECASE)

    def __call__(self, question: str) -> str:
        m = self._PATTERN.search(question)
        fragment = m.group(0).strip() if m else question.strip()
        fragment = fragment.rstrip("?").strip()
        return f"The question asks for {fragment}"


class NLICoherenceReward:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-base",
                 device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False
        # HF NLI checkpoints use id2label; find entail/contradict indices
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_id = next(i for i, l in id2label.items() if "entail" in l)
        self.contradict_id = next(i for i, l in id2label.items() if "contra" in l)
        self.intent = DeclarativeIntentExtractor()

    @torch.no_grad()
    def probabilities(self, premise: str, hypothesis: str) -> torch.Tensor:
        enc = self.tokenizer(premise, hypothesis, return_tensors="pt",
                             truncation=True, max_length=256).to(self.device)
        logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1)[0]

    def __call__(self, question: str, sdg, target_node: str) -> float:
        node = sdg.node(target_node)
        d_hat = self.intent(question)
        pre_texts = [sdg.node(p).text for p in sdg.prerequisites(target_node)
                     if p in sdg.node_ids()]
        premise = (f"Problem: {sdg.text}\n"
                   f"Prerequisite facts: {'; '.join(pre_texts) if pre_texts else 'none'}\n"
                   f"{d_hat}.")
        hypothesis = _dependency_text(node)
        p = self.probabilities(premise, hypothesis)
        return float(p[self.entail_id] - p[self.contradict_id])

    def passes_gate(self, question: str, sdg, target_node: str,
                    tau_nli: float = 0.6) -> bool:
        """Entailment-probability gate used for node resolution."""
        node = sdg.node(target_node)
        d_hat = self.intent(question)
        premise = f"Problem: {sdg.text}\n{d_hat}."
        p = self.probabilities(premise, _dependency_text(node))
        return float(p[self.entail_id]) >= tau_nli


def _dependency_text(node) -> str:
    return (f"The question addresses the {node.type} '{node.text}' "
            f"and helps derive it")
