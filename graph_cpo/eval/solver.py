"""Solver harness for downstream utility evaluation (Protocols A/B).

The solver receives the MWP plus, optionally, the Socratic sub-questions as
context (chain-of-thought scaffolding) and must produce a final numeric answer
under a token limit and wall-clock timeout.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from ..models.policy import from_dtype_arg

SOLVE_TEMPLATE = """Solve the math word problem step by step.

Problem: {problem}
{scaffold}Show your final answer on the last line in the form: #### <number>"""

SCAFFOLD = "Useful sub-questions to guide your reasoning:\n{questions}\n\n"


def build_solve_prompt(problem: str, sub_questions: Optional[List[str]] = None) -> str:
    scaffold = ""
    if sub_questions:
        scaffold = SCAFFOLD.format(
            questions="\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions)))
    return SOLVE_TEMPLATE.format(problem=problem, scaffold=scaffold)


_ANS_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_BARE_RE = re.compile(r"(?:answer is|equals|=)\s*\**(-?[\d,]+\.?\d*)\**\s*\.?\s*$",
                      re.IGNORECASE | re.MULTILINE)


def extract_answer(text: str) -> Optional[str]:
    matches = _ANS_RE.findall(text)
    if matches:
        return _norm(matches[-1])
    matches = _BARE_RE.findall(text)
    if matches:
        return _norm(matches[-1])
    m = re.findall(r"-?[\d,]+\.?\d*", text)
    return _norm(m[-1]) if m else None


def _norm(s: str) -> str:
    try:
        f = float(s.replace(",", ""))
        if f == int(f):
            return str(int(f))
        return f"{f:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return s


def answer_correct(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred.replace(",", "")) - float(str(gold).replace(",", ""))) < 1e-4
    except ValueError:
        return pred.strip() == str(gold).strip()


class SolverHarness:
    """Wraps the downstream solver (e.g. Qwen3-32B or DeepSeek-R1-Distill-Llama-8B)."""

    def __init__(self, backbone: str, max_tokens: int = 512, timeout_s: int = 60,
                 device: str = "cpu", dtype: str = "float32"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone, **from_dtype_arg(getattr(torch, dtype))).to(device).eval()
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.device = device

    @torch.no_grad()
    def solve(self, problem: str, sub_questions: Optional[List[str]] = None) -> str:
        prompt = build_solve_prompt(problem, sub_questions)
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start = time.time()
        out = self.model.generate(
            **ids, max_new_tokens=self.max_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id)
        gen = out[0][ids["input_ids"].size(1):]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        if time.time() - start > self.timeout_s:
            return ""      # treat timeout as failure (Protocol A constraints)
        return text

    def accuracy(self, problems: List[dict], sub_questions_by_id: Optional[dict] = None,
                 gold_by_id: Optional[dict] = None) -> float:
        """Fraction solved correctly (with or without sub-question scaffolding)."""
        correct = 0
        for p in problems:
            pid = p.get("id", p.get("question", "")[:40])
            sq = (sub_questions_by_id or {}).get(pid)
            gold = (gold_by_id or {}).get(pid, p.get("gold_answer") or p.get("answer"))
            out = self.solve(p["question"], sq)
            correct += int(answer_correct(extract_answer(out), _gold_from(gold)))
        return correct / max(len(problems), 1)


def _gold_from(gold) -> Optional[str]:
    """GSM8K gold answers come as '... #### 42'; extract the number."""
    if gold is None:
        return None
    s = str(gold)
    m = _ANS_RE.search(s)
    return _norm(m.group(1)) if m else _norm(s)
