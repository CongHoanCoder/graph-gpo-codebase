"""Format adherence reward:

R_form(q_t) = 1{ q_t has valid tags }

Strict binary check for parseable, tagged output — needed for stable parsing
during RL; intentionally down-weighted by the curriculum.
"""
from __future__ import annotations

import re

_OPEN = re.compile(r"<question>")
_CLOSE = re.compile(r"</question>")

# Tolerated fallback: a single interrogative sentence (for warm-up stages where
# the policy has not yet learned the XML-style tags). Set strict=False to allow.
_FALLBACK_Q = re.compile(r"\?\s*$", re.MULTILINE)


def format_reward(question: str, strict: bool = True) -> float:
    text = question.strip()
    if strict:
        if not _OPEN.search(text) or not _CLOSE.search(text):
            return 0.0
        # exactly one well-formed tagged span, tag in order
        m = re.fullmatch(r"\s*<question>(.*)</question>\s*", text, re.DOTALL)
        return 1.0 if m and "?" in m.group(1) else 0.0
    return 1.0 if _FALLBACK_Q.search(text) else 0.0


def extract_question(text: str) -> str:
    """Strip tags/downstream chatter, returning the inner question text."""
    m = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


class FormatReward:
    def __init__(self, strict: bool = True):
        self.strict = strict

    def __call__(self, question: str) -> float:
        return format_reward(question, strict=self.strict)
