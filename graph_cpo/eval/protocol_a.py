"""Protocol A — direct augmentation evaluation.

For each benchmark problem:
  1. The generator produces a sequence of Socratic sub-questions.
  2. The solver answers with (a) the sub-questions as context [treatment] and
     (b) no sub-questions [control], under fixed token/timeout constraints.
Metrics:
  - Success Rate: % augmented MWPs solved correctly (both questions and answer
    must be valid; used to filter valid augmented pairs).
  - Purpose-Orientation: NLI-based sequential coherence of the sequence.
  - Targeted: BERTScore-style overlap of questions with MWP entities/operators.
Controls: original MWPs without sub-questions; random/non-Socratic questions.
All ablations and both RQs are analyzed under this protocol.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..sdg.schema import SDG
from ..trajectory import Trajectory, sample_trajectories
from .solver import SolverHarness, answer_correct, extract_answer, _gold_from

# Content words (entities/operators) used by the Targeted metric fallback.
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "how", "many", "much", "what",
    "which", "who", "if", "then", "and", "or", "to", "of", "in", "for", "each",
    "there", "this", "that", "her", "his", "their", "he", "she", "they", "it",
    "will", "would", "have", "has", "had", "with", "from", "after", "before",
}


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def targeted_score(question: str, problem_text: str) -> float:
    """Token-overlap with MWP entities/operators (BERTScore stand-in).

    Uses BERTScore with a DeBERTa backbone when the optional `bert-score`
    package is installed; otherwise a content-word F1 (same ranking behavior
    for this metric's purpose).
    """
    try:
        from bert_score import score as _bscore
        P, R, F1 = _bscore([question], [problem_text], lang="en",
                           model_type="microsoft/deberta-xlarge-mnli", verbose=False)
        return float(F1[0])
    except Exception:
        q = _content_words(question)
        p = _content_words(problem_text)
        if not q or not p:
            return 0.0
        overlap = len(q & p)
        precision = overlap / len(q)
        recall = overlap / len(p)
        return 2 * precision * recall / max(precision + recall, 1e-9)


def purpose_orientation(questions: List[str], problem_text: str,
                        nli_model: Optional[str] = None, device: str = "cpu") -> float:
    """NLI-based sequential coherence: does each question follow from the
    problem and the questions asked so far, and advance the final goal?"""
    if not questions:
        return 0.0
    if nli_model is None:
        try:
            from ..rewards.coherence import NLICoherenceReward
            nli = NLICoherenceReward(device=device)
        except Exception:
            return _lexical_coherence(questions, problem_text)
    else:
        from ..rewards.coherence import NLICoherenceReward
        nli = NLICoherenceReward(nli_model, device=device)
    scores = []
    history = ""
    for q in questions:
        premise = f"Problem: {problem_text}\n{history}"
        hypothesis = f"The next logical step asks: {q}"
        p = nli.probabilities(premise, hypothesis)
        scores.append(float(p[nli.entail_id]))
        history += q + " "
    return float(np.mean(scores))


def _lexical_coherence(questions: List[str], problem_text: str) -> float:
    p = _content_words(problem_text)
    vals = []
    for q in questions:
        qw = _content_words(q)
        vals.append(len(qw & p) / max(len(qw), 1))
    return float(np.mean(vals)) if vals else 0.0


@dataclass
class ProtocolAResult:
    num_problems: int
    success_treatment: float        # solved with sub-questions
    success_control: float          # solved without sub-questions
    success_random: float           # solved with random (non-Socratic) questions
    purpose_orientation: float
    targeted: float
    per_problem: List[dict] = field(default_factory=list)


def random_questions(problem: str, k: int = 3, seed: int = 0) -> List[str]:
    """Non-Socratic control: generic questions about the surface text."""
    rng = np.random.RandomState(seed)
    generic = [
        "What is this problem about?",
        "What operations do you see?",
        "Can you solve it?",
        "What numbers appear in the text?",
        "Is the problem about adding or subtracting?",
    ]
    idx = rng.choice(len(generic), size=min(k, len(generic)), replace=False)
    return [generic[i] for i in idx]


def run_protocol_a(problems: List[Dict], solver: SolverHarness,
                   generator: Optional[Callable[[SDG], Trajectory]] = None,
                   nli_model: Optional[str] = None, device: str = "cpu",
                   save_path: Optional[str] = None) -> ProtocolAResult:
    """Full Protocol A evaluation. `generator` maps an SDG to a trajectory."""
    per_problem = []
    succ_t = succ_c = succ_r = 0
    po_scores, tg_scores = [], []
    for item in problems:
        text = item["question"]
        gold = _gold_from(item.get("gold_answer") or item.get("answer"))
        pid = item.get("id", text[:40])
        questions: List[str] = []
        if generator is not None and "sdg" in item:
            traj = generator(item["sdg"])
            questions = traj.questions
        # treatment vs control
        out_t = solver.solve(text, questions)
        ok_t = answer_correct(extract_answer(out_t), gold)
        out_c = solver.solve(text, None)
        ok_c = answer_correct(extract_answer(out_c), gold)
        rnd = random_questions(text, k=len(questions) or 3, seed=hash(pid) % 10000)
        out_r = solver.solve(text, rnd)
        ok_r = answer_correct(extract_answer(out_r), gold)
        succ_t += int(ok_t); succ_c += int(ok_c); succ_r += int(ok_r)
        po = purpose_orientation(questions, text, nli_model=nli_model, device=device)
        tg = float(np.mean([targeted_score(q, text) for q in questions])) if questions else 0.0
        po_scores.append(po); tg_scores.append(tg)
        per_problem.append({"id": pid, "questions": questions,
                            "solved_treatment": ok_t, "solved_control": ok_c,
                            "solved_random": ok_r, "purpose_orientation": po,
                            "targeted": tg})
    result = ProtocolAResult(
        num_problems=len(problems),
        success_treatment=succ_t / max(len(problems), 1),
        success_control=succ_c / max(len(problems), 1),
        success_random=succ_r / max(len(problems), 1),
        purpose_orientation=float(np.mean(po_scores)) if po_scores else 0.0,
        targeted=float(np.mean(tg_scores)) if tg_scores else 0.0,
        per_problem=per_problem)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
    return result
