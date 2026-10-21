"""Sequence validity (purpose orientation of generated question sequences).

A generated sequence counts as valid iff it is logically sound and solvable by
a strong teacher model under strict reasoning constraints: the teacher must
reach the correct final answer using ONLY the sub-questions as scaffolding,
and every question must pass the graph resolution gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from ..sdg.schema import SDG
from ..trajectory import Trajectory
from .solver import SolverHarness, answer_correct, extract_answer, _gold_from


@dataclass
class ValidityResult:
    num_sequences: int
    validity_rate: float
    per_problem: List[dict]


def sequence_validity(problems: List[Dict], teacher: SolverHarness,
                      generator: Callable[[SDG], Trajectory],
                      require_gates: bool = True) -> ValidityResult:
    per = []
    valid = 0
    for item in problems:
        sdg: Optional[SDG] = item.get("sdg")
        if sdg is None:
            continue
        traj = generator(sdg)
        questions = traj.questions
        gold = _gold_from(item.get("gold_answer") or item.get("answer"))
        # 1) teacher solves using only the questions as scaffolding
        out = teacher.solve(item["question"], questions)
        solvable = answer_correct(extract_answer(out), gold)
        # 2) structural soundness: every question is non-empty, interrogative,
        #    and (optionally) resolved its graph target
        sound = all(q.strip().endswith("?") and len(q.strip()) > 8 for q in questions)
        if require_gates:
            sound = sound and all(ts.resolved for ts in traj.timesteps)
        ok = bool(solvable and sound and questions)
        valid += int(ok)
        per.append({"id": item.get("id"), "valid": ok, "solvable": solvable,
                    "sound": sound, "num_questions": len(questions)})
    n = len(per)
    return ValidityResult(num_sequences=n, validity_rate=valid / max(n, 1), per_problem=per)
