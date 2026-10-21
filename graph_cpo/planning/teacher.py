"""Teacher Socratic sequences for the targeted-SFT warm start.

Target scale: 8,000 MWPs (4,000 GSM8K + 4,000 MathQA), each gold solution
step is verbalized as one Socratic sub-question aligned to the corresponding
graph node, yielding one teacher trajectory per problem.

Input format (JSONL): {"id", "question", "solution", "socratic" (optional)}
If "socratic" is absent, we derive questions from the solution steps with
templates + the silver graph's node order.

Teacher questions are wrapped in the <question>...</question> tags the policy
is trained to emit (the format reward scores raw tagged generations).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ..sdg.silver_labels import extract_silver_sdg
from ..sdg.schema import SDG
from ..models.policy import QUESTION_END, QUESTION_TAG

_STEP_RE = re.compile(r"<<[^=<>]+=([-\d\.,]+)>>")
_ANSWER_RE = re.compile(r"####\s*([-\d\.,]+)")


@dataclass
class TeacherStep:
    question: str           # one Socratic sub-question
    target_node: str        # graph node id it addresses
    answer: str = ""        # the step's gold value (never shown to the student)


@dataclass
class TeacherTrajectory:
    problem_id: str
    steps: List[TeacherStep] = field(default_factory=list)


# Question templates keyed by node type.
_TEMPLATES = {
    "quantity": "What is the value of {text}?",
    "operation": "What calculation ({text}) should be performed here?",
    "condition": "How does the condition '{text}' affect the problem?",
    "variable": "After this step, what is {text}?",
    "goal": "What is the final answer to: {text}?",
}


def build_teacher_from_silver(problem_id: str, question: str, solution: str) -> Optional[TeacherTrajectory]:
    """Verbalize each gold solution step as a sub-question aligned to a graph node."""
    sdg = extract_silver_sdg(problem_id, question, solution)
    traj = TeacherTrajectory(problem_id=problem_id)
    # align solution-step results to derived variable nodes in order
    var_nodes = [n for n in sdg.nodes if n.type == "variable"]
    results = _STEP_RE.findall(solution)
    for node, res in zip(var_nodes, results):
        q = _TEMPLATES["variable"].format(text=node.text.split(" = ")[0].replace("result of ", "the intermediate value "))
        traj.steps.append(TeacherStep(question=_tag(q), target_node=node.id, answer=res))
    goal_nodes = sdg.goal_nodes()
    if goal_nodes and _final_answer(solution) is not None:
        traj.steps.append(TeacherStep(
            question=_tag(_TEMPLATES["goal"].format(text=question)),
            target_node=goal_nodes[0].id,
            answer=_final_answer(solution)))
    return traj if traj.steps else None


def _tag(q: str) -> str:
    """Wrap a sub-question in the tags the policy is trained to emit."""
    return f"{QUESTION_TAG}{q}{QUESTION_END}"


def load_dataset(path: str | Path) -> Iterator[Dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_teachers(trajectory: TeacherTrajectory, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps({
            "problem_id": trajectory.problem_id,
            "steps": [vars(s) for s in trajectory.steps],
        }, ensure_ascii=False) + "\n")


def load_teachers(teacher_dir: str | Path) -> List[TeacherTrajectory]:
    out: List[TeacherTrajectory] = []
    for p in sorted(Path(teacher_dir).glob("*.jsonl")):
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                out.append(TeacherTrajectory(
                    problem_id=d["problem_id"],
                    steps=[TeacherStep(**s) for s in d["steps"]]))
    return out


def _final_answer(solution: str) -> Optional[str]:
    m = _ANSWER_RE.search(solution)
    return m.group(1) if m else None
