"""Protocol B — downstream fine-tuning evaluation.

1. Sample seed MWPs from GSM8K train; generate 3-5 Socratic sub-question
   variants per seed.
2. Fine-tune a FRESH solver (same LoRA configuration for both arms) on the valid
   augmented QA pairs, using the sub-questions as chain-of-thought context; the
   baseline solver is fine-tuned on the unaugmented training set.
3. Measure mean accuracy over `num_decoding_runs` decoding runs per problem on
   the benchmark suites (AIME-24 / GSM8K / MathQA / OlympiadBench).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..sdg.schema import SDG
from ..trajectory import Trajectory
from .solver import SolverHarness, answer_correct, extract_answer, _gold_from


def build_augmented_pairs(seeds: List[Dict], generator: Callable[[SDG], Trajectory],
                          variants_per_seed: int = 3, seed: int = 42) -> List[Dict]:
    """Generate augmented QA pairs: {question, sub_questions, answer}."""
    rng = random.Random(seed)
    pairs = []
    for item in seeds:
        sdg = item.get("sdg")
        if sdg is None:
            continue
        for v in range(variants_per_seed):
            rng.seed(hash((item.get("id", item["question"]), v)) % (2**31))
            traj = generator(sdg)
            qs = traj.questions
            if not qs:
                continue
            pairs.append({
                "id": f"{item.get('id', '')}_v{v}",
                "question": item["question"],
                "sub_questions": qs,
                "answer": _gold_from(item.get("gold_answer") or item.get("answer")),
            })
    return pairs


@dataclass
class ProtocolBResult:
    method: str
    num_benchmarks: int
    mean_accuracy: float
    per_benchmark: Dict[str, float]
    std_over_runs: float


def finetune_solver_on_pairs(solver_backbone: str, pairs: List[Dict],
                             out_dir: str, epochs: int = 1, lr: float = 1e-5,
                             device: str = "cpu", batch_size: int = 1) -> str:
    """LoRA fine-tune a fresh solver on augmented QA pairs (same recipe as the
    baseline fine-tuning). Returns the adapter directory."""
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)
    from .solver import build_solve_prompt
    from ..models.policy import from_dtype_arg

    tok = AutoTokenizer.from_pretrained(solver_backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        solver_backbone, **from_dtype_arg(torch.float32))
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    def enc(pair):
        prompt = build_solve_prompt(pair["question"], pair.get("sub_questions"))
        target = f"{pair.get('answer','')}"
        text = prompt + "\n#### " + target + (tok.eos_token or "")
        out = tok(text, truncation=True, max_length=1024)
        out["labels"] = out["input_ids"][:]
        return out

    ds = [enc(p) for p in pairs]
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, logging_steps=10, save_strategy="no", report_to=[],
        remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    adapter = str(Path(out_dir) / "adapter")
    model.save_pretrained(adapter)
    return adapter


def evaluate_solver_accuracy(solver: SolverHarness, benchmark_problems: List[Dict],
                             num_runs: int = 16, seed: int = 42) -> Dict[str, float]:
    """Mean accuracy per benchmark over multiple decoding runs."""
    results = {}
    by_bench: Dict[str, List[Dict]] = {}
    for p in benchmark_problems:
        by_bench.setdefault(p.get("benchmark", "default"), []).append(p)
    for bench, items in by_bench.items():
        runs = []
        for r in range(num_runs):
            np.random.seed(seed + r); random.seed(seed + r)
            correct = 0
            for p in items:
                gold = _gold_from(p.get("gold_answer") or p.get("answer"))
                out = solver.solve(p["question"], p.get("sub_questions"))
                correct += int(answer_correct(extract_answer(out), gold))
            runs.append(correct / max(len(items), 1))
        results[bench] = float(np.mean(runs))
        results[f"{bench}_std"] = float(np.std(runs))
    return results


def paired_t_test(treatment_runs: List[float], control_runs: List[float]):
    """Paired t-test across decoding runs."""
    from scipy import stats  # optional dependency
    t, p = stats.ttest_rel(treatment_runs, control_runs)
    return float(t), float(p)


def run_protocol_b(seeds: List[Dict], benchmark_problems: List[Dict],
                   generator: Callable[[SDG], Trajectory], solver_backbone: str,
                   work_dir: str, method_name: str = "graph_cpo",
                   variants_per_seed: int = 3, num_runs: int = 16,
                   baseline: bool = False, device: str = "cpu") -> ProtocolBResult:
    """End-to-end Protocol B for one generator (or the unaugmented baseline)."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    pairs: List[Dict] = []
    if not baseline:
        pairs = build_augmented_pairs(seeds, generator, variants_per_seed)
        with open(Path(work_dir) / "augmented_pairs.jsonl", "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
    adapter = finetune_solver_on_pairs(solver_backbone, pairs or [
        {"question": s["question"], "sub_questions": None,
         "answer": _gold_from(s.get("gold_answer") or s.get("answer"))} for s in seeds],
        out_dir=str(Path(work_dir) / "solver_ft"))
    solver = SolverHarness(solver_backbone, device=device)   # base; adapter merged below
    try:
        from peft import PeftModel
        solver.model = PeftModel.from_pretrained(solver.model, adapter)
    except Exception as e:                                    # pragma: no cover
        print(f"[protocol_b] adapter load failed ({e}); evaluating base solver")
    per_bench = evaluate_solver_accuracy(solver, benchmark_problems, num_runs=num_runs)
    accs = [v for k, v in per_bench.items() if not k.endswith("_std")]
    return ProtocolBResult(
        method=method_name, num_benchmarks=len(accs),
        mean_accuracy=float(np.mean(accs)) if accs else 0.0,
        per_benchmark={k: v for k, v in per_bench.items() if not k.endswith("_std")},
        std_over_runs=float(np.mean([v for k, v in per_bench.items()
                                     if k.endswith("_std")])) if accs else 0.0)


def save_result(result: ProtocolBResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
