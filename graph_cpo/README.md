# Graph-CPO Implementation

Implementation of **"Graph-CPO: Graph-Grounded Contrastive Pedagogical
Optimization for Socratic Question Generation"** (see the [repository
README](../README.md) for the project overview and quick start).

This package implements the full pipeline. It is config-driven: the dev
config (`configs/dev_qwen_0.5b.yaml`) runs on a small model on 1 GPU or CPU, and
`configs/paper_qwen32b.yaml` runs the full Qwen3-32B / 4×A100 setup —
same code, different YAML. All commands below assume the repository root as the
working directory.

## Layout

| Path | Contents |
|---|---|
| `sdg/schema.py`, `sdg/silver_labels.py`, `sdg/tagger.py`, `sdg/rel_classifier.py`, `sdg/dag.py`, `sdg/gnn.py` | Semantic Dependency Graph construction (BERT tagger → biaffine classifier → cycle removal → relation-aware GNN) |
| `planning/frontier.py`, `planning/selector.py`, `planning/action_frame.py`, `planning/prefix.py`, `planning/teacher.py` | Frontier tracking, node selection, action frames, graph prefix, teacher sequences |
| `models/policy.py`, `models/rollout.py` | Graph-Augmented Policy (Qwen + LoRA + soft graph prefix) and rollout engines |
| `trainer/sft.py` | Targeted supervised warm start |
| `rewards/contrastive.py`, `rewards/coherence.py`, `rewards/format.py` | The three reward dimensions (relevance, coherence, format) |
| `advantage.py` | Decoupled adaptive advantage + curriculum weighting |
| `losses.py` | Graph-CPO objective (clipped surrogate + KL + anchor blend) |
| `trainer/rl.py`, `trainer/grpo_scalar.py` | RL trainer + scalar-GRPO baseline (reward-collapse ablation) |
| `eval/protocol_a.py`, `eval/protocol_b.py`, `eval/validity.py`, `eval/socratic_ablation.py`, `eval/graph_quality.py`, `eval/cost.py`, `eval/report.py` | Evaluation harness (Protocols A/B, validity, ablations, graph quality, cost, LaTeX tables) |
| `tests/` | Unit + golden tests (including the worked advantage example) |

## Pipeline

```bash
# 0. install
pip install -r graph_cpo/requirements.txt
CFG=graph_cpo/configs/dev_qwen_0.5b.yaml      # or graph_cpo/configs/paper_qwen32b.yaml

# 1. silver SDG labels from GSM8K solution steps (target: 12,000 problems)
python -m graph_cpo.scripts.make_silver_labels $CFG --input data/train.jsonl --limit 12000

# 2. train the tagger and the GNN + biaffine relation classifier
python -m graph_cpo.scripts.train_tagger   $CFG --epochs 3
python -m graph_cpo.scripts.train_relf     $CFG --epochs 3

# 3. build + cache SDGs for train/test splits (offline, cached)
python -m graph_cpo.scripts.build_graphs $CFG --input data/train.jsonl  --limit 8000
python -m graph_cpo.scripts.build_graphs $CFG --input data/test.jsonl   --limit 1000

# 3b. graph construction quality vs. expert-annotated graphs
python -m graph_cpo.scripts.eval_graphs $CFG --gold data/sdg/expert_graphs.jsonl

# 4. teacher Socratic sequences (target: 8,000 MWPs)
python -m graph_cpo.scripts.build_teachers $CFG --input data/train.jsonl --limit 8000

# 5. targeted SFT warm start (pi_theta init + pi_ref)
python -m graph_cpo.scripts.train_sft $CFG --limit 2000

# 6. Graph-CPO RL (and the scalar-GRPO baseline for the reward-collapse figure)
python -m graph_cpo.scripts.train_rl $CFG --limit 500 --epochs 50
python -m graph_cpo.scripts.train_rl $CFG --limit 500 --epochs 50 --baseline scalar

# 7. evaluation
python -m graph_cpo.scripts.run_eval_a    $CFG --input data/test.jsonl --limit 200
python -m graph_cpo.scripts.run_eval_b    $CFG --seeds data/train.jsonl --benchmarks data/test.jsonl --limit 1000
python -m graph_cpo.scripts.run_validity  $CFG --input data/test.jsonl --limit 200
```

Smoke test (one command, end to end, tiny models):

```bash
python -m graph_cpo.scripts.demo_end_to_end $CFG --n_problems 5
```

Expert-graph annotation export (for the expert-annotated SDG evaluation set):

```bash
python -m graph_cpo.scripts.export_annotation_template \
    --input data/test.jsonl --output data/sdg/expert_graphs_template.jsonl --n 1000
```

## Tests

```bash
python -m unittest discover -s graph_cpo/tests -t .
```

## Notes on design choices

- **Relevance embedding space.** Since a GNN embedding and a question encoder
  live in different spaces, the
  default (`rewards.rel_space: text`) computes both sides with the same frozen
  encoder (`Enc(d(v))`); `gnn_projected` instead learns a projection from GNN
  space into encoder space (`models/encoder.py:GNNToEncoderProjector`).
- **NLI judge.** Full scale: `microsoft/deberta-v3-large-mnli`; dev config uses
  `cross-encoder/nli-deberta-base` for speed. Architecturally distinct from the
  Qwen generator in both cases (no self-rewarding).
- **Rollout engine.** `training.rollout_engine: hf` works anywhere (soft graph
  prefix supported); `vllm` is recommended for 32B rollout throughput — the
  serialized action frame carries the graph signal there.

