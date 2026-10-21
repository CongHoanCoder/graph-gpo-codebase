"""Typed configuration for Graph-CPO.

The large-scale hyperparameters live in configs/paper_qwen32b.yaml; the dev
config (configs/dev_qwen_0.5b.yaml) shrinks model size and batch scale only.
All algorithmic constants (curriculum, contrastive penalty, selection weights)
are identical in both configs.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class ModelConfig:
    backbone: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora: LoraConfig = field(default_factory=LoraConfig)
    gradient_checkpointing: bool = False
    dtype: str = "float32"


@dataclass
class GNNConfig:
    layers: int = 2
    hidden: int = 128
    num_relations: int = 6


@dataclass
class SDGConfig:
    tagger_model: str = "bert-base-uncased"
    rel_classifier_hidden: int = 256
    gnn: GNNConfig = field(default_factory=GNNConfig)
    prefix_dim: int = 128
    silver_dir: str = "data/sdg/silver"
    graph_cache: str = "graphs/"
    min_edge_conf: float = 0.3


@dataclass
class PlanningConfig:
    lambda_g: float = 0.4
    lambda_d: float = 0.3
    lambda_c: float = 0.2
    lambda_r: float = 0.1
    max_questions: int = 5
    sim_threshold: float = 0.75
    nli_threshold: float = 0.6
    teacher_dir: str = "data/teachers"


@dataclass
class RewardsConfig:
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    rel_space: str = "text"          # text | gnn_projected
    nli_model: str = "cross-encoder/nli-deberta-base"
    eta: float = 0.5


@dataclass
class SFTConfig:
    lr: float = 2.0e-5
    batch_size: int = 2
    grad_accum: int = 2
    warmup_steps: int = 20
    epochs: int = 1


@dataclass
class RLConfig:
    lr: float = 1.0e-5
    epochs: int = 2
    rollouts_per_prompt: int = 4
    kl_beta: float = 0.01
    clip_eps: float = 0.2
    curriculum_lambda: float = 0.01
    omega_min: float = 0.1
    omega_max: float = 0.8
    alpha0: float = 0.8
    alpha_min: float = 0.05
    alpha_lambda: float = 0.01
    batch_prompts: int = 4
    adv_eps: float = 1.0e-8
    # Ablation switches (reward-dimension ablation):
    #   mode: cpo (independent normalization) | scalar (GRPO: scalarize first)
    #   use_rewards: subset of ["rel", "coh", "form"]
    #   dynamic_weighting: False => fixed weights (the "w/o DW" variant)
    mode: str = "cpo"
    use_rewards: List[str] = field(default_factory=lambda: ["rel", "coh", "form"])
    dynamic_weighting: bool = True


@dataclass
class TrainingConfig:
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    rollout_engine: str = "hf"
    valid_every_epochs: int = 1
    max_new_tokens: int = 96


@dataclass
class EvalConfig:
    solver_backbone: str = "Qwen/Qwen2.5-0.5B-Instruct"
    solver_max_tokens: int = 512
    solver_timeout_s: int = 60
    num_decoding_runs: int = 2
    protocol_b_seeds: int = 50
    protocol_b_variants: int = 2
    benchmarks: List[str] = field(default_factory=lambda: ["gsm8k"])


@dataclass
class Config:
    experiment_name: str = "graph_cpo"
    seed: int = 42
    output_dir: str = "runs/default"
    model: ModelConfig = field(default_factory=ModelConfig)
    sdg: SDGConfig = field(default_factory=SDGConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _build(cls, data: dict):
    """Recursively construct a dataclass from a nested dict."""
    if data is None:
        return cls()
    import typing
    hints = typing.get_type_hints(cls)
    field_map = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in field_map:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        nested = hints.get(key)
        if dataclasses.is_dataclass(nested) and isinstance(value, dict):
            kwargs[key] = _build(nested, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return _build(Config, data)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
