"""Rollout engines: sample M sub-questions per (state, action frame).

Default engine uses HuggingFace generate() and works with the graph prefix on
any device (CPU/small GPU). If vLLM is installed and configured
(training.rollout_engine: vllm), a vLLM-backed engine is used for large-scale
runs (the serialized action frame carries the graph signal there).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from .policy import GraphAugmentedPolicy, build_prompt


class RolloutEngine(ABC):
    @abstractmethod
    def sample(self, prompts: List[str], graph_prefixes: Optional[torch.Tensor] = None,
               num_return: int = 8, max_new_tokens: int = 96) -> List[List[str]]:
        """For each prompt, return `num_return` sampled continuations."""


class HFRolloutEngine(RolloutEngine):
    def __init__(self, policy: GraphAugmentedPolicy, temperature: float = 0.8,
                 top_p: float = 0.95, device: str = "cpu"):
        self.policy = policy.to(device)
        self.temperature = temperature
        self.top_p = top_p
        self.device = device

    @torch.no_grad()
    def sample(self, prompts: List[str], graph_prefixes: Optional[torch.Tensor] = None,
               num_return: int = 8, max_new_tokens: int = 96) -> List[List[str]]:
        tok = self.policy.tokenizer
        all_outputs: List[List[str]] = []
        for i, prompt in enumerate(prompts):
            enc = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=tok.model_max_length // 2).to(self.device)
            prefix = None
            if graph_prefixes is not None:
                prefix = graph_prefixes[i].to(self.device)
            outs = self.policy.generate(
                enc["input_ids"], attention_mask=enc["attention_mask"],
                graph_prefix=prefix, max_new_tokens=max_new_tokens,
                temperature=self.temperature, top_p=self.top_p, do_sample=True)
            new_tokens = outs[:, enc["input_ids"].size(1):]
            texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
            all_outputs.append(texts)
        return all_outputs


class VLLMRolloutEngine(RolloutEngine):
    """vLLM-backed sampling (large-scale runs). Prefix conditioning unavailable;
    the serialized action frame carries the graph signal for rollouts, while the
    trained soft prefix is applied in SFT and in reward/logprob re-scoring."""

    def __init__(self, backbone: str, temperature: float = 0.8, top_p: float = 0.95,
                 tensor_parallel: int = 4):
        from vllm import LLM
        self.llm = LLM(model=backbone, tensor_parallel_size=tensor_parallel,
                       trust_remote_code=True)
        self.sampling = {"temperature": temperature, "top_p": top_p}

    def sample(self, prompts: List[str], graph_prefixes: Optional[torch.Tensor] = None,
               num_return: int = 8, max_new_tokens: int = 96):
        from vllm import SamplingParams
        params = SamplingParams(n=num_return, max_tokens=max_new_tokens, **self.sampling)
        outs = self.llm.generate(prompts, params)
        return [[o.text for o in r.outputs] for r in outs]


def build_rollout_engine(kind: str, policy: GraphAugmentedPolicy, backbone: str,
                         device: str = "cpu", tensor_parallel: int = 4) -> RolloutEngine:
    if kind == "vllm":
        try:
            return VLLMRolloutEngine(backbone, tensor_parallel=tensor_parallel)
        except ImportError:
            print("[graph_cpo] vLLM not installed; falling back to HF rollout engine.")
    return HFRolloutEngine(policy, device=device)


__all__ = ["RolloutEngine", "HFRolloutEngine", "VLLMRolloutEngine",
           "build_rollout_engine", "build_prompt"]
