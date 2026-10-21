"""Graph-Augmented Policy: a Qwen (causal LM) backbone
with LoRA adapters and soft graph-prefix injection.

Generation input per timestep:
    x_t = [P; q_{1:t-1}; Ser(z_t); <question>]
plus the continuous graph prefix p_t prepended to the token embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

QUESTION_TAG = "<question>"
QUESTION_END = "</question>"


def from_dtype_arg(torch_dtype):
    """transformers renamed torch_dtype -> dtype; emit the modern kwarg.

    `dtype=` is supported from transformers 4.56 onward; older versions need
    `torch_dtype=`.
    """
    import transformers
    ver = tuple(int(p) for p in transformers.__version__.split(".")[:2]
                if p.isdigit())
    if ver >= (4, 56):
        return {"dtype": torch_dtype}
    return {"torch_dtype": torch_dtype}

PROMPT_TEMPLATE = (
    "Problem: {problem}\n"
    "Previous sub-questions:\n{history}\n"
    "Plan: {frame}\n"
    "Generate the next Socratic sub-question now."
)


def build_prompt(problem: str, history: List[str], frame_text: str) -> str:
    hist = "\n".join(f"{i+1}. {q}" for i, q in enumerate(history)) if history else "(none)"
    return PROMPT_TEMPLATE.format(problem=problem, history=hist, frame=frame_text)


@dataclass
class PolicyOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    input_ids: torch.Tensor
    prefix_len: int


class GraphAugmentedPolicy(nn.Module):
    def __init__(self, backbone: str, tokenizer: Optional[str] = None,
                 lora_enabled: bool = True, lora_r: int = 64, lora_alpha: int = 128,
                 lora_dropout: float = 0.05,
                 lora_targets: Optional[List[str]] = None,
                 graph_prefix_dim: int = 1024, dtype: str = "float32",
                 gradient_checkpointing: bool = False):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or backbone)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone, **from_dtype_arg(torch_dtype))
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if lora_enabled:
            cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=lora_targets or [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, cfg)
        self.embed_dim = self.model.config.hidden_size
        # p_t (graph prefix) is projected to a short sequence of soft tokens
        self.prefix_proj = nn.Linear(graph_prefix_dim, self.embed_dim)
        self.num_prefix_tokens = 4
        self.prefix_len = self.num_prefix_tokens

    # ------------------------------------------------------------------ utils
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _prefix_embeddings(self, graph_prefix: torch.Tensor) -> torch.Tensor:
        """graph_prefix: (prefix_dim,) or (num_prefix, prefix_dim) -> soft tokens."""
        if graph_prefix.dim() == 1:
            graph_prefix = graph_prefix.unsqueeze(0)
        emb = self.prefix_proj(graph_prefix.to(self.prefix_proj.weight.dtype))
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        return emb.to(next(self.model.parameters()).device)

    # ---------------------------------------------------------------- forward
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                graph_prefix: Optional[torch.Tensor] = None) -> PolicyOutput:
        """Compute logits (shifted internally by HF) with optional soft prefix."""
        # move inputs to the model's device (rollouts/tokenization may be CPU)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        if graph_prefix is None:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             labels=labels, use_cache=False)
            return PolicyOutput(out.loss, out.logits, input_ids, 0)

        prefix = self._prefix_embeddings(graph_prefix).to(device)      # (B, P, D)
        tok_emb = self.model.get_input_embeddings()(input_ids)          # (B, T, D)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        bsz = input_ids.size(0)
        prefix_mask = torch.ones(bsz, prefix.size(1), dtype=torch.long, device=device)
        full_mask = torch.cat(
            [prefix_mask, attention_mask if attention_mask is not None
             else torch.ones_like(input_ids)], dim=1)
        full_labels = None
        if labels is not None:
            pad = torch.full((bsz, prefix.size(1)), -100, dtype=torch.long, device=device)
            full_labels = torch.cat([pad, labels], dim=1)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=full_mask,
                         labels=full_labels, use_cache=False)
        return PolicyOutput(out.loss, out.logits, input_ids, prefix.size(1))

    # ------------------------------------------------------------ logprobs/RL
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                 graph_prefix: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 96, temperature: float = 0.8,
                 top_p: float = 0.95, do_sample: bool = True) -> torch.Tensor:
        """Autoregressive decoding; returns FULL sequences (prompt + new tokens)."""
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                          temperature=temperature, top_p=top_p,
                          pad_token_id=self.tokenizer.pad_token_id)
        if graph_prefix is None:
            return self.model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask, **gen_kwargs)
        prefix = self._prefix_embeddings(graph_prefix).to(device)
        tok_emb = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        bsz = input_ids.size(0)
        prefix_mask = torch.ones(bsz, prefix.size(1), dtype=torch.long, device=device)
        full_mask = torch.cat(
            [prefix_mask, attention_mask if attention_mask is not None
             else torch.ones_like(input_ids)], dim=1)
        out = self.model.generate(inputs_embeds=inputs_embeds, attention_mask=full_mask,
                                  **gen_kwargs)
        # generate() with inputs_embeds returns only generated tokens + embeds;
        # append the text prompt ids so downstream code sees the full sequence.
        return torch.cat([input_ids, out], dim=1)

    def token_logprobs(self, input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None,
                       graph_prefix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-token logprobs of tokens 1..T-1 under the current policy, (B, T-1).

        Position k of the logits predicts sequence token k+1; with P soft-prefix
        embeddings prepended, token i is predicted at position P+i-1.
        """
        out = self.forward(input_ids, attention_mask=attention_mask,
                           graph_prefix=graph_prefix)
        logits = out.logits                       # (B, P+T, V)
        # (inputs are moved to the model device inside forward)
        p = out.prefix_len
        window = logits[:, p:-1, :]               # predicts tokens 1..T-1
        logp = F.log_softmax(window, dim=-1)
        return logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------- persistence
    def save_pretrained(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)               # LoRA adapters
        self.tokenizer.save_pretrained(path)
        torch.save(self.prefix_proj.state_dict(),
                   os.path.join(path, "prefix_proj.pt"))
        torch.save(self.num_prefix_tokens, os.path.join(path, "num_prefix_tokens.pt"))

    @classmethod
    def load_with_adapters(cls, adapter_path: str, backbone: str, **kwargs) -> "GraphAugmentedPolicy":
        import os
        from peft import PeftModel
        policy = cls(backbone, lora_enabled=False, **kwargs)
        policy.model = PeftModel.from_pretrained(policy.model, adapter_path)
        prefix_path = os.path.join(adapter_path, "prefix_proj.pt")
        if os.path.exists(prefix_path):
            policy.prefix_proj.load_state_dict(torch.load(prefix_path))
        return policy
