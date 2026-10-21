"""Shared helpers for SFT and RL training."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..trajectory import Trajectory


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nll_of_answer(policy, prompt: str, answer: str,
                  graph_prefix: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Teacher-forcing NLL of `answer` given `prompt` (targeted-SFT loss)."""
    tok = policy.tokenizer
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    answer_ids = tok(answer, return_tensors="pt", add_special_tokens=False)["input_ids"]
    eos = torch.tensor([[tok.eos_token_id or tok.pad_token_id]])
    answer_ids = torch.cat([answer_ids, eos], dim=1)
    device = next(policy.parameters()).device
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(device)
    labels = torch.cat([torch.full_like(prompt_ids, -100), answer_ids], dim=1).to(device)
    out = policy(input_ids=input_ids, labels=labels, graph_prefix=graph_prefix)
    return out.loss


@torch.no_grad()
def _token_ids(tok, text: str, add_special_tokens: bool) -> torch.Tensor:
    return tok(text, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"]


def gen_token_logprobs(policy, prompt: str, generation: str,
                       graph_prefix: Optional[torch.Tensor] = None,
                       grad: bool = False
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token logprobs of `generation` under the policy given `prompt`.

    grad=False (default) detaches — used for behavior-policy (theta_old) and
    reference-policy scores. grad=True keeps the graph for the policy update.
    Returns (logprobs (T_gen,), token_ids (T_gen,)).
    """
    tok = policy.tokenizer
    prompt_ids = _token_ids(tok, prompt, True)
    gen_ids = _token_ids(tok, generation, False)
    if gen_ids.numel() == 0:
        return torch.zeros(0), gen_ids.squeeze(0)
    full = torch.cat([prompt_ids, gen_ids], dim=1).to(next(policy.parameters()).device)
    with torch.set_grad_enabled(grad):
        logp = policy.token_logprobs(full, graph_prefix=graph_prefix)   # (1, T-1)
    p = prompt_ids.size(1)
    span = logp[0, p - 1: p - 1 + gen_ids.size(1)]
    if not grad:
        span = span.detach()
    # keep the policy's device: callers combine these with device-side tensors
    return span, gen_ids.squeeze(0)


def pad_trajectory_rewards(trajs: List[Trajectory],
                           dims: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Stack per-timestep rewards of M rollouts into (M, T) arrays + mask.

    Short trajectories are padded by repeating their last timestep reward with
    mask 0, so the advantage normalization sees consistent shapes without
    polluting the loss.
    """
    T = max(len(t.timesteps) for t in trajs)
    M = len(trajs)
    out: Dict[str, np.ndarray] = {k: np.full((M, T), np.nan) for k in dims}
    mask = np.zeros((M, T))
    for m, traj in enumerate(trajs):
        for t, ts in enumerate(traj.timesteps):
            assert ts.rewards is not None, "rewards must be computed before padding"
            for k in dims:
                out[k][m, t] = getattr(ts.rewards, k)
            mask[m, t] = 1.0
    return out, mask


def questions_as_list(trajs: List[Trajectory]) -> List[List[str]]:
    return [t.questions for t in trajs]
