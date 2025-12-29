from __future__ import annotations

"""Global greedy rank selection from singular-value spectra.

This module implements a *training-free* alternative to sensitivity-based search.
It computes singular values for each nn.Linear weight and uses a global greedy
(priority queue) scheme to reduce ranks until a target total parameter ratio is met.

Parameter model
---------------
For a dense weight W (m x n), we assume a low-rank factorization with rank k
has parameter count: k*(m+n) (bias ignored).

Greedy objective (proxy)
-----------------------
Reducing rank from k -> k-step increases the Frobenius optimal approximation error
by approximately sum_{i=k-step+1..k} s_i^2, where s_i are singular values of W.

We greedily choose the next reduction action that minimizes one of:
- 'energy':         delta_energy
- 'energy_per_param': delta_energy / delta_params_saved
- 'percent':        (s_k / s_1)  (simple heuristic)

The default 'energy_per_param' tends to be a reasonable tradeoff.

This produces a per-layer rank config usable by compress_model_*.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import heapq
import torch
import torch.nn as nn


@dataclass
class LayerSpectrum:
    full_name: str
    linear: nn.Linear
    m: int
    n: int
    r_max: int
    s2_cum: torch.Tensor  # cumulative sum of squared singular values (CPU)
    s1: float


def _svdvals(weight: torch.Tensor) -> torch.Tensor:
    """Compute singular values in float32 for stability.

    Uses torch.linalg.svdvals. This can be expensive for large matrices.
    """
    # svdvals on fp16 can be unstable; use fp32
    w = weight
    if w.dtype != torch.float32:
        w = w.float()
    return torch.linalg.svdvals(w)


def _k_upper_no_expand(m: int, n: int, r_max: int) -> int:
    """Maximum k such that k*(m+n) <= m*n (i.e., does not increase params)."""
    # floor(mn/(m+n)) is the largest rank where factor params do not exceed dense params
    k = (m * n) // (m + n)
    return int(min(r_max, k))


def _round_down_to_step(k: int, step: int) -> int:
    if step <= 1:
        return k
    return (k // step) * step


def _energy_between(cum: torch.Tensor, k_next: int, k_cur: int) -> float:
    """Return sum_{i=k_next+1..k_cur} s_i^2 using 1-indexed k.

    cum is 0-indexed cumulative sum: cum[j] = sum_{i=1..j+1} s_i^2.
    """
    if k_cur <= 0 or k_next >= k_cur:
        return 0.0
    hi = k_cur - 1
    lo = k_next - 1
    if lo < 0:
        return float(cum[hi])
    return float(cum[hi] - cum[lo])


def spectrum_greedy_search_truncation_rank(
    linear_info: Dict[nn.Linear, Dict[str, object]],
    param_ratio_target: float,
    rank_step: int = 128,
    score_mode: str = "energy_per_param",
    min_rank: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, int]:
    """Select per-layer truncation ranks using global greedy over spectra.

    Args:
        linear_info: mapping from nn.Linear -> {"full_name": str, ...}
        param_ratio_target: target total parameter ratio (compressed / dense)
        rank_step: decrement step for rank (keeps ranks on a grid)
        score_mode: 'energy', 'energy_per_param', or 'percent'
        min_rank: minimum rank allowed per layer (default: rank_step)
        device: optional device string; if provided, weights are moved temporarily
        verbose: print progress

    Returns:
        {layer_full_name: selected_rank}
    """
    if param_ratio_target <= 0:
        raise ValueError("param_ratio_target must be > 0")

    if min_rank is None:
        # keep consistent with stepped search
        min_rank = max(1, int(rank_step))

    # 1) Collect target Linear layers (exclude lm_head)
    layers: List[Tuple[str, nn.Linear]] = []
    for lin, info in linear_info.items():
        full_name = str(info.get("full_name", ""))
        if not full_name:
            continue
        if "lm_head" in full_name:
            continue
        layers.append((full_name, lin))

    if not layers:
        raise ValueError("No eligible nn.Linear layers found (after excluding lm_head).")

    # 2) Compute dense baseline parameter count
    dense_params = 0
    shapes: Dict[str, Tuple[int, int]] = {}
    for full_name, lin in layers:
        m, n = lin.weight.shape
        dense_params += int(m * n)
        shapes[full_name] = (int(m), int(n))

    target_params = float(dense_params) * float(param_ratio_target)

    # 3) Compute spectra and initial ranks
    spectra: Dict[str, LayerSpectrum] = {}
    ranks: Dict[str, int] = {}

    # Compute initial rank as the largest not-expanding rank, rounded down to step.
    cur_params = 0
    for full_name, lin in layers:
        m, n = shapes[full_name]
        r_max = min(m, n)
        k_upper = _k_upper_no_expand(m, n, r_max)
        if k_upper <= 0:
            # pathological; skip compression by setting rank 1
            k_upper = 1
        k0 = _round_down_to_step(k_upper, rank_step)
        if k0 < min_rank:
            k0 = min_rank if min_rank <= k_upper else k_upper
        k0 = int(max(1, min(k0, r_max)))
        ranks[full_name] = k0
        cur_params += int(k0 * (m + n))

    if verbose:
        cur_ratio = cur_params / dense_params
        print(f"[spectrum] initial ratio={cur_ratio:.4f} target={param_ratio_target:.4f} (dense_params={dense_params})")

    # Early exit if already under budget
    if cur_params <= target_params:
        if verbose:
            print("[spectrum] already <= target budget; returning initial ranks")
        return ranks

    # 4) Build LayerSpectrum objects (compute singular values)
    for full_name, lin in layers:
        m, n = shapes[full_name]
        r_max = min(m, n)
        w = lin.weight
        if device is not None:
            w = w.to(device)
        s = _svdvals(w)
        # Ensure descending (svdvals should already be sorted, but be safe)
        s = torch.sort(s, descending=True).values
        s2 = (s * s).detach().float().cpu()
        s2_cum = torch.cumsum(s2, dim=0)
        s1 = float(s[0].detach().cpu()) if s.numel() > 0 else 0.0
        spectra[full_name] = LayerSpectrum(
            full_name=full_name,
            linear=lin,
            m=m,
            n=n,
            r_max=r_max,
            s2_cum=s2_cum,
            s1=s1,
        )

    # 5) Global greedy with heap
    # Each heap item: (score, full_name, k_cur, k_next, delta_params, delta_energy)
    heap: List[Tuple[float, str, int, int, int, float]] = []

    # Track the last *applied* reduction action for reporting.
    last_action: Optional[Tuple[str, int, int, int, float, float]] = None
    # (layer, k_cur, k_next, delta_params, delta_energy, score)

    def push_next_action(layer: str) -> None:
        k_cur = ranks[layer]
        spec = spectra[layer]
        step = int(rank_step)
        k_next = k_cur - step
        if k_next < min_rank:
            return
        if k_next <= 0:
            return
        # energy increase from dropping (k_next+1..k_cur)
        delta_energy = _energy_between(spec.s2_cum, k_next, k_cur)
        delta_params = int(step * (spec.m + spec.n))

        if score_mode == "energy":
            score = float(delta_energy)
        elif score_mode == "percent":
            # heuristic: smaller normalized tail singular value first
            # use s_k / s_1 at the boundary (approx)
            # (we approximate using energy increment if needed)
            # Here we just map energy increment to a monotone proxy.
            score = float(delta_energy) / (spec.s1 * spec.s1 + 1e-12)
        elif score_mode == "energy_per_param":
            score = float(delta_energy) / float(max(1, delta_params))
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

        heapq.heappush(heap, (score, layer, k_cur, k_next, delta_params, float(delta_energy)))

    # seed heap
    for layer in ranks.keys():
        push_next_action(layer)

    it = 0
    max_iter = 10_000_000
    while cur_params > target_params and heap and it < max_iter:
        it += 1
        score, layer, k_expected, k_next, delta_params, delta_energy = heapq.heappop(heap)
        # stale check
        if ranks[layer] != k_expected:
            continue

        # apply
        ranks[layer] = k_next
        cur_params -= delta_params

        # record last applied action
        last_action = (layer, k_expected, k_next, delta_params, float(delta_energy), float(score))

        # push next possible action for this layer
        push_next_action(layer)

    if verbose:
        final_ratio = cur_params / dense_params
        print(f"[spectrum] done: final ratio={final_ratio:.4f} (target={param_ratio_target:.4f}), steps={it}")
        if last_action is not None:
            layer, k_cur, k_next, delta_params, delta_energy, last_score = last_action
            spec = spectra[layer]

            # s_{layer,k_cur}^2 can be recovered from cumulative sums.
            # cum[j] = sum_{i=1..j+1} s_i^2
            if k_cur <= 0:
                s_k2 = 0.0
            elif k_cur == 1:
                s_k2 = float(spec.s2_cum[0])
            else:
                s_k2 = float(spec.s2_cum[k_cur - 1] - spec.s2_cum[k_cur - 2])
            s_k = float(torch.sqrt(torch.tensor(max(0.0, s_k2))))
            s1 = float(spec.s1)
            s_ratio = (s_k / s1) if s1 > 0 else 0.0

            # Per-removed-singular-value "bang-for-buck" proxy (one direction)
            per_sv_eff = s_k2 / float(max(1, (spec.m + spec.n)))

            print(
                "[spectrum] last_drop: "
                f"layer={layer}, rank {k_cur}->{k_next}, "
                f"score({score_mode})={last_score:.6e}, "
                f"s_lk/s_l1={s_ratio:.6e}, s_lk^2={s_k2:.6e}, "
                f"per_sv_energy_per_param={per_sv_eff:.6e}, "
                f"delta_energy(chunk)={delta_energy:.6e}, delta_params={delta_params}"
            )
        if cur_params > target_params:
            print("[spectrum] WARNING: could not reach target ratio (hit min_rank or exhausted actions).")

    return ranks
