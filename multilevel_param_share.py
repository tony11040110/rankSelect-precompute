
import math
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn


def _layer_param_counts(m: int, n: int, r: int) -> int:
    """
    Parameter count for single-level SVDLinear using ranks (U,S,V fused into 3 Linear layers):
        A: (m, r), S: (r, r), B: (r, n)
    Bias terms are ignored since they are small compared to weights and
    are the same across configurations.
    """
    return m * r + r * r + r * n


def _multilevel_param_counts(m: int, n: int, r_outer: int, r_inner: int) -> int:
    """
    Parameter count for a 2-level factorization:
        A1: (m, r_outer)
        M : (r_outer, r_inner)
        B2: (r_inner, n)
    """
    return m * r_outer + r_outer * r_inner + r_inner * n


def compute_layer_multilevel_ranks_by_param_share(
    m: int,
    n: int,
    r_base: int,
    p_inner: float,
    tol_ratio: float = 0.05,
) -> Tuple[bool, int, int, Dict[str, Any]]:
    """
    給定單層 baseline rank r_base，計算 multilevel (r_outer, r_inner)，使得：
      1) 最終的 multilevel 參數數量 ~ baseline 相同
      2) 第二層所「負責的壓縮量」佔 baseline 壓縮量的比例為 p_inner（以參數量計）

    參數:
        m, n      : 原始 Linear 的形狀 (out_features, in_features)
        r_base    : baseline 單層 SVD 的 rank
        p_inner   : 想要第二層負責的壓縮比例 (0~1)，以「參數量」定義
        tol_ratio : 允許 multilevel 總參數數 vs baseline 的相對誤差

    回傳:
        feasible  : 是否有可行解
        r_outer   : 第一層 rank
        r_inner   : 第二層 rank
        stats     : 一些中間計算結果 (字典)
    """
    # 0) sanity check
    if r_base <= 0 or r_base > min(m, n):
        return False, 0, 0, {
            "reason": f"invalid r_base={r_base}, m={m}, n={n}",
        }
    if not (0.0 <= p_inner <= 1.0):
        return False, 0, 0, {
            "reason": f"p_inner must be in [0,1], got {p_inner}",
        }

    # 1) baseline 參數數
    P_full = m * n
    P_base = _layer_param_counts(m, n, r_base)
    Delta_base = P_full - P_base  # baseline 總共壓掉多少參數

    if Delta_base <= 0:
        # 代表 baseline 沒有真的壓縮（或形狀太小），直接放棄 multilevel
        return False, 0, 0, {
            "reason": "Delta_base <= 0; baseline does not compress this layer.",
            "P_full": P_full,
            "P_base": P_base,
        }

    # 2) 由 p_inner 決定 outer 應該壓掉多少參數
    Delta_inner = p_inner * Delta_base
    Delta_outer = (1.0 - p_inner) * Delta_base
    P_outer_target = P_full - Delta_outer

    # 3) 解 outer rank:
    #    P_outer(r) = m * r + r^2 + r * n ~= P_outer_target
    #    r^2 + (m+n) * r - P_outer_target = 0
    a = 1.0
    b = float(m + n)
    c = -float(P_outer_target)

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False, 0, 0, {
            "reason": f"quadratic discriminant < 0 for P_outer_target={P_outer_target}",
            "disc": disc,
        }

    r_outer_real = (-b + math.sqrt(disc)) / (2.0 * a)
    # 向上取整，避免 outer 參數太少
    r_outer = int(math.ceil(r_outer_real))

    # 4) 外層 rank 約束: baseline 至少壓縮一部分
    r_outer = max(r_outer, r_base)
    r_outer = min(r_outer, min(m, n))

    # 5) 解 inner rank，要求 multilevel 總參數數約等於 baseline
    #    P_multi(m, n, r_outer, r_inner) = P_base
    # => r_inner * (r_outer + n) = P_base - m * r_outer
    denom = r_outer + n
    num = P_base - m * r_outer

    if denom <= 0:
        return False, 0, 0, {
            "reason": f"denominator <= 0 when solving inner rank; denom={denom}",
            "P_base": P_base,
            "m": m,
            "r_outer": r_outer,
        }

    r_inner_real = num / float(denom)
    # 這裡用 round，你也可以改成 floor/ceil 看需求
    r_inner = int(round(r_inner_real))

    # 6) 檢查 inner rank 合理性
    if r_inner <= 0 or r_inner > r_outer:
        return False, 0, 0, {
            "reason": "inner rank out of range",
            "r_inner_real": r_inner_real,
            "r_inner": r_inner,
            "r_outer": r_outer,
        }

    # 7) 檢查 multilevel 總參數數 vs baseline 差多少
    P_multi = _multilevel_param_counts(m, n, r_outer, r_inner)
    rel_err = abs(P_multi - P_base) / float(P_base)

    feasible = rel_err <= tol_ratio

    stats = {
        "P_full": P_full,
        "P_base": P_base,
        "P_multi": P_multi,
        "Delta_base": Delta_base,
        "Delta_outer": Delta_outer,
        "Delta_inner": Delta_inner,
        "P_outer_target": P_outer_target,
        "r_outer_real": r_outer_real,
        "r_inner_real": r_inner_real,
        "rel_err": rel_err,
    }

    if not feasible:
        stats["reason"] = (
            f"relative param-count error {rel_err:.4f} exceeds tol_ratio={tol_ratio}"
        )

    return feasible, r_outer, r_inner, stats


def build_multilevel_config_from_baseline_by_param_share(
    model: nn.Module,
    baseline_rank_config: Dict[str, int],
    p_inner: float,
    tol_ratio: float = 0.05,
    verbose: bool = False,
) -> Tuple[Dict[str, Dict[str, int]], bool, Dict[str, Dict[str, float]]]:
    """
    給定:
      - model: 已載入的 HF causal LM model
      - baseline_rank_config: dict[layer_full_name] = r_base (int)
      - p_inner: 第二層佔 baseline 壓縮量的比例 (0~1)，以「參數量」計
      - tol_ratio: 允許 multilevel vs baseline 總參數數的相對誤差

    回傳:
      - multi_config: dict[layer_name] = {"outer": r_outer, "inner": r_inner}
      - global_feasible: 是否所有 layer 都可行
      - layer_stats: 每層的統計資訊（可用來 debug / 記錄）
    """
    # 先建立 module-name 映射，方便查維度
    module_dict = {name: m for name, m in model.named_modules()}

    multi_config: Dict[str, Dict[str, int]] = {}
    layer_stats: Dict[str, Dict[str, float]] = {}
    global_feasible = True

    for layer_name, r_base in baseline_rank_config.items():
        if layer_name not in module_dict:
            if verbose:
                print(f"[ParamShare] WARNING: layer {layer_name} not found in model.named_modules(), skip.")
            global_feasible = False
            continue

        mod = module_dict[layer_name]
        if not isinstance(mod, nn.Linear):
            if verbose:
                print(f"[ParamShare] WARNING: {layer_name} is not nn.Linear (got {type(mod)}), skip.")
            global_feasible = False
            continue

        W = mod.weight  # [out_features, in_features]
        m, n = W.shape

        feasible, r_outer, r_inner, stats = compute_layer_multilevel_ranks_by_param_share(
            m=m,
            n=n,
            r_base=int(r_base),
            p_inner=p_inner,
            tol_ratio=tol_ratio,
        )

        layer_stats[layer_name] = {
            "feasible": float(1.0 if feasible else 0.0),
            "m": float(m),
            "n": float(n),
            "r_base": float(r_base),
            "r_outer": float(r_outer),
            "r_inner": float(r_inner),
            "P_full": float(stats.get("P_full", 0.0)),
            "P_base": float(stats.get("P_base", 0.0)),
            "P_multi": float(stats.get("P_multi", 0.0)),
            "rel_err": float(stats.get("rel_err", 0.0)),
        }

        if not feasible:
            # 若這一層在指定 p_inner 下不可行（例如 baseline 沒有壓縮、或解不出合理的 inner rank），
            # 則「不要」把它放進 multi_config，代表這一層維持原本（whiten + baseline 或 full-rank）結構。
            global_feasible = False
            if verbose:
                reason = stats.get("reason", "unknown")
                print(
                    f"[ParamShare] layer={layer_name} infeasible for p_inner={p_inner}: {reason}"
                )
            # 不中斷迴圈，全部 layer 都算完再回傳
            continue

        multi_config[layer_name] = {
            "outer": int(r_outer),
            "inner": int(r_inner),
        }

    return multi_config, global_feasible, layer_stats
