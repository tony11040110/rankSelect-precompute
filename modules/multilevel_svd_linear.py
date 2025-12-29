import torch
import torch.nn as nn

from modules.svd_linear import SVDLinear

# 最大允許的 B 截斷相對誤差 (Frobenius)，可以自己調 0.1、0.05 等
ERRB_MAX: float | None = 0.2


class MultiSVDLinear(nn.Module):
    """
    Multilevel SVD factorized linear layer.

    第 1 層：用 SVDLinear.from_linear_whiten_rank 得到
        W ≈ A @ B

    第 2 層：只對 B 做 SVD
        B ≈ U2 @ diag(S2) @ V2^T

    最後實作成三層 nn.Linear：
        inp -> B1Linear -> B2Linear -> ALinear -> out
    """

    def __init__(
        self,
        A: torch.Tensor,
        U2: torch.Tensor,
        S2: torch.Tensor,
        V2T: torch.Tensor,
        bias=None,
        name: str | None = None,
    ) -> None:
        super().__init__()

        m, r = A.shape
        r_u2, r2 = U2.shape
        r_s2 = S2.size(0)
        r_v2t, n = V2T.shape

        assert r == r_u2, f"A and U2 must have compatible shapes, got {A.shape}, {U2.shape}"
        assert r2 == r_s2, f"U2 and S2 rank mismatch, got {U2.shape}, {S2.shape}"
        assert r2 == r_v2t, f"S2 and V2T mismatch, got {S2.shape}, {V2T.shape}"

        self.in_features = n
        self.outer_rank = r
        self.inner_rank = r2
        self.out_features = m
        self.name = name

        # Layer 1: input -> inner_rank
        self.B1Linear = nn.Linear(self.in_features, self.inner_rank, bias=False)
        # Layer 2: inner_rank -> outer_rank
        self.B2Linear = nn.Linear(self.inner_rank, self.outer_rank, bias=False)
        # Layer 3: outer_rank -> out_features
        use_bias = bias is not None
        self.ALinear = nn.Linear(self.outer_rank, self.out_features, bias=use_bias)

        # M1 = diag(S2) @ V2^T  [inner_rank, in_features]
        M1 = torch.diag(S2) @ V2T
        M2 = U2                    # [outer_rank, inner_rank]
        M3 = A                     # [out_features, outer_rank]

        with torch.no_grad():
            self.B1Linear.weight.copy_(M1)
            self.B2Linear.weight.copy_(M2)
            self.ALinear.weight.copy_(M3)
            if use_bias:
                self.ALinear.bias.copy_(bias)

    @staticmethod
    def from_linear_whiten_rank(
        linear: nn.Linear,
        name: str,
        rank: int,
        inner_rank: int,
        succinct: bool = False,
    ):
        """
        1) 先跑 SVDLinear.from_linear_whiten_rank 拿到 W ≈ A @ B
        2) 再對 B 做一次 SVD，拆成三層
        3) 依照 ERRB_MAX 自動調高 inner_rank，控制 B 的截斷誤差
        """

        base = SVDLinear.from_linear_whiten_rank(
            linear=linear,
            name=name,
            rank=rank,
            succinct=succinct,
            # 不強制指定 sigma_fuse，沿用 repo 預設
        )

        if not isinstance(base, SVDLinear):
            # 例如 rank >= min(m, n) 或某些特殊情況直接回傳 nn.Linear
            return base

        A = base.ALinear.weight.data.detach().float()  # [m, r]
        B = base.BLinear.weight.data.detach().float()  # [r, n]

        # full SVD on B
        U2_full, S2_full, V2T_full = torch.linalg.svd(B, full_matrices=False)
        r2_full = S2_full.size(0)

        # 先 clamp 一次 inner_rank
        if inner_rank is None or inner_rank > r2_full:
            inner_rank = r2_full

        # ---- 利用奇異值決定最小 inner_rank, 控制 rel_err_B ----
        if ERRB_MAX is not None and r2_full > 0:
            with torch.no_grad():
                s2_sq = S2_full ** 2
                total = s2_sq.sum()
                # cum[k-1] = sum_{i<=k} sigma_i^2
                cumsum = s2_sq.cumsum(0)
                # 對每個 k 計算最佳截斷誤差 (平方)
                # err^2(k) = (total - cum[k-1]) / total
                errs_sq = (total - cumsum) / (total + 1e-8)

                # 找出滿足 err(k) <= ERRB_MAX 的最小 k
                target = ERRB_MAX ** 2
                ok = (errs_sq <= target).nonzero(as_tuple=False)

                if ok.numel() > 0:
                    k_min = int(ok[0].item()) + 1  # index 從 0 開始，所以 +1
                else:
                    # 沒有任何 k 能滿足門檻 -> 只能退回 full rank
                    k_min = r2_full

                if k_min > inner_rank:
                    print(
                        f"[MultiSVD][{name}] bump inner_rank {inner_rank} -> {k_min} "
                        f"to keep rel_err_B <= {ERRB_MAX}"
                    )
                    inner_rank = k_min

        # 最終截斷
        inner_rank = max(1, min(inner_rank, r2_full))

        U2 = U2_full[:, :inner_rank]      # [r, inner_rank]
        S2 = S2_full[:inner_rank]         # [inner_rank]
        V2T = V2T_full[:inner_rank, :]    # [inner_rank, n]

        bias = linear.bias.data.clone() if linear.bias is not None else None

        multi = MultiSVDLinear(
            A=A,
            U2=U2,
            S2=S2,
            V2T=V2T,
            bias=bias,
            name=name,
        )

        multi = multi.to(linear.weight.dtype).to(linear.weight.device)
        
        # ---- Sanity check: 權重不能有 NaN / Inf，否則退回 base SVDLinear ----
        with torch.no_grad():
            bad = False
            for p in multi.parameters():
                if not torch.isfinite(p).all():
                    bad = True
                    break
            if bad:
                print(f"[MultiSVD][{name}] non-finite weights after construction; "
                      f"fallback to single-level SVDLinear")
                return base
        # -------------------------------------------------------------------
        
        return multi

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.B1Linear(inp)
        x = self.B2Linear(x)
        x = self.ALinear(x)
        return x
