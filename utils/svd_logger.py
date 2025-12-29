# utils/svd_logger.py
import os, json, time
import torch

def _energy_from_singular_values(s):
    # 能量以 s^2 比例計算
    s2 = (s**2).float()
    total = torch.clamp(s2.sum(), min=1e-12)
    e = (s2 / total)
    cum = torch.cumsum(e, dim=0)
    return e.tolist(), cum.tolist()

class SVDLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 不覆蓋，直接 append；可另外寫一個 run_header
    def run_header(self, meta: dict):
        rec = {"_type":"run_header", "ts": time.time(), **meta}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log(self, *, layer:str, w_name:str, m:int, n:int, rank:int,
            S:torch.Tensor, U:torch.Tensor, V:torch.Tensor,
            svd_iter:int=None, extra:dict=None):
        # 主 SVD 的能量
        S = S.detach().cpu()
        U = U.detach().cpu()
        V = V.detach().cpu()
        energy, cum_energy = _energy_from_singular_values(S)

        # 對 U、V 再做一次 SVD（注意：U, V 為正交矩陣，其奇異值多半接近 1）
        try:
            Su = torch.linalg.svdvals(U)  # shape [min(m, r)]
        except Exception:
            Su = torch.linalg.svd(U, full_matrices=False).S
        try:
            Sv = torch.linalg.svdvals(V)  # shape [min(r, n)] or [min(n, r)]
        except Exception:
            Sv = torch.linalg.svd(V, full_matrices=False).S

        u_energy, u_cum = _energy_from_singular_values(Su)
        v_energy, v_cum = _energy_from_singular_values(Sv)

        rec = {
            "_type": "svd_event",
            "ts": time.time(),
            "layer": layer,          # e.g. model.layers.0.self_attn.q_proj
            "weight": w_name,        # 建議就填 "weight" 或你想標的別名
            "m": int(m),             # out_features
            "n": int(n),             # in_features
            "r": int(S.numel()),
            "svd_iter": svd_iter,    # 可為 None；若你有多輪搜尋可自增
            "S": S.tolist(),
            "S_energy": energy,
            "S_cum_energy": cum_energy,
            "U_S": Su.tolist(),
            "U_energy": u_energy,
            "U_cum_energy": u_cum,
            "V_S": Sv.tolist(),
            "V_energy": v_energy,
            "V_cum_energy": v_cum,
        }
        if extra: rec.update(extra)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# 方便在任何地方取得 logger 路徑
def get_logger_from_env():
    path = os.environ.get("SVDLOG_PATH", None)
    return SVDLogger(path) if path else None
