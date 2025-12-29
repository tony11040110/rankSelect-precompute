import json
import sys

# 使用方式：
#   python testtool.py config_dump_mlshare_10.json
# 若沒給參數，預設讀 config_dump_mlshare_10.json
path = sys.argv[1] if len(sys.argv) > 1 else "config_dump_mlshare_10.json"

print(f"[INFO] Load config from: {path}")
with open(path, "r") as f:
    cfg = json.load(f)

print("[INFO] Top-level keys:", list(cfg.keys()))

# 新版檔案裡的 per-layer 統計存在 "layers"
if "layers" in cfg:
    layer_stats = cfg["layers"]
else:
    # 防呆：如果以後你改成 "layer_stats"，也能讀
    layer_stats = cfg.get("layer_stats", {})

print(f"[INFO] Num layers in stats: {len(layer_stats)}")

# 只看前幾層，確認 outer / inner / P_full / P_base / P_multi
for name, stats in list(layer_stats.items())[:20]:
    m = stats.get("m")
    n = stats.get("n")
    r_base = stats.get("r_base")
    r_outer = stats.get("r_outer")
    r_inner = stats.get("r_inner")
    P_full = stats.get("P_full")
    P_base = stats.get("P_base")
    P_multi = stats.get("P_multi")
    rel_err = stats.get("rel_err")

    feasible = bool(stats.get("feasible", 0.0) > 0.5)

    print(f"\n{name}")
    print(f"  feasible = {feasible}")
    print(f"  m={m}, n={n}")
    print(f"  r_base={r_base}, r_outer={r_outer}, r_inner={r_inner}")
    print(f"  P_full={P_full:.0f}, P_base={P_base:.0f}, P_multi={P_multi:.0f}, rel_err={rel_err:.3%}")
