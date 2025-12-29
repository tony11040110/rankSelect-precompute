import json
import numpy as np
import argparse

def safe_var_and_first(arr):
    """回傳：(標準化後的變異數, 原始第一個值)。"""
    if len(arr) == 0:
        return (float('nan'), float('nan'))
    a = np.array(arr, dtype=np.float32)
    first_val = float(a[0])
    max_val = np.nanmax(a)
    if not np.isfinite(max_val) or max_val == 0:
        return (float('nan'), first_val)
    a_norm = a / max_val
    return (float(np.var(a_norm)), first_val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to svd_spectrum.jsonl")
    args = parser.parse_args()

    layer_stats = {}

    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_type") != "svd_event":
                continue

            layer = rec.get("layer", "unknown")

            S = rec.get("S", [])
            U_S = rec.get("U_S", [])
            V_S = rec.get("V_S", [])

            s_var, s_first = safe_var_and_first(S)
            u_var, u_first = safe_var_and_first(U_S)
            v_var, v_first = safe_var_and_first(V_S)

            layer_stats[layer] = (s_var, s_first, u_var, u_first, v_var, v_first)

    # 印出結果
    print(f"{'Layer':60s}  {'Var(S)':>10s} {'S[0]':>10s}  {'Var(U_S)':>10s} {'U_S[0]':>10s}  {'Var(V_S)':>10s} {'V_S[0]':>10s}")
    print("-" * 110)
    for layer, vals in sorted(layer_stats.items(), key=lambda x: x[0]):
        s_var, s_first, u_var, u_first, v_var, v_first = vals
        print(f"{layer:60s}  {s_var:10.6f} {s_first:10.6f}  {u_var:10.6f} {u_first:10.6f}  {v_var:10.6f} {v_first:10.6f}")

if __name__ == "__main__":
    main()
