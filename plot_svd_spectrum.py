
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to svd_spectrum.jsonl")
    ap.add_argument("--out", type=str, default="./svd_plots", help="Output directory for images")
    ap.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    ap.add_argument("--figsize_w", type=float, default=10.0, help="Figure width (inches)")
    ap.add_argument("--figsize_h", type=float, default=6.0, help="Figure height (inches)")
    ap.add_argument("--latest_only", action="store_true",
                    help="If multiple events exist per (kind, layer), keep only the latest (default True).")
    ap.add_argument("--no_latest_only", dest="latest_only", action="store_false")
    ap.set_defaults(latest_only=True)
    return ap.parse_args()


def extract_layer_idx_and_kind(layer_path: str):
    parts = layer_path.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                layer_idx = None
            break
    kind = parts[-1] if len(parts) > 0 else layer_path
    return layer_idx, kind


def append_event(store, rec):
    if rec.get("_type") != "svd_event":
        return
    layer = rec["layer"]
    ts = float(rec.get("ts", 0.0))
    S = rec.get("S", [])
    U_S = rec.get("U_S", [])
    V_S = rec.get("V_S", [])
    layer_idx, kind = extract_layer_idx_and_kind(layer)

    if kind not in store:
        store[kind] = {}

    y_key = layer_idx if layer_idx is not None else layer

    entry = store[kind].get(y_key)
    new_payload = {
        "S": np.array(S, dtype=np.float32),
        "U_S": np.array(U_S, dtype=np.float32),
        "V_S": np.array(V_S, dtype=np.float32),
        "ts": ts,
        "layer": layer,
        "layer_idx": layer_idx,
    }
    if entry is None:
        store[kind][y_key] = new_payload
    else:
        replace = ts >= entry["ts"]
        if (not replace) and (len(S) > len(entry["S"])):
            replace = True
        if replace:
            store[kind][y_key] = new_payload


def to_matrix(rows_dict, field):
    items = list(rows_dict.items())
    def sort_key(it):
        y_key, payload = it
        li = payload.get("layer_idx", None)
        if isinstance(li, int):
            return (0, li)
        return (1, str(y_key))
    items.sort(key=sort_key)

    y_labels = []
    arrays = []
    max_len = 0
    for y_key, payload in items:
        arr = payload[field]
        arrays.append(arr)
        max_len = max(max_len, arr.shape[0])
        if payload.get("layer_idx") is not None:
            y_labels.append(str(payload["layer_idx"]))
        else:
            y_labels.append(payload.get("layer", "unknown"))

    num_rows = len(arrays)
    mat = np.full((num_rows, max_len), np.nan, dtype=np.float32)
    for r, arr in enumerate(arrays):
        L = arr.shape[0]
        if L > 0:
            mat[r, :L] = arr
    return mat, y_labels


def normalize_rows_to_percent(mat):
    norm = mat.copy()
    for r in range(norm.shape[0]):
        row = norm[r, :]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        max_val = np.nanmax(row[finite])
        if not np.isfinite(max_val) or max_val == 0.0:
            abs_max = np.nanmax(np.abs(row[finite]))
            if np.isfinite(absmax) and abs_max > 0.0:
                norm[r, finite] = (np.abs(row[finite]) / abs_max) * 100.0
        else:
            norm[r, finite] = (row[finite] / max_val) * 100.0
    return norm


def plot_heatmap(mat_percent, y_labels, title, out_path, dpi=140, figsize=(10, 6)):
    import numpy.ma as ma

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=figsize, dpi=dpi)
    masked = ma.masked_invalid(mat_percent)
    im = plt.imshow(masked, aspect="auto", interpolation="nearest", vmin=0.0, vmax=100.0)
    cbar = plt.colorbar(im)
    cbar.set_label("% of layer-wise max")
    plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
    plt.xlabel("index")
    plt.ylabel("layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    store = {}

    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_type") == "svd_event":
                append_event(store, rec)

    if not store:
        print("No svd_event found in log. Check the log path or logger usage.")
        return

    # Prepare subfolders for each field
    subdirs = {
        "S": os.path.join(args.out, "S"),
        "U_S": os.path.join(args.out, "U_S"),
        "V_S": os.path.join(args.out, "V_S"),
    }
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)

    for kind, rows in store.items():
        for field in ["S", "U_S", "V_S"]:
            mat, y_labels = to_matrix(rows, field)
            mat_percent = normalize_rows_to_percent(mat)
            title = f"{kind} – {field} (row-wise %)"
            # Save into its own subfolder per field
            filename = f"{kind}_rowpct.png"
            out_path = os.path.join(subdirs[field], filename)
            plot_heatmap(mat_percent, y_labels, title, out_path, dpi=args.dpi, figsize=(args.figsize_w, args.figsize_h))
            print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
