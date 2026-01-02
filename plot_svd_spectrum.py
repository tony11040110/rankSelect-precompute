#!/usr/bin/env python3
"""
plot_svd_spectrum.py

- Reads svd spectrum jsonl logs (one JSON object per line).
- Produces per-projection heatmaps for singular-value spectra (S) across layers.
- Optionally overlays selection_rank as a short vertical tick per layer.

Output layout (matches bench expectation):
  <out>/S/<kind>_proj_<scale>/<kind>_proj_<scale>.png

Notes:
- We intentionally ignore U_S / V_S (legacy multilevel artifacts).
- X-axis is 1-indexed rank k (so marker is drawn at k, not k-1).

Usage example:
  python -u plot_svd_spectrum.py \
    --log bench_runs/<ts>/exp1_svd_spectrum.jsonl \
    --out bench_runs/<ts>/svd_plots/exp1 \
    --selection bench_runs/<ts>/exp1_config_dump.json \
    --scale rowpct
"""
import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


VERSION = "v5.0"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to svd_spectrum.jsonl")
    ap.add_argument("--out", type=str, required=True, help="Output root directory for images")
    ap.add_argument("--selection", type=str, default="", help="Optional selection config_dump.json (for selection_rank overlay)")
    ap.add_argument("--scale", type=str, default="raw", choices=["raw", "rowpct", "log10"],
                    help="Value scaling: raw S, rowpct (percent energy by row), log10(raw).")
    ap.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    ap.add_argument("--figsize_w", type=float, default=12.0, help="Figure width (inches)")
    ap.add_argument("--figsize_h", type=float, default=7.0, help="Figure height (inches)")
    ap.add_argument("--latest_only", action="store_true",
                    help="If multiple events exist per (kind, layer), keep only the latest (default True).")
    ap.add_argument("--no_latest_only", dest="latest_only", action="store_false")
    ap.set_defaults(latest_only=True)
    ap.add_argument("--k_max", type=int, default=0,
                    help="If >0, truncate plotted K to at most this many ranks (useful for quick smoke tests).")
    return ap.parse_args()


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_KIND_CANDIDATES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _extract_layer_idx(layer_name: str) -> Optional[int]:
    m = _LAYER_RE.search(layer_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_kind(layer_name: str, weight_field: str) -> Optional[str]:
    # Prefer "weight" if present; fall back to "layer" string.
    s = weight_field or layer_name or ""
    for k in _KIND_CANDIDATES:
        if k in s:
            return k
    return None


def _load_selection_ranks(path: str) -> Dict[Tuple[int, str], int]:
    """
    Expected exp1_config_dump.json format:
      {"value":"rank","blocks":[{"q_proj":..., "k_proj":..., ...}, ...]}
    blocks index corresponds to layer index.
    """
    if not path:
        return {}
    with open(path, "r") as f:
        sel = json.load(f)

    blocks = None
    if isinstance(sel, dict) and "blocks" in sel and isinstance(sel["blocks"], list):
        blocks = sel["blocks"]
    elif isinstance(sel, list):
        blocks = sel
    else:
        return {}

    out: Dict[Tuple[int, str], int] = {}
    for layer_idx, blk in enumerate(blocks):
        if not isinstance(blk, dict):
            continue
        for kind, v in blk.items():
            if kind not in _KIND_CANDIDATES:
                continue
            # allow int/float strings
            try:
                out[(layer_idx, kind)] = int(v)
            except Exception:
                continue
    return out


def _as_vec(x: Any) -> Optional[np.ndarray]:
    if isinstance(x, list) and len(x) > 0:
        try:
            return np.asarray(x, dtype=np.float64)
        except Exception:
            return None
    return None


def _rowpct_from_record(rec: Dict[str, Any], s: np.ndarray) -> np.ndarray:
    # Prefer precomputed energy fractions if available.
    if isinstance(rec.get("S_energy"), list):
        ve = _as_vec(rec.get("S_energy"))
        if ve is not None and ve.shape == s.shape:
            return ve * 100.0
    # Fallback: normalize squared S ("energy") within the row.
    e = s * s
    denom = float(np.sum(e)) if float(np.sum(e)) != 0.0 else 1.0
    return (e / denom) * 100.0


def _read_latest_spectra(log_path: str, latest_only: bool = True) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Returns:
      key: (kind, layer_idx)
      val: {"S": np.ndarray, "layer": str, "weight": str, "ts": ..., "svd_iter": ..., "line_no": ...}
    """
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    n_total = 0
    n_used = 0

    with open(log_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if "S" not in rec:
                continue

            layer_name = rec.get("layer") or rec.get("name") or rec.get("module") or ""
            if not isinstance(layer_name, str) or not layer_name:
                continue
            layer_idx = _extract_layer_idx(layer_name)
            if layer_idx is None:
                continue

            weight_field = rec.get("weight") if isinstance(rec.get("weight"), str) else ""
            kind = _extract_kind(layer_name, weight_field)
            if kind is None:
                continue

            s = _as_vec(rec.get("S"))
            if s is None or s.ndim != 1:
                continue

            key = (kind, layer_idx)
            meta = {
                "S": s,
                "rec": rec,
                "layer": layer_name,
                "weight": weight_field,
                "ts": rec.get("ts", 0),
                "svd_iter": rec.get("svd_iter", 0),
                "line_no": line_no,
            }

            if not latest_only or key not in out:
                out[key] = meta
                n_used += 1
            else:
                prev = out[key]
                # Compare by (ts, svd_iter, line_no)
                try:
                    cand = (meta["ts"], meta["svd_iter"], meta["line_no"])
                    cur = (prev["ts"], prev["svd_iter"], prev["line_no"])
                except Exception:
                    cand = (meta["line_no"],)
                    cur = (prev["line_no"],)
                if cand >= cur:
                    out[key] = meta

    print(f"[plot_svd_spectrum] version={VERSION} log={log_path} latest_only={latest_only} "
          f"parsed_lines={n_total} spectra_entries={len(out)}")
    return out


def _plot_kind_heatmap(kind: str,
                       per_layer: Dict[int, Dict[str, Any]],
                       out_root: str,
                       scale: str,
                       sel_ranks: Dict[Tuple[int, str], int],
                       dpi: int,
                       figsize_w: float,
                       figsize_h: float,
                       k_max: int) -> None:
    layer_indices = sorted(per_layer.keys())
    if not layer_indices:
        return
    L = max(layer_indices) + 1
    K = max(int(per_layer[i]["S"].shape[0]) for i in layer_indices)

    if k_max and k_max > 0:
        K = min(K, k_max)

    mat = np.full((L, K), np.nan, dtype=np.float64)

    for li in layer_indices:
        s_full: np.ndarray = per_layer[li]["S"]
        rec: Dict[str, Any] = per_layer[li]["rec"]
        s = s_full[:K]

        if scale == "raw":
            v = s
        elif scale == "log10":
            v = np.log10(np.maximum(s, 1e-30))
        elif scale == "rowpct":
            v = _rowpct_from_record(rec, s)[:K]
        else:
            v = s
        if v.shape[0] != K:
            vv = np.full((K,), np.nan, dtype=np.float64)
            vv[:v.shape[0]] = v
            v = vv
        mat[li, :] = v

    out_dir = os.path.join(out_root, "S", f"{kind}_proj_{scale}")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{kind}_proj_{scale}.png")

    masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h), dpi=dpi)
    # extent makes x-axis 1..K and y-axis 0..L-1; with origin='upper' we keep layer 0 at top.
    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[0.5, K + 0.5, L - 0.5, -0.5],
    )
    cbar = fig.colorbar(im, ax=ax)
    if scale == "rowpct":
        cbar.set_label("row energy (%)")
    elif scale == "log10":
        cbar.set_label("log10(S)")
    else:
        cbar.set_label("S")

    ax.set_title(f"S spectrum heatmap: {kind} (scale={scale})")
    ax.set_xlabel("rank k (1-indexed)")
    ax.set_ylabel("layer index")

    # y ticks for all layers (small font). If too many layers, show every 2.
    step = 1 if L <= 40 else 2
    yticks = list(range(0, L, step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"L{y:02d}" for y in yticks], fontsize=8)

    # Overlay selection ranks as short vertical ticks per layer (k is 1-indexed).
    n_marks = 0
    for li in layer_indices:
        k = sel_ranks.get((li, kind), None)
        if k is None:
            continue
        # clip into plotting range
        if not (1 <= int(k) <= K):
            continue
        kk = float(k)
        ax.plot([kk, kk], [li - 0.35, li + 0.35], linewidth=1.0)
        n_marks += 1

    # Tight layout and save
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    print(f"[plot_svd_spectrum] wrote {out_png} layers={len(layer_indices)}/{L} K={K} marks={n_marks}")


def main():
    args = parse_args()
    spectra = _read_latest_spectra(args.log, latest_only=args.latest_only)

    sel_ranks = _load_selection_ranks(args.selection) if args.selection else {}
    if args.selection and not sel_ranks:
        print("[WARN] No selection ranks parsed from selection file (format mismatch or empty).")

    # group by kind
    by_kind: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for (kind, layer_idx), meta in spectra.items():
        by_kind.setdefault(kind, {})[layer_idx] = meta

    if not by_kind:
        print("[WARN] No plottable spectra found in log.")
        return

    for kind, per_layer in sorted(by_kind.items(), key=lambda x: x[0]):
        _plot_kind_heatmap(
            kind=kind,
            per_layer=per_layer,
            out_root=args.out,
            scale=args.scale,
            sel_ranks=sel_ranks,
            dpi=args.dpi,
            figsize_w=args.figsize_w,
            figsize_h=args.figsize_h,
            k_max=args.k_max,
        )


if __name__ == "__main__":
    main()
