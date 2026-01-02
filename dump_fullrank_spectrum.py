#!/usr/bin/env python3
"""
Dump full-rank singular-value spectra (S) for LLaMA-style linear weights into a JSONL file.

This is intended to be a lightweight alternative to llm_rs.py for producing spectra for plotting
(e.g., heatmaps) without running perplexity eval, rank search, sensitivity, or compression.

Outputs one JSON record per Linear layer with fields compatible with plot_svd_spectrum.py:
- layer, weight, m, n, r
- S, S_energy, S_cum_energy
"""
import argparse
import json
import os
import re
import time
from typing import Iterable, Tuple

import torch
from transformers import AutoModelForCausalLM


DEFAULT_REGEX = r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"


def iter_linear_modules(model: torch.nn.Module, name_regex: str, include_lm_head: bool) -> Iterable[Tuple[str, torch.nn.Linear]]:
    cre = re.compile(name_regex) if name_regex else None
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if (not include_lm_head) and name.endswith("lm_head"):
            continue
        if cre is not None and cre.match(name) is None:
            continue
        yield name, mod


def to_device_str(device: str) -> torch.device:
    if device.lower() in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--name_regex", default=DEFAULT_REGEX, help="Regex to select Linear module names.")
    ap.add_argument("--include_lm_head", action="store_true")
    ap.add_argument("--max_layers", type=int, default=0, help="0 = no limit. For smoke tests.")
    ap.add_argument("--flush_every", type=int, default=1)
    args = ap.parse_args()

    device = to_device_str(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[dump_fullrank_spectrum] model_id={args.model_id} out={args.out} device={device} dtype={dtype} regex={args.name_regex}")

    # Keep model on CPU to avoid holding the entire checkpoint on GPU.
    # We move one weight matrix at a time to the target device for SVD.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    n_written = 0
    t0 = time.time()

    with open(args.out, "w") as f:
        for name, lin in iter_linear_modules(model, args.name_regex, args.include_lm_head):
            if args.max_layers and n_written >= args.max_layers:
                break

            w = lin.weight.detach()
            m, n = int(w.shape[0]), int(w.shape[1])
            r = min(m, n)

            # Compute svdvals in float32 for numerical stability.
            # Prefer GPU if available; fall back to CPU if a CUDA OOM occurs.
            s = None
            used = str(device)
            try:
                w_dev = w.to(device=device, dtype=torch.float32, non_blocking=False)
                s = torch.linalg.svdvals(w_dev)
            except torch.cuda.OutOfMemoryError:
                used = "cpu"
                s = torch.linalg.svdvals(w.to(dtype=torch.float32, device="cpu"))
            finally:
                if device.type == "cuda":
                    # Best-effort cleanup between layers
                    del w_dev
                    torch.cuda.empty_cache()

            s = s.to("cpu", dtype=torch.float32)
            energy = (s * s)
            denom = float(energy.sum().item()) if energy.numel() else 1.0
            frac = (energy / denom).tolist()
            cum = torch.cumsum(energy / denom, dim=0).tolist()

            rec = {
                "_type": "svd_spectrum",
                "ts": time.time(),
                "method": "fullrank",
                "layer": name,
                "weight": name.split(".")[-1],
                "m": m,
                "n": n,
                "r": r,
                "device": used,
                "S": s.tolist(),
                "S_energy": frac,
                "S_cum_energy": cum,
            }
            f.write(json.dumps(rec) + "\n")
            n_written += 1

            if n_written % max(1, args.flush_every) == 0:
                f.flush()

            if n_written % 10 == 0:
                dt = time.time() - t0
                print(f"[dump_fullrank_spectrum] wrote={n_written} last={name} shape=({m},{n}) elapsed={dt:.1f}s")

    dt = time.time() - t0
    print(f"[dump_fullrank_spectrum] DONE records={n_written} elapsed={dt:.1f}s out={args.out}")


if __name__ == "__main__":
    main()
