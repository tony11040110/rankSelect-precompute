"""config_eval_svdmodel.py

This script mirrors `eval_svdmodel.py` but adds a rank-config driven path.

Key additions
-------------
- Load a JSON rank config and estimate the resulting parameter ratio.
- Optionally rebuild a compressed model from a base model using the rank config.
- When rebuilding, support `--method whiten` to match the `llm_rs.py` pipeline:
  (1) build calibration loader, (2) insert whiten scaling matrices, (3) compress with
  `compress_model_whiten`.

Rank JSON formats
-----------------
The loader accepts multiple shapes, including the format you confirmed:

  { "value": "rank", "blocks": [
      {"q_proj": ..., "k_proj": ..., "v_proj": ..., "o_proj": ...,
       "gate_proj": ..., "up_proj": ..., "down_proj": ...},
      ... (32 layers)
    ]
  }

This is expanded into a flat mapping of module_name -> rank, e.g.
  model.layers.0.self_attn.q_proj -> 128
  model.layers.0.mlp.down_proj -> 1024

Compatibility
-------------
Evaluation functions are reused from `eval_svdmodel.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import torch


# Ensure local imports work when running from outside this directory.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


# -----------------------------
# Rank-config parsing utilities
# -----------------------------

def _normalize_key(k: str) -> str:
    """Normalize keys so we can match common JSON conventions."""
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    return k


def _expand_blocks_rank_cfg(obj: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Expand {value:'rank', blocks:[...]} into a flat module_name->rank dict.

    Expected blocks format (for LLaMA-2 7B):
      blocks[i] has keys in {q,k,v,o}_proj and {gate,up,down}_proj.

    Mapping:
      model.layers.{i}.self_attn.{q/k/v/o}_proj
      model.layers.{i}.mlp.{gate/up/down}_proj
    """
    if not isinstance(obj, dict):
        return None
    if obj.get("value") != "rank":
        return None
    blocks = obj.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return None

    attn_keys = {"q_proj", "k_proj", "v_proj", "o_proj"}
    mlp_keys = {"gate_proj", "up_proj", "down_proj"}

    out: Dict[str, int] = {}
    for i, blk in enumerate(blocks):
        if not isinstance(blk, dict):
            return None
        for k, v in blk.items():
            if k in attn_keys:
                name = f"model.layers.{i}.self_attn.{k}"
            elif k in mlp_keys:
                name = f"model.layers.{i}.mlp.{k}"
            else:
                # Ignore unknown keys to keep compatibility with potential extra fields.
                continue
            out[_normalize_key(name)] = int(v)

    return out if out else None


def load_rank_json(path: str) -> Dict[str, int]:
    """Load a rank json and return a dict: module_name -> rank (int).

    Accepted shapes:
      - Flat dict: {"model.layers.0.self_attn.q_proj": 128, ...}
      - Nested: {"ranks": {...}}
      - List: [{"name": "...", "rank": 128}, ...]
      - Blocks: {"value": "rank", "blocks": [ {...}, ... ]}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Preferred: the blocks format used by llm_rs config dumps.
    if isinstance(obj, dict):
        expanded = _expand_blocks_rank_cfg(obj)
        if expanded is not None:
            return expanded

    # Case 1: dict wrappers / direct dict
    if isinstance(obj, dict):
        if "ranks" in obj and isinstance(obj["ranks"], dict):
            d = obj["ranks"]
            return {_normalize_key(str(k)): int(v) for k, v in d.items()}
        if "items" in obj and isinstance(obj["items"], list):
            items = obj["items"]
            out: Dict[str, int] = {}
            for it in items:
                if isinstance(it, dict) and "name" in it and "rank" in it:
                    out[_normalize_key(str(it["name"]))] = int(it["rank"])
            if out:
                return out
        # Heuristic: assume remaining keys map to ranks
        out2: Dict[str, int] = {}
        ok = True
        for k, v in obj.items():
            if isinstance(v, (int, float)):
                out2[_normalize_key(str(k))] = int(v)
            else:
                ok = False
                break
        if ok and out2:
            return out2

    # Case 2: list of items
    if isinstance(obj, list):
        out3: Dict[str, int] = {}
        for it in obj:
            if isinstance(it, dict) and "name" in it and "rank" in it:
                out3[_normalize_key(str(it["name"]))] = int(it["rank"])
        if out3:
            return out3

    raise ValueError(
        f"Unrecognized rank json format: {path}. "
        "Expected dict(name->rank), list of {name, rank}, or {value:'rank', blocks:[...]}."
    )


@dataclass
class ParamAccounting:
    base_total_params: int
    base_target_linear_params: int
    base_other_params: int
    est_compressed_target_params: int
    est_compressed_total_params: int
    est_param_ratio: float
    matched_layers: int
    missing_layers: int


def estimate_params_from_ranks(
    model: torch.nn.Module,
    rank_dict: Dict[str, int],
    *,
    include_bias: bool = True,
    strict: bool = False,
) -> ParamAccounting:
    """Estimate compressed parameter count given a base model and per-Linear ranks.

    For each nn.Linear W \in R^{out x in} with truncation rank r:
      compressed params ~= r*(in + out) (+ bias if include_bias and bias exists)

    Parameters not belonging to the matched Linear layers are counted as-is.
    """
    base_total = sum(p.numel() for p in model.parameters())

    base_target_linear = 0
    base_other = base_total
    est_target = 0
    matched = 0
    missing = 0

    rank_dict_norm = {_normalize_key(k): int(v) for k, v in rank_dict.items()}

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            key = _normalize_key(name)
            if key in rank_dict_norm:
                r = int(rank_dict_norm[key])
                out_f, in_f = int(mod.out_features), int(mod.in_features)
                full = out_f * in_f
                if include_bias and mod.bias is not None:
                    full += mod.bias.numel()
                base_target_linear += full
                base_other -= full

                comp = r * (in_f + out_f)
                if include_bias and mod.bias is not None:
                    comp += mod.bias.numel()
                est_target += int(comp)
                matched += 1
            else:
                if strict:
                    missing += 1

    est_total = base_other + est_target
    est_ratio = float(est_total) / float(base_total) if base_total else 0.0

    return ParamAccounting(
        base_total_params=int(base_total),
        base_target_linear_params=int(base_target_linear),
        base_other_params=int(base_other),
        est_compressed_target_params=int(est_target),
        est_compressed_total_params=int(est_total),
        est_param_ratio=float(est_ratio),
        matched_layers=int(matched),
        missing_layers=int(missing),
    )


# -----------------------------
# Optional rebuild-from-base
# -----------------------------

def apply_svd_ranks_inplace(
    model: torch.nn.Module,
    rank_dict: Dict[str, int],
    *,
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module:
    """Replace matched nn.Linear modules with SVDLinear using specified ranks."""
    try:
        from svd_linear import SVDLinear  # project-local
    except Exception as e:
        raise ImportError("Failed to import SVDLinear. Ensure svd_linear.py is in PYTHONPATH.") from e

    rank_dict_norm = {_normalize_key(k): int(v) for k, v in rank_dict.items()}

    # Parent module references to replace children.
    name_to_parent: Dict[str, Tuple[torch.nn.Module, str]] = {}
    named_modules = dict(model.named_modules())
    for name, _ in model.named_modules():
        if name == "":
            continue
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = named_modules[parent_name]
        else:
            parent = model
            child_name = name
        name_to_parent[name] = (parent, child_name)

    replaced = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, torch.nn.Linear):
            continue
        key = _normalize_key(name)
        if key not in rank_dict_norm:
            continue

        r = int(rank_dict_norm[key])
        full_rank = min(mod.in_features, mod.out_features)
        if r >= full_rank:
            continue

        new_mod = SVDLinear.from_linear_rank(
            mod, name=name, rank=r, act_aware=False, succinct=False, sigma_fuse="UV"
        )
        new_mod = new_mod.to(dtype)

        parent, child = name_to_parent[name]
        setattr(parent, child, new_mod)
        replaced += 1

    click.secho(f"[Rebuild:svd] Replaced {replaced} Linear layers with SVDLinear.", fg="green")
    return model


def apply_whiten_ranks_inplace(
    model: torch.nn.Module,
    tokenizer,
    rank_dict: Dict[str, int],
    args: argparse.Namespace,
) -> torch.nn.Module:
    """Match llm_rs.py: whiten profiling -> compress_model_whiten."""
    # Lazy imports to keep import cost minimal.
    from datautils import get_calib_data
    from whiten_utils import insert_whiten_scale_matrix, compress_model_whiten

    # llm_rs sets model.seqlen; whiten_utils reads it.
    model.seqlen = int(getattr(args, "seqlen", 2048))

    calib_loader = get_calib_data(
        args.calib_dataset,
        tokenizer,
        args.base_model,
        nsamples=int(args.n_calib_samples),
        seqlen=int(model.seqlen),
        seed=int(args.calib_seed),
    )

    insert_whiten_scale_matrix(
        model=model,
        calib_loader=calib_loader,
        calib_dataset=args.calib_dataset,
        dev=str(args.device),
    )

    # compress_model_whiten expects module-name -> rank (int) (or dict with rank/outer)
    selection_result = {_normalize_key(k): int(v) for k, v in rank_dict.items()}
    compress_model_whiten(model, selection_result, args)

    # Move everything back to GPU similar to llm_rs.py
    model.half()
    torch.cuda.empty_cache()
    model.to(str(args.device))

    click.secho("[Rebuild:whiten] Compression done.", fg="green")
    return model


# -----------------------------
# Main
# -----------------------------


def _print_param_accounting(acct: ParamAccounting) -> None:
    click.secho("================ Param Accounting (Rank JSON) ================", fg="cyan")
    click.secho(f"Base total params: {acct.base_total_params:,}", fg="cyan")
    click.secho(f"Base target Linear params (matched): {acct.base_target_linear_params:,}", fg="cyan")
    click.secho(f"Base other params: {acct.base_other_params:,}", fg="cyan")
    click.secho(f"Estimated compressed target params: {acct.est_compressed_target_params:,}", fg="cyan")
    click.secho(f"Estimated compressed total params: {acct.est_compressed_total_params:,}", fg="cyan")
    click.secho(f"Estimated param ratio: {acct.est_param_ratio:.4f}", fg="cyan")
    click.secho(
        f"Matched layers: {acct.matched_layers} | Missing layers (strict only): {acct.missing_layers}",
        fg="cyan",
    )


def main(args: argparse.Namespace) -> None:
    import time
    import pprint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Reuse evaluation functions from eval_svdmodel.py
    import eval_svdmodel as base_eval

    # Bind global args needed by base_eval.eval_decoding/eval_ttft
    base_eval.args = args

    rank_dict: Optional[Dict[str, int]] = None
    if args.rank_json:
        rank_dict = load_rank_json(args.rank_json)
        click.secho(f"Loaded rank json: {args.rank_json} (entries={len(rank_dict)})", fg="yellow")

    # -------------------------
    # Load model(s)
    # -------------------------
    if args.rebuild_from_base:
        if not args.base_model:
            raise ValueError("--rebuild_from_base requires --base_model")
        if not rank_dict:
            raise ValueError("--rebuild_from_base requires --rank_json")

        click.secho(f"[Rebuild] Loading base model: {args.base_model}", fg="yellow")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16, device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        acct = estimate_params_from_ranks(model, rank_dict, strict=args.strict)
        _print_param_accounting(acct)

        model.eval()

        if args.method == "whiten":
            # Move model to device first (llm_rs does this before whitening)
            model.to(str(args.device))
            torch.cuda.empty_cache()
            model = apply_whiten_ranks_inplace(model, tokenizer, rank_dict, args)
        elif args.method == "svd":
            model = apply_svd_ranks_inplace(model, rank_dict, dtype=torch.float16)
            model.to(str(args.device))
        else:
            raise ValueError(f"Unknown --method: {args.method}")

        torch.cuda.empty_cache()

        if args.save_rebuilt_dir:
            os.makedirs(args.save_rebuilt_dir, exist_ok=True)
            click.secho(f"[Rebuild] Saving rebuilt model to: {args.save_rebuilt_dir}", fg="yellow")
            model.save_pretrained(args.save_rebuilt_dir)
            tokenizer.save_pretrained(args.save_rebuilt_dir)

    else:
        # Default: load --model_name for evaluation
        if not args.model_name:
            raise ValueError(
                "--model_name is required unless --rebuild_from_base is set. "
                "If you do not have a saved compressed model, use --rebuild_from_base."
            )
        click.secho(f"Eval model: {args.model_name}", fg="yellow")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16, device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model.eval()
        model.to(str(args.device))
        torch.cuda.empty_cache()

        # If user provided --base_model, compute ratio relative to base.
        if rank_dict and args.base_model:
            click.secho(f"Computing compression ratio relative to base model: {args.base_model}", fg="yellow")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model, torch_dtype=torch.float16, device_map="cpu"
            )
            acct = estimate_params_from_ranks(base_model, rank_dict, strict=args.strict)
            _print_param_accounting(acct)
            del base_model
            torch.cuda.empty_cache()
        elif rank_dict:
            click.secho("No --base_model provided. Estimating params using loaded model as reference.", fg="yellow")
            acct = estimate_params_from_ranks(model, rank_dict, strict=args.strict)
            _print_param_accounting(acct)

    # -------------------------
    # Run evaluation
    # -------------------------
    start = time.time()

    log: Dict[str, Any] = {}
    decode_results: Dict[str, Any] = {}
    ttft_results: Dict[str, Any] = {}
    ppl_results: Dict[str, Any] = {}
    zero_results: Dict[str, Any] = {}
    mmlu_results: Dict[str, Any] = {}

    if args.eval_decoding:
        click.secho(f"Generate token length for decoding evaluation: {args.generate_len}", fg="yellow")
        decode_results = base_eval.eval_decoding(
            model, tokenizer, generated_len=args.generate_len, batch_size=args.speedup_bs, device=str(args.device)
        )

    if args.eval_ttft:
        ttft_results = base_eval.eval_ttft(
            model, tokenizer, original_len=args.prompt_len, batch_size=args.speedup_bs, device=str(args.device)
        )

    if args.ppl:
        ppl, neg_loss = base_eval.eval_ppl(model, tokenizer, seqlen=2048, batch_size=1)
        ppl_results["wikitext2"] = {"ppl": ppl, "neg_loss": neg_loss}
        log["PPL"] = ppl_results

    if args.zero_shot:
        zero_results = base_eval.lm_eval_zero_shot(
            model=model,
            tokenizer=tokenizer,
            tasks=args.tasks,
            batch_size=args.batch_size,
            peft=args.peft,
            parallelize=args.parallelize,
            report_to_wandb=args.report_to_wandb,
        )
        metric_vals = {
            task: round(result.get("acc,none", result.get("acc", 0.0)), 4)
            for task, result in zero_results.items()
        }
        if metric_vals:
            metric_vals["acc_avg"] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        log["Zero-shot"] = metric_vals

    if args.mmlu:
        mmlu_results = base_eval.lm_eval_mmlu(
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            peft=args.peft,
            parallelize=args.parallelize,
            report_to_wandb=args.report_to_wandb,
        )
        if "mmlu" in mmlu_results and "acc,none" in mmlu_results["mmlu"]:
            log["MMLU, acc"] = round(mmlu_results["mmlu"]["acc,none"], 4)

    end = time.time()
    click.secho(f"Evaluation time: {(end - start)//60} min", fg="yellow")

    # Pretty print summary
    if decode_results:
        print("================ Decoding ================")
        pprint.pprint(decode_results)
        log["Decoding"] = decode_results
    if ttft_results:
        print("================ Prefill ================")
        pprint.pprint(ttft_results)
        log["TTFT"] = ttft_results
    if ppl_results:
        print("================ PPL ================")
        pprint.pprint(ppl_results)
    if "Zero-shot" in log:
        print("================ Zero-shot ================")
        pprint.pprint(log["Zero-shot"])
    if mmlu_results:
        print("================ MMLU ================")
        if "MMLU, acc" in log:
            print("MMLU, acc:", log["MMLU, acc"])

    # Write outputs
    if args.log_to_file:
        os.makedirs("eval_results", exist_ok=True)
        if args.rebuild_from_base:
            tag = f"rebuilt__{(args.base_model or 'base').lstrip('./').replace('/', '_')}__{args.method}"
        else:
            tag = (
                args.model_name.lstrip("./")
                .replace("/", "_")
                .replace(".pt", "")
                .replace(".bin", "")
            )
        if args.rank_json:
            rank_tag = os.path.basename(args.rank_json).replace(".json", "")
            tag = f"{tag}__rankcfg_{rank_tag}"
        with open(f"eval_results/{tag}_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        click.secho(f"Saved: eval_results/{tag}_eval_results.json", fg="green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Evaluation model selection
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Model dir/repo for evaluation (compressed or base). "
            "If omitted, you must set --rebuild_from_base."
        ),
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model dir/repo (for rebuild and/or param accounting)",
    )
    parser.add_argument(
        "--rank_json",
        type=str,
        default=None,
        help="Rank config JSON (flat mapping, list of items, or {value:'rank', blocks:[...]})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="In strict mode, every nn.Linear without a rank entry counts as missing",
    )

    # Rebuild & compression method
    parser.add_argument(
        "--rebuild_from_base",
        action="store_true",
        help="Rebuild a compressed model from --base_model and --rank_json.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="whiten",
        choices=["whiten", "svd"],
        help="Compression method used when rebuilding from base.",
    )
    parser.add_argument(
        "--save_rebuilt_dir",
        type=str,
        default=None,
        help="If set, save rebuilt model/tokenizer to this directory.",
    )

    # Whiten calib settings (used only when --method whiten)
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4"],
        help="Calibration dataset",
    )
    parser.add_argument("--n_calib_samples", type=int, default=256, help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length for calibration")
    parser.add_argument("--calib_seed", type=int, default=3, help="Seed for calibration data")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for rebuild/eval, e.g. cuda:0")

    # knobs referenced by whiten_utils.compress_model_whiten
    parser.add_argument(
        "--search_with_succinct",
        action="store_true",
        help="Build succinct SVDLinear when compressing (whiten path)",
    )
    parser.add_argument(
        "--sigma_fuse",
        type=bool,
        default=True,
        help="Pass-through for SVDLinear sigma_fuse (whiten path)",
    )

    # Keep flags compatible with eval_svdmodel.py
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--peft", action="store_true", help="Load peft model (kept for compatibility)")
    parser.add_argument("--bits", default=16, type=int, help="Bits for quantization")

    parser.add_argument("--ppl", action="store_true", help="Evaluate wikitext2 perplexity")
    parser.add_argument("--zero-shot", dest="zero_shot", action="store_true", help="Evaluate lmeval harness")
    parser.add_argument("--mmlu", action="store_true", help="Evaluate mmlu")
    parser.add_argument(
        "--tasks",
        default="piqa,boolq,hellaswag,arc_easy,arc_challenge,winogrande,openbookqa",
        help="Tasks for lmeval harness",
    )

    # Speed / decoding metrics
    parser.add_argument("--eval_decoding", action="store_true", help="Evaluate decoding speed")
    parser.add_argument("--eval_ttft", action="store_true", help="Evaluate prefill/TTFT")
    parser.add_argument("--generate_len", default=512, type=int, help="Generated token length")
    parser.add_argument("--speedup_bs", default=1, type=int, help="Batch size for speed evaluation")
    parser.add_argument("--prompt_len", default=2048, type=int, help="Prompt length for TTFT")

    # Logging
    parser.add_argument("--parallelize", action="store_true", help="Parallelize for lmeval")
    parser.add_argument("--report_to_wandb", action="store_true", help="Report to wandb")
    parser.add_argument("--log_to_file", action="store_true", help="Log results to a json file")

    main(parser.parse_args())
