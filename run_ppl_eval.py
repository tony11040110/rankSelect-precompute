#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from whiten_utils import insert_whiten_scale_matrix, find_layers
from modules.svd_linear import SVDLinear

try:
    from datautils import get_calib_data
except Exception as e:
    raise ImportError(
        "Cannot import get_calib_data from datautils. Ensure repo root is in PYTHONPATH. "
        f"Original error: {e}"
    )


def get_ppl_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    elif "c4" in name:
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        testenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        testenc = testenc.input_ids[:, :(256 * seqlen)]
        return TokenizerWrapper(testenc)
    else:
        raise NotImplementedError(name)


@torch.no_grad()
def eval_ppl(model, tokenizer, datasets, seqlen=2048, device="cuda:0"):
    model = model.to(device)
    if isinstance(device, str):
        device = torch.device(device)

    results = {}

    for dataset in datasets.split(","):
        testloader = get_ppl_eval_loaders(dataset, tokenizer, seqlen=seqlen)
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen

        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        nlls = []
        for i in tqdm(range(nsamples), desc=f"ppl:{dataset}"):
            batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(device)
            outputs = model.model(batch)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][:, 1:].to(device)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
        model.config.use_cache = use_cache
        results.update({dataset: {"ppl": float(ppl.item())}})

    return results


def scaling_cache_path(model_id: str, calib_dataset: str) -> str:
    safe = model_id.replace("/", "_")
    if calib_dataset == "wikitext2":
        return f"cache/whiten/{safe}_w2_scaling_matrices_fp16.pt"
    if calib_dataset == "c4":
        return f"cache/whiten/{safe}_c4_scaling_matrices_fp16.pt"
    raise ValueError("calib_dataset must be wikitext2 or c4")


def save_scaling_cache_from_model(model, cache_file: str):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    layers = model.model.layers
    scaling_mats = []
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        d = {}
        for name, mod in subset.items():
            if hasattr(mod, "scaling_diag_matrix"):
                d[name] = mod.scaling_diag_matrix.detach().cpu()
        scaling_mats.append(d)
    torch.save(scaling_mats, cache_file)


def load_scaling_cache_to_model(model, cache_file: str):
    scaling_mats = torch.load(cache_file, map_location="cpu")
    layers = model.model.layers
    if len(scaling_mats) != len(layers):
        raise ValueError(f"cache layers={len(scaling_mats)} != model layers={len(layers)}")
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name, mod in subset.items():
            if name in scaling_mats[i]:
                mod.scaling_diag_matrix = scaling_mats[i][name].to(mod.weight.device)


def apply_rank_config(model, blocks):
    layers = model.model.layers
    if len(layers) != len(blocks):
        raise ValueError(f"rank blocks={len(blocks)} != model layers={len(layers)}")

    for i, rk in enumerate(blocks):
        print(
            f"[WHITEN-SVD] layer {i+1}/{len(blocks)} "
            f"q={rk['q_proj']} k={rk['k_proj']} v={rk['v_proj']} o={rk['o_proj']} "
            f"gate={rk['gate_proj']} up={rk['up_proj']} down={rk['down_proj']}",
            flush=True,
        )

        layer = layers[i]
        attn = layer.self_attn
        mlp = layer.mlp

        attn.q_proj = SVDLinear.from_linear_whiten_rank(attn.q_proj, name=f"layers.{i}.self_attn.q_proj", rank=int(rk["q_proj"]))
        attn.k_proj = SVDLinear.from_linear_whiten_rank(attn.k_proj, name=f"layers.{i}.self_attn.k_proj", rank=int(rk["k_proj"]))
        attn.v_proj = SVDLinear.from_linear_whiten_rank(attn.v_proj, name=f"layers.{i}.self_attn.v_proj", rank=int(rk["v_proj"]))
        attn.o_proj = SVDLinear.from_linear_whiten_rank(attn.o_proj, name=f"layers.{i}.self_attn.o_proj", rank=int(rk["o_proj"]))

        mlp.gate_proj = SVDLinear.from_linear_whiten_rank(mlp.gate_proj, name=f"layers.{i}.mlp.gate_proj", rank=int(rk["gate_proj"]))
        mlp.up_proj = SVDLinear.from_linear_whiten_rank(mlp.up_proj, name=f"layers.{i}.mlp.up_proj", rank=int(rk["up_proj"]))
        mlp.down_proj = SVDLinear.from_linear_whiten_rank(mlp.down_proj, name=f"layers.{i}.mlp.down_proj", rank=int(rk["down_proj"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--tokenizer_id", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="wikitext2,c4")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--calib_dataset", type=str, default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--n_calib_samples", type=int, default=256)
    # NOTE: if you use whiten_* configs, whitening should match the original pipeline.
    # Default to GPU for speed; override with --whiten_device cpu if needed.
    parser.add_argument("--whiten_device", type=str, default="cuda:0")
    parser.add_argument("--force_reprofile", action="store_true")

    args = parser.parse_args()
    t0 = time.time()

    with open(args.config, "r") as f:
        rank_cfg = json.load(f)
    if rank_cfg.get("value", "rank") != "rank":
        raise ValueError(f"unsupported config value={rank_cfg.get('value')}")
    blocks = rank_cfg["blocks"]

    tok_id = args.tokenizer_id or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    model.eval()

    cache_file = scaling_cache_path(model.config._name_or_path, args.calib_dataset)
    if (not args.force_reprofile) and os.path.exists(cache_file):
        print(f"[WHITEN] load cache: {cache_file}", flush=True)
        load_scaling_cache_to_model(model, cache_file)
    else:
        # Important: whiten_utils expects tensors and the layer being forwarded to be on the same device.
        # If dev is CUDA while model is still on CPU, you'll hit CPU/CUDA mismatch.
        print(f"[WHITEN] cache missing or force_reprofile=1 -> profiling on {args.whiten_device}", flush=True)
        model.seqlen = args.seqlen
        calib_loader = get_calib_data(
            args.calib_dataset,
            tokenizer,
            args.model_id,
            nsamples=args.n_calib_samples,
            seqlen=args.seqlen,
            seed=3,
        )

        whiten_dev = torch.device(args.whiten_device)
        moved_to_cuda = (whiten_dev.type == "cuda")
        if moved_to_cuda:
            # Move the whole model to whitening GPU to avoid device mismatch inside whiten_utils.
            model.to(whiten_dev)
            torch.cuda.synchronize()

        insert_whiten_scale_matrix(model, calib_loader, calib_dataset=args.calib_dataset, dev=args.whiten_device)

        # Ensure scaling matrices are on CPU for caching / later CPU-side factorization.
        for layer in model.model.layers:
            for _, mod in find_layers(layer).items():
                if hasattr(mod, "scaling_diag_matrix"):
                    mod.scaling_diag_matrix = mod.scaling_diag_matrix.detach().cpu()

        save_scaling_cache_from_model(model, cache_file)
        print(f"[WHITEN] saved cache: {cache_file}", flush=True)

        if moved_to_cuda:
            # Return model back to CPU for rank-application (cheaper memory footprint);
            # scaling matrices are already CPU tensors.
            model.to("cpu")
            torch.cuda.empty_cache()

    apply_rank_config(model, blocks)

    model.half()
    results = eval_ppl(model, tokenizer, args.datasets, args.seqlen, args.device)
    print(results, flush=True)

    w2 = results.get("wikitext2", {}).get("ppl", "NA")
    c4 = results.get("c4", {}).get("ppl", "NA")
    print(f"PPL_RESULT	wikitext2	{w2}	c4	{c4}", flush=True)

    elapsed = int(time.time() - t0)
    print(f"ELAPSED_SEC	{elapsed}", flush=True)


if __name__ == "__main__":
    main()
