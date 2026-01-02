#!/usr/bin/env python3
import argparse, math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    ap = argparse.ArgumentParser(description="Baseline perplexity evaluation (no compression / no precompute).")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=2048, help="Stride between windows. Default=seqlen (non-overlap).")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu|cuda:0 ...")
    return ap.parse_args()

def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Dtype
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Dataset
    ds_name = args.dataset
    if ds_name.lower() in ("wikitext2", "wikitext-2", "wikitext"):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
        tag = "Wikitext2"
    else:
        ds = load_dataset(ds_name, split="test")
        if "text" not in ds.column_names:
            raise ValueError(f"Dataset {ds_name} missing 'text' column; columns={ds.column_names}")
        text = "\n\n".join(ds["text"])
        tag = ds_name

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype if device != "cpu" else torch.float32,
        device_map=None,
    )
    model.to(device)
    model.eval()

    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    seqlen = args.seqlen
    stride = args.stride if args.stride > 0 else seqlen

    nll_sum = 0.0
    n_tokens = 0

    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for start in range(0, input_ids.numel() - 1, stride):
            end = min(start + seqlen, input_ids.numel())
            if end - start < 2:
                break
            batch = input_ids[start:end].unsqueeze(0).to(device)

            outputs = model(input_ids=batch, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nll_sum += float(nll)
            n_tokens += shift_labels.numel()

            if end == input_ids.numel():
                break

    ppl = math.exp(nll_sum / max(1, n_tokens))

    # Keep greppable line for existing extract_ppl()
    if tag == "Wikitext2":
        print(f"Wikitext2 Test PPL: {ppl}")
    else:
        print(f"{tag} Test PPL: {ppl}")

if __name__ == "__main__":
    main()
