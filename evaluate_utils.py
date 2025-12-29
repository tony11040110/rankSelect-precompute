import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

from datautils import get_eval_loaders
from datasets import load_dataset
import time
import re


@torch.no_grad()
def evaluate_perplexity(model, input_ids, seqlen: int = 2048):
    """
    input_ids: [N, L] 的 LongTensor（通常是 wikitext2 encode 完的）
    seqlen:    每個 eval chunk 的長度
    """
    model.eval()

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    n_tokens = input_ids.numel()
    nsamples = n_tokens // seqlen
    if nsamples == 0:
        raise ValueError(f"evaluate_perplexity: not enough tokens ({n_tokens}) for seqlen={seqlen}")

    losses = []

    for i in range(nsamples):
        start = i * seqlen
        end   = (i + 1) * seqlen
        batch = input_ids[:, start:end]

        # forward
        outputs = model(input_ids=batch)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits

        # ---------- NaN / Inf debug on logits ----------
        if not torch.isfinite(logits).all():
            print(f"[NaN DEBUG] non-finite logits at step {i}")
            print("  any NaN:", torch.isnan(logits).any().item(),
                  "any Inf:", torch.isinf(logits).any().item())
            print("  logits min/max:",
                  logits.min().item(),
                  logits.max().item())
            print("  batch input_ids min/max:",
                  batch.min().item(),
                  batch.max().item())
            raise RuntimeError("NaN/Inf in logits; see [NaN DEBUG]")
        # ------------------------------------------------

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        # ---------- NaN / Inf debug on loss ----------
        if not torch.isfinite(loss):
            print(f"[NaN DEBUG] non-finite loss at step {i}")
            print("  loss:", loss.item())
            print("  shift_logits min/max:",
                  shift_logits.min().item(),
                  shift_logits.max().item())
            raise RuntimeError("NaN/Inf in loss; see [NaN DEBUG]")
        # ---------------------------------------------

        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)

    # ---------- NaN / Inf debug on mean_loss ----------
    if not math.isfinite(mean_loss):
        print("[NaN DEBUG] non-finite mean_loss")
        print("  losses:", losses)
        raise RuntimeError("NaN/Inf in mean_loss; see [NaN DEBUG]")
    # --------------------------------------------------

    ppl = math.exp(mean_loss)
    return ppl


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    model_name,
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=1,
):
    """
    model: model name
    limit: number of test samples for debug, set to -1 is no limit
    num_fewshot: Number of examples in few-shot context
    eval_ppl: str datasets are split by , such as 'wikitext2,ptb,c4'
    """
    
    results = {}
    
    for dataset in eval_ppl.split(","):
        cache_testloader = (
            f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
        )
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader, weights_only = False)
            # print(f"load calibration from {cache_testloader}")
        else:
            testloader = get_eval_loaders(dataset, tokenizer)
            torch.save(testloader, cache_testloader)
        # print(dataset)
        testenc = testloader.input_ids
        nsamples = testenc.numel() // model.seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                model.device
            )
            outputs = model.model(batch)
            hidden_states = outputs[0]  # .to(model.lm_head.weight.device)
            logits = model.lm_head(hidden_states)  # .contiguous()
            shift_logits = logits[:, :-1, :]  # .contiguous()
            shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][
                :, 1:
            ].to(model.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)
            
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * model.seqlen))
        print(dataset, ppl.item())
        model.config.use_cache = use_cache
        # pprint(model)
        results[dataset] = ppl.item()

    return results

# Function to evaluate perplexity (ppl)
def my_eval_ppl(model, testenc, bs=1, device=None):
    model.seqlen = 2048
    
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    loss_lst = []
    print(f"nsamples: {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs)):
        # if i % 50 == 0:
        #     print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
        # loss_lst.append(loss.float())

    # Compute perplexity
    # print("neg_log_likelihood: ")
    # for i in nlls:
    #     print(float(i), end=', ')
    # print("\n")
    
    # print("loss: ")
    # for i in loss_lst:
    #     print(float(i), end=', ')
    # print("\n")
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item(), torch.stack(nlls).sum() / (nsamples * model.seqlen)