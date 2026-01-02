import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import math

from datautils import get_eval_loaders
from datasets import load_dataset
import time
import re


@torch.no_grad()
def evaluate_perplexity(model, input_ids: torch.Tensor, seqlen: int = 2048, micro_batch: int = 1):
    """Robust perplexity evaluation.

    Supports both:
      - input_ids: [1, L]  (single long stream)
      - input_ids: [B, L]  (B independent sequences)

    Accumulates token-level negative log-likelihood (sum) and returns exp(mean_nll).
    Skips chunks with length < 2 (cannot form next-token labels).
    """
    model.eval()
    device = next(model.parameters()).device

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    elif input_ids.dim() != 2:
        raise ValueError(f"evaluate_perplexity: expected 1D/2D input_ids, got shape={tuple(input_ids.shape)}")

    input_ids = input_ids.to(torch.long)

    B, L = input_ids.shape
    micro_batch = max(1, int(micro_batch))
    micro_batch = min(micro_batch, B)

    total_nll = 0.0
    total_tokens = 0

    for b0 in range(0, B, micro_batch):
        batch_cpu = input_ids[b0 : b0 + micro_batch]
        for t0 in range(0, L, seqlen):
            t1 = min(L, t0 + seqlen)
            chunk = batch_cpu[:, t0:t1]
            if chunk.shape[1] < 2:
                continue

            chunk = chunk.to(device, non_blocking=True)
            outputs = model(input_ids=chunk, use_cache=False)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()

            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tokens += int(shift_labels.numel())

    if total_tokens == 0:
        return float('nan')

    mean_nll = total_nll / total_tokens
    return float(math.exp(mean_nll))


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