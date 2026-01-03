import os
import torch
import torch.nn as nn
from modules.svd_linear import SVDLinear
from whiten_utils import find_layers
from evaluate_utils import evaluate_model, evaluate_perplexity
from tqdm import tqdm
import numpy as np
import click

def _resolve_eval_device(args, model):
    """Resolve target device for sensitivity/perplexity evaluation.

    Priority:
      1) args.device if provided and not 'auto'
      2) CUDA if available (cuda:0 respects CUDA_VISIBLE_DEVICES)
      3) model's current parameter device
    """
    dev = getattr(args, "device", None)
    if isinstance(dev, str) and dev and dev.lower() != "auto":
        try:
            return torch.device(dev)
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
from pathlib import Path

import math
import torch
import torch.nn as nn

@torch.no_grad()
def _evaluate_perplexity_from_batches(model, input_batches, device=None):
    """Evaluate perplexity on a list/iterable of token batches.
    Notes:
      - We explicitly control the eval device to avoid CPU/CUDA mismatch when some
        tensors/buffers are on a different device.
      - `input_batches` items are expected to be 1D or 2D Long tensors containing token ids.
    """
    # Prefer the embedding device (more reliable than next(model.parameters()) if some buffers live on CPU)
    if device is None:
        try:
            device = model.get_input_embeddings().weight.device
        except Exception:
            # fallback
            device = next(model.parameters()).device
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for input_ids in input_batches:
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.numel() == 0:
            continue

        # Ensure shape [B, T]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(device, non_blocking=True)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits

        # shift: predict token t+1 from t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="sum")
        nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_nll += float(nll.item())
        total_tokens += int(shift_labels.numel())

    if total_tokens == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_nll / total_tokens)).item())

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    total_nll = 0.0
    total_tokens = 0

    for input_ids in input_batches:
        # 每次只把一個 batch 丟上 GPU
        input_ids = input_ids.to(device)  # [B, L]

        # 關掉 cache，省記憶體
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # shift 一格當 LM label
        shift_logits = logits[..., :-1, :].contiguous()   # [B, L-1, V]
        shift_labels = input_ids[..., 1:].contiguous()    # [B, L-1]

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # [(B*(L-1)), V]
            shift_labels.view(-1),                         # [(B*(L-1))]
        )
        # 這個 batch 的 NLL
        batch_nll = loss.sum().item()
        total_nll += batch_nll
        total_tokens += shift_labels.numel()

        # 釋放暫存
        del input_ids, outputs, logits, shift_logits, shift_labels, loss
        torch.cuda.empty_cache()

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return float(ppl)

@torch.no_grad()
def get_calib_sensitivity_ratio(model, calib_loader, args, use_cache=True, step=0.1):
    model_id = model.config._name_or_path
    if args.method == "asvd":
        cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.method}_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_step_{step}.pt"
    else:
        cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.method}_{args.n_calib_samples}_{args.calib_dataset}_step_{step}.pt"
    
    click.secho(f"[Sensitivity_list] Search cache_file={cache_file}", fg="yellow")
    if os.path.exists(cache_file) and use_cache:
        click.secho(f"File {cache_file} exist.", fg="green")
        click.secho(f"Load cache_file={cache_file}", fg="yellow")
        saves_dict = torch.load(cache_file, map_location="cpu")
        base_ppl = saves_dict["base_ppl"]
        sensitivity_dict = saves_dict["sensitivity_dict"]
        return sensitivity_dict, base_ppl
    model.eval()
    
    click.secho(f"[Sensitivity_list] No cache_file={cache_file}", fg="red")
    click.secho(f"[Sensitivity_list] Create sensitivity list...", fg="yellow")

    device = _resolve_eval_device(args, model)
    model = model.to(device)
    click.secho(f"[Sensitivity_list] eval_device={device}", fg="cyan", dim=True)

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)
    
    # Evaluate the ppl of the evaluation samples
    eval_input_ids = torch.cat([calib_loader[i]["input_ids"] for i in range(args.n_calib_samples)], 0) 
    base_ppl = _evaluate_perplexity_from_batches(model, calib_batches, device=device)
    click.secho(f"[Sensitivity] base_ppl ppl: {base_ppl}", fg="yellow")
    click.secho(f"[Sensitivity] eval_input_ids.shape={eval_input_ids.shape}", fg="yellow")

    sensitivity_dict = {}
    
    # generate a list in range 0 to 1 with step 0.01
    param_ratio_candidates = np.arange(step, 1.0, step=step).tolist()
    # Round to 2 decimal places
    param_ratio_candidates = [round(_, 2) for _ in param_ratio_candidates]
    
    
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        sensitivity_dict[info["full_name"]] = {}
        for param_ratio in param_ratio_candidates:
            # Different methods implementation
            if args.method == "asvd":
                svd_linear = SVDLinear.from_linear(
                    raw_linear,
                    param_ratio=param_ratio,
                    alpha=args.alpha,
                    act_aware=True,
                )
            elif args.method == "whiten":
                svd_linear = SVDLinear.from_linear_whiten(
                    raw_linear,
                    param_ratio=param_ratio
                )
            elif args.method == "svd":
                svd_linear = SVDLinear.from_linear(
                    raw_linear,
                    param_ratio=param_ratio,
                    act_aware=False,
                )
            setattr(info["father"], info["name"], svd_linear)

            ppl = _evaluate_perplexity_from_batches(model, calib_batches, device=device)
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"{info['full_name']} {param_ratio} {ppl}")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    
    save_sensitivity_dict = {
        "base_ppl": base_ppl,
        "sensitivity_dict": sensitivity_dict
    }
    
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_sensitivity_dict, cache_file)
    click.secho(f"[Sensitivity] Save the sensitivity list to:  {cache_file}", fg="yellow")
    return sensitivity_dict, base_ppl

@torch.no_grad()
def get_calib_sensitivity_step_rank(model, calib_loader, args, use_cache=True, rankstep=128):
    model_id = model.config._name_or_path
    
    # Optional: local sensitivity around a warmup rank config (7-point search).
    warmup_rank_dict = getattr(args, "warmup_rank_dict", None)
    local_points = int(getattr(args, "local_points", 0) or 0)
    use_local = isinstance(warmup_rank_dict, dict) and local_points > 0

    warmup_tag = ""
    if use_local:
        # Include baseline tag in cache name to avoid collisions across warmups.
        base = getattr(args, "baseline_config", None)
        if isinstance(base, str) and len(base) > 0:
            base_tag = os.path.splitext(os.path.basename(base))[0]
        else:
            base_tag = "warmup"
        warmup_tag = f"_local{local_points}_{base_tag}"

    if args.method == "asvd":
        cache_file = (
            f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.method}_{args.scaling_method}_{args.alpha}_"
            f"{args.n_calib_samples}_{args.calib_dataset}_rankstep_{rankstep}{warmup_tag}.pt"
        )
    else:
        cache_file = (
            f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.method}_{args.n_calib_samples}_"
            f"{args.calib_dataset}_rankstep_{rankstep}{warmup_tag}.pt"
        )
    
    click.secho(f"Search cache_file={cache_file}", fg="yellow")
    if os.path.exists(cache_file) and use_cache:
        click.secho(f"File {cache_file} exist.", fg="green")
        click.secho(f"Load cache_file={cache_file}", fg="yellow")
        saves_dict = torch.load(cache_file, map_location="cpu")
        base_ppl = saves_dict["base_ppl"]
        sensitivity_dict = saves_dict["sensitivity_dict"]
        return sensitivity_dict, base_ppl
    model.eval()
    
    click.secho(f"No cache_file={cache_file}", fg="red")
    click.secho(f"Create sensitivity list...", fg="yellow")

    device = _resolve_eval_device(args, model)
    # Move whole model so downstream evaluators (which infer device from model params) run on GPU.
    model = model.to(device)
    click.secho(f"[Sensitivity] eval_device={device}", fg="cyan", dim=True)
    
    calib_batches = [calib_loader[i]["input_ids"] for i in range(args.n_calib_samples)]
    base_ppl = _evaluate_perplexity_from_batches(model, calib_batches, device=device)

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    # Evaluate the ppl of the evaluation samples
    eval_input_ids = torch.cat([calib_loader[i]["input_ids"] for i in range(args.n_calib_samples)], 0) 
    base_ppl = _evaluate_perplexity_from_batches(model, calib_batches, device=device)
    click.secho(f"[Sensitivity] base_ppl: {base_ppl}", fg="yellow")
    click.secho(f"[Sensitivity] eval_input_ids.shape={eval_input_ids.shape}", fg="yellow")
    
    sensitivity_dict = {}

    # Pre-compute total eval jobs for tqdm.
    num_eval = 0
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        max_rank = min(raw_linear.weight.shape[0], raw_linear.weight.shape[1])
        if use_local:
            # Approximate: local_points per layer (dedup/clamp may reduce slightly).
            num_eval += int(local_points)
        else:
            num_eval += max(0, (max_rank // int(rankstep)) - 1)
    
    pbar = tqdm(total=num_eval)
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        sensitivity_dict[info["full_name"]] = {}
        
        max_rank = min(raw_linear.weight.shape[0], raw_linear.weight.shape[1])

        if use_local:
            # 7-point local sweep around warmup rank: k0 + t*rankstep, t in [-3..3]
            k0 = int(warmup_rank_dict.get(info["full_name"], max_rank))
            # Clamp k0 into valid range.
            k0 = max(1, min(k0, max_rank))
            half = local_points // 2
            candidates = [k0 + (t * rankstep) for t in range(-half, half + 1)]
            # Clamp & deduplicate
            rank_candidates = sorted({max(1, min(int(k), max_rank)) for k in candidates})
        else:
            rank_candidates = [i for i in range(rankstep, max_rank, rankstep)]
        
        for rank in rank_candidates:
            # Different methods implementation
            if args.method == "asvd":
                svd_linear = SVDLinear.from_linear_rank(
                    raw_linear,
                    name=info["full_name"],
                    rank=rank,
                    alpha=args.alpha,
                    act_aware=True,
                )
            elif args.method == "whiten":
                svd_linear = SVDLinear.from_linear_whiten_rank(
                    raw_linear,
                    name=info["full_name"],
                    rank=rank,
                )
            elif args.method == "svd":
                svd_linear = SVDLinear.from_linear_rank(
                    raw_linear,
                    name=info["full_name"],
                    rank=rank,
                    act_aware=False,
                )
            setattr(info["father"], info["name"], svd_linear)

            ppl = _evaluate_perplexity_from_batches(model, calib_batches, device=device)
            sensitivity_dict[info["full_name"]][rank] = ppl
            print(f"{info['full_name']} {rank} {ppl}")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    
    save_sensitivity_dict = {
        "base_ppl": base_ppl,
        "sensitivity_dict": sensitivity_dict
    }
    
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_sensitivity_dict, cache_file)
    click.secho(f"[Sensitivity] Save the sensitivity list to:  {cache_file}", fg="yellow")
    return sensitivity_dict, base_ppl

@torch.no_grad()
@torch.no_grad()
@torch.no_grad()

def get_calib_sensitivity_step_rank_compressed_baseline(model, calib_loader, args, use_cache=True, rankstep=128):
    '''
    Compressed-baseline sensitivity (local-7).

    Builds a baseline-compressed model using args.warmup_rank_dict (from --baseline_config),
    evaluates base_ppl on that baseline model, then performs a local rank sweep (e.g. 7 points)
    around each layer's baseline rank while keeping all other layers at the baseline ranks.

    Finally, restores the model back to original dense nn.Linear modules.
    '''
    model_id = model.config._name_or_path

    warmup_rank_dict = getattr(args, "warmup_rank_dict", None)
    local_points = int(getattr(args, "local_points", 0) or 0)
    if not (isinstance(warmup_rank_dict, dict) and local_points > 0):
        raise ValueError(
            "compressed-baseline sensitivity requires args.warmup_rank_dict and args.local_points>0 "
            "(set by llm_rs.py when --baseline_config is provided)."
        )

    base = getattr(args, "baseline_config", None)
    if isinstance(base, str) and len(base) > 0:
        base_tag = os.path.splitext(os.path.basename(base))[0]
    else:
        base_tag = "warmup"
    warmup_tag = f"_basecompressed_local{local_points}_{base_tag}"

    if args.method == "asvd":
        cache_file = (
            f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.method}_{args.scaling_method}_{args.alpha}_"
            f"{args.n_calib_samples}_{args.calib_dataset}_rankstep_{rankstep}{warmup_tag}.pt"
        )
    else:
        cache_file = (
            f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.method}_{args.n_calib_samples}_"
            f"{args.calib_dataset}_rankstep_{rankstep}{warmup_tag}.pt"
        )

    click.secho(f"[Sensitivity(base=compressed)] cache_file={cache_file}", fg="yellow")
    if os.path.exists(cache_file) and use_cache:
        click.secho(f"[Sensitivity(base=compressed)] Cache hit: {cache_file}", fg="green")
        saved = torch.load(cache_file, map_location="cpu")
        return saved["sensitivity_dict"], saved["base_ppl"]

    device = _resolve_eval_device(args, model)
    model = model.to(device)
    click.secho(f"[Sensitivity_list] eval_device={device}", fg="cyan", dim=True)

    # Prepare eval batches
    max_eval_samples = args.n_calib_samples
    calib_batches = []
    for i in range(max_eval_samples):
        batch = calib_loader[i]
        calib_batches.append(batch["input_ids"])

    layers = find_layers(model)
    module_dict = {name: m for name, m in model.named_modules()}

    # Build linear_info list
    linear_info_list = []
    for full_name, raw_linear in layers.items():
        if not isinstance(raw_linear, nn.Linear):
            continue
        if full_name == "lm_head":
            continue

        father_name = ".".join(full_name.split(".")[:-1])
        child_name = full_name.split(".")[-1]
        father = module_dict[father_name] if father_name in module_dict else model
        linear_info_list.append(
            {"full_name": full_name, "father": father, "name": child_name, "raw_linear": raw_linear}
        )

    # 1) Replace ALL layers with baseline-compressed modules
    baseline_modules = {}
    for info in linear_info_list:
        full_name = info["full_name"]
        raw_linear = info["raw_linear"]
        H, W = raw_linear.weight.shape
        max_rank = min(H, W)

        k0 = int(warmup_rank_dict.get(full_name, max_rank))
        k0 = max(1, min(k0, max_rank))

        if args.method == "asvd":
            svd_base = SVDLinear.from_linear_rank(
                raw_linear, name=full_name, rank=k0, alpha=args.alpha, act_aware=True
            )
        elif args.method == "whiten":
            svd_base = SVDLinear.from_linear_whiten_rank(raw_linear, name=full_name, rank=k0)
        elif args.method == "svd":
            svd_base = SVDLinear.from_linear_rank(raw_linear, name=full_name, rank=k0, act_aware=False)
        else:
            raise ValueError(f"Unsupported method={args.method} in compressed-baseline sensitivity")

        baseline_modules[full_name] = svd_base
        setattr(info["father"], info["name"], svd_base)

    base_ppl = _evaluate_perplexity_from_batches(model, calib_batches, device=device)
    click.secho(f"[Sensitivity(base=compressed)] base_ppl = {base_ppl}", fg="yellow")

    sensitivity_dict = {}
    half = local_points // 2

    for info in tqdm(linear_info_list, desc="[Sensitivity(base=compressed)] local sweep"):
        full_name = info["full_name"]
        raw_linear = info["raw_linear"]
        base_module = baseline_modules[full_name]

        H, W = raw_linear.weight.shape
        max_rank = min(H, W)
        k0 = int(warmup_rank_dict.get(full_name, max_rank))
        k0 = max(1, min(k0, max_rank))

        candidates = [k0 + (t * rankstep) for t in range(-half, half + 1)]
        rank_candidates = sorted({max(1, min(int(k), max_rank)) for k in candidates})

        layer_sens = {}
        for rank in rank_candidates:
            if args.method == "asvd":
                svd_linear = SVDLinear.from_linear_rank(
                    raw_linear, name=full_name, rank=rank, alpha=args.alpha, act_aware=True
                )
            elif args.method == "whiten":
                svd_linear = SVDLinear.from_linear_whiten_rank(raw_linear, name=full_name, rank=rank)
            elif args.method == "svd":
                svd_linear = SVDLinear.from_linear_rank(raw_linear, name=full_name, rank=rank, act_aware=False)
            else:
                raise ValueError(f"Unsupported method={args.method} in compressed-baseline sensitivity")

            setattr(info["father"], info["name"], svd_linear)
            ppl = _evaluate_perplexity_from_batches(model, calib_batches, device=device)
            layer_sens[int(rank)] = float(ppl)

            setattr(info["father"], info["name"], base_module)

        sensitivity_dict[full_name] = layer_sens

    # 3) Restore original dense linears
    for info in linear_info_list:
        setattr(info["father"], info["name"], info["raw_linear"])

    save_sensitivity_dict = {"base_ppl": base_ppl, "sensitivity_dict": sensitivity_dict}
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_sensitivity_dict, cache_file)
    click.secho(f"[Sensitivity(base=compressed)] Save the sensitivity list to:  {cache_file}", fg="yellow")
    return sensitivity_dict, base_ppl


@torch.no_grad()
def truncate_sensitivity_list(module_dict, sensitivity_dict, step_rank):
    import copy
    
    new_sensitivity_dict = copy.deepcopy(sensitivity_dict)
    
    for layer, lists in sensitivity_dict.items():
        raw_linear = module_dict[layer]
        H, W = raw_linear.weight.shape
        
        if (H * W) % (H + W) == 0:
            truncate_rank = int(((H * W) / (H + W) // step_rank) - 1) * step_rank
        else:
            truncate_rank = int(((H * W) / (H + W)) // step_rank) * step_rank
        
        # print(layer, "truncate rank: ", truncate_rank)
        
        new_lists = copy.deepcopy(lists)
        for rank in lists.keys():
            if rank > truncate_rank:
                new_lists.pop(rank)
        new_sensitivity_dict[layer] = new_lists
    
    return new_sensitivity_dict