import os
import torch
import torch.nn as nn
from modules.svd_linear import SVDLinear
from modules.multilevel_svd_linear import MultiSVDLinear
from whiten_utils import find_layers
from evaluate_utils import evaluate_model, evaluate_perplexity
from tqdm import tqdm
import numpy as np
import click
from pathlib import Path

import math
import torch
import torch.nn as nn

@torch.no_grad()
def _evaluate_perplexity_from_batches(model, input_batches):
    """
    用多個小 batch 依序算 PPL，避免一次把所有 sample 丟進 GPU。
    input_batches: list[Tensor]，每個 tensor shape = [B, L]
    回傳: ppl (float)
    """
    device = next(model.parameters()).device
    model.eval()

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
    base_ppl = evaluate_perplexity(model, eval_input_ids)
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

            ppl = evaluate_perplexity(model, eval_input_ids)
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
    base_ppl = evaluate_perplexity(model, eval_input_ids)
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

            ppl = evaluate_perplexity(model, eval_input_ids)
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

    base_ppl = _evaluate_perplexity_from_batches(model, calib_batches)
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
            ppl = _evaluate_perplexity_from_batches(model, calib_batches)
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

def get_calib_sensitivity_step_rank_multilevel(
    model,
    calib_loader,
    args,
    use_cache: bool = True,
    outer_rank_step: int = 128,
    inner_rank_step: int = 128,
):
    """
    Multilevel 版本的 sensitivity：
        - outer_rank: 第一層 SVD 的 rank
        - inner_rank: 第二層（在 V 上）的 rank

    回傳：
        multi_sensitivity, base_ppl

    其中 multi_sensitivity[layer_full_name] 結構為：
        {
            "max_rank": int,
            "outer_step": int,
            "inner_step": int,
            "grid": {
                outer_rank: {
                    inner_rank: ppl_float,
                    ...
                },
            },
        }
    """
    device = next(model.parameters()).device
    model.eval()

    # 解析 uniform inner 設定（如果有）
    inner_uniform_value = getattr(args, "inner_uniform_value", None)
    if inner_uniform_value is not None:
        inner_uniform_value = int(inner_uniform_value)

    # cache 檔名
    model_id = model.config._name_or_path
    uniform_suffix = ""
    if inner_uniform_value is not None:
        uniform_suffix = f"_uni{inner_uniform_value}"

    cache_file = (
        f"cache/lists/"
        f"{model_id.replace('/','_')}_sensi_multilevel_"
        f"{args.n_calib_samples}_{args.calib_dataset}_"
        f"outer{outer_rank_step}_inner{inner_rank_step}{uniform_suffix}.pt"
    )

    click.secho(f"[MultiLevel Sensitivity] cache_file = {cache_file}", fg="yellow")

    if use_cache and os.path.exists(cache_file):
        click.secho("[MultiLevel Sensitivity] load cache", fg="green")
        saved = torch.load(cache_file, map_location="cpu")
        base_ppl = saved["base_ppl"]
        multi_sensitivity = saved["multi_sensitivity"]
        return multi_sensitivity, base_ppl

    # === 準備 eval data & base ppl ===
    max_eval_samples = args.n_calib_samples
    calib_batches = []
    for i in range(max_eval_samples):
        batch = calib_loader[i]
        calib_batches.append(batch["input_ids"])

    base_ppl = _evaluate_perplexity_from_batches(model, calib_batches)
    click.secho(f"[MultiLevel Sensitivity] base_ppl = {base_ppl}", fg="yellow")

    # === 找出所有 Linear layer ===
    layers = find_layers(model)
    module_dict = {name: m for name, m in model.named_modules()}
    multi_sensitivity = {}
    any_layer_used = False

    # ---------- 預先統計 total_jobs，供 tqdm 用 ---------- #
    total_jobs = 0
    for full_name, module in layers.items():
        if not isinstance(module, nn.Linear):
            continue
        if full_name == "lm_head":
            continue

        H, W = module.weight.shape
        max_rank = min(H, W)

        outer_ranks = list(range(outer_rank_step, max_rank + 1, outer_rank_step))
        if max_rank not in outer_ranks:
            outer_ranks.append(max_rank)
        outer_ranks = sorted(set(outer_ranks))

        if inner_uniform_value is not None:
            outer_ranks = [r for r in outer_ranks if r >= inner_uniform_value]
            if not outer_ranks:
                continue

        for outer_rank in outer_ranks:
            if outer_rank <= 0 or outer_rank > max_rank:
                continue

            if inner_uniform_value is not None:
                inner_ranks = [inner_uniform_value]
            else:
                inner_max = outer_rank
                inner_ranks = list(range(inner_rank_step, inner_max + 1, inner_rank_step))
                if inner_max not in inner_ranks:
                    inner_ranks.append(inner_max)
                inner_ranks = sorted(set(inner_ranks))

            for inner_rank in inner_ranks:
                if inner_rank <= 0 or inner_rank > outer_rank:
                    continue
                total_jobs += 1

    pbar = tqdm(
        total=total_jobs,
        desc="[MultiLevel Sensitivity]",
        dynamic_ncols=True,
    )
    # ------------------------------------------------------ #

    for full_name, module in layers.items():
        if not isinstance(module, nn.Linear):
            continue
        if full_name == "lm_head":
            continue

        father_name = ".".join(full_name.split(".")[:-1])
        child_name = full_name.split(".")[-1]
        father = module_dict[father_name] if father_name in module_dict else model

        H, W = module.weight.shape
        max_rank = min(H, W)

        # outer_rank 掃描集合
        outer_ranks = list(range(outer_rank_step, max_rank + 1, outer_rank_step))
        if max_rank not in outer_ranks:
            outer_ranks.append(max_rank)
        outer_ranks = sorted(set(outer_ranks))

        # 如果有 uniform inner，outer 需要 >= inner_uniform_value
        if inner_uniform_value is not None:
            outer_ranks = [r for r in outer_ranks if r >= inner_uniform_value]
            if not outer_ranks:
                click.secho(
                    f"[MultiLevel Sensitivity] WARNING: no valid outer_ranks "
                    f"for layer {full_name} with inner_uniform={inner_uniform_value}",
                    fg="red",
                )
                continue

        click.secho(
            f"[MultiLevel Sensitivity] layer = {full_name}, "
            f"max_rank = {max_rank}, outer_ranks = {outer_ranks}",
            fg="cyan",
        )

        layer_entry = {
            "max_rank": int(max_rank),
            "outer_step": int(outer_rank_step),
            "inner_step": int(inner_rank_step),
            "grid": {},
        }

        # 至少有一層可以參與 multilevel
        any_layer_used = True

        for outer_rank in outer_ranks:
            if outer_rank <= 0 or outer_rank > max_rank:
                continue

            # inner_rank 掃描集合
            if inner_uniform_value is not None:
                inner_ranks = [inner_uniform_value]
            else:
                inner_max = outer_rank
                inner_ranks = list(range(inner_rank_step, inner_max + 1, inner_rank_step))
                if inner_max not in inner_ranks:
                    inner_ranks.append(inner_max)
                inner_ranks = sorted(set(inner_ranks))

            click.secho(
                f"[MultiLevel Sensitivity]  layer = {full_name}, "
                f"outer_rank = {outer_rank}, inner_ranks = {inner_ranks}",
                fg="blue",
            )

            layer_entry["grid"][int(outer_rank)] = {}

            for inner_rank in inner_ranks:
                if inner_rank <= 0 or inner_rank > outer_rank:
                    continue

                # 建立 multilevel SVD module
                svd_linear = MultiSVDLinear.from_linear_whiten_rank(
                    linear=module,
                    name=full_name,
                    rank=int(outer_rank),
                    inner_rank=int(inner_rank),
                    succinct=getattr(args, "search_with_succinct", False),
                )

                # 替換進 model
                setattr(father, child_name, svd_linear)

                # 計算 ppl
                ppl_val = _evaluate_perplexity_from_batches(model, calib_batches)

                layer_entry["grid"][int(outer_rank)][int(inner_rank)] = ppl_val

                click.secho(
                    f"[MultiLevel Sensitivity]  {full_name} "
                    f"(outer = {outer_rank}, inner = {inner_rank}) "
                    f"-> ppl = {ppl_val}",
                    fg="white",
                )

                # 進度條前進一格
                pbar.update(1)

                # 換回原本 Linear
                setattr(father, child_name, module)

        multi_sensitivity[full_name] = layer_entry

    pbar.close()

    # 如果有設定 uniform inner，但沒有任何 layer 參與 multilevel，
    # 代表這個 inner_uniform_value 對整個模型是無效的。
    if inner_uniform_value is not None and not any_layer_used:
        raise ValueError(
            f"[MultiLevel Sensitivity] inner_uniform_value={inner_uniform_value} "
            f"is larger than max_rank of all layers; no layer can be compressed."
        )

    # 存 cache
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base_ppl": base_ppl,
            "multi_sensitivity": multi_sensitivity,
        },
        cache_file,
    )
    click.secho(
        f"[MultiLevel Sensitivity] save to: {cache_file}",
        fg="yellow",
    )

    return multi_sensitivity, base_ppl

    # === 準備 eval data & base ppl ===
    max_eval_samples = args.n_calib_samples
    calib_batches = []
    for i in range(max_eval_samples):
        batch = calib_loader[i]
        calib_batches.append(batch["input_ids"])

    base_ppl = _evaluate_perplexity_from_batches(model, calib_batches)
    click.secho(f"[MultiLevel Sensitivity] base_ppl = {base_ppl}", fg="yellow")

    # === 找出所有 Linear layer ===
    layers = find_layers(model)
    module_dict = {name: m for name, m in model.named_modules()}
    multi_sensitivity = {}
    any_layer_used = False

    for full_name, module in layers.items():
        if not isinstance(module, nn.Linear):
            continue
        if full_name == "lm_head":
            continue

        father_name = ".".join(full_name.split(".")[:-1])
        child_name = full_name.split(".")[-1]
        father = module_dict[father_name] if father_name in module_dict else model

        H, W = module.weight.shape
        max_rank = min(H, W)

        # outer_rank 掃描集合
        outer_ranks = list(range(outer_rank_step, max_rank + 1, outer_rank_step))
        if max_rank not in outer_ranks:
            outer_ranks.append(max_rank)
        outer_ranks = sorted(set(outer_ranks))

        # 如果有 uniform inner，outer 需要 >= inner_uniform_value
        if inner_uniform_value is not None:
            outer_ranks = [r for r in outer_ranks if r >= inner_uniform_value]
            if not outer_ranks:
                click.secho(
                    f"[MultiLevel Sensitivity] WARNING: no valid outer_ranks "
                    f"for layer {full_name} with inner_uniform={inner_uniform_value}",
                    fg="red",
                )
                continue

        click.secho(
            f"[MultiLevel Sensitivity] layer = {full_name}, "
            f"max_rank = {max_rank}, outer_ranks = {outer_ranks}",
            fg="cyan",
        )

        layer_entry = {
            "max_rank": int(max_rank),
            "outer_step": int(outer_rank_step),
            "inner_step": int(inner_rank_step),
            "grid": {},
        }

        # 至少有一層可以參與 multilevel
        any_layer_used = True

        for outer_rank in outer_ranks:
            if outer_rank <= 0 or outer_rank > max_rank:
                continue

            # inner_rank 掃描集合
            if inner_uniform_value is not None:
                inner_ranks = [inner_uniform_value]
            else:
                inner_max = outer_rank
                inner_ranks = list(range(inner_rank_step, inner_max + 1, inner_rank_step))
                if inner_max not in inner_ranks:
                    inner_ranks.append(inner_max)
                inner_ranks = sorted(set(inner_ranks))

            click.secho(
                f"[MultiLevel Sensitivity]  layer = {full_name}, "
                f"outer_rank = {outer_rank}, inner_ranks = {inner_ranks}",
                fg="blue",
            )

            layer_entry["grid"][int(outer_rank)] = {}

            for inner_rank in inner_ranks:
                if inner_rank <= 0 or inner_rank > outer_rank:
                    continue

                # 建立 multilevel SVD module
                svd_linear = MultiSVDLinear.from_linear_whiten_rank(
                    linear=module,
                    name=full_name,
                    rank=int(outer_rank),
                    inner_rank=int(inner_rank),
                    succinct=getattr(args, "search_with_succinct", False),
                )

                # 替換進 model
                setattr(father, child_name, svd_linear)

                # 計算 ppl
                ppl_val = _evaluate_perplexity_from_batches(model, calib_batches)

                layer_entry["grid"][int(outer_rank)][int(inner_rank)] = ppl_val

                click.secho(
                    f"[MultiLevel Sensitivity]  {full_name} "
                    f"(outer = {outer_rank}, inner = {inner_rank}) "
                    f"-> ppl = {ppl_val}",
                    fg="white",
                )

                # 換回原本 Linear
                setattr(father, child_name, module)

        multi_sensitivity[full_name] = layer_entry

    # 如果有設定 uniform inner，但沒有任何 layer 參與 multilevel，
    # 代表這個 inner_uniform_value 對整個模型是無效的。
    if inner_uniform_value is not None and not any_layer_used:
        raise ValueError(
            f"[MultiLevel Sensitivity] inner_uniform_value={inner_uniform_value} "
            f"is larger than max_rank of all layers; no layer can be compressed."
        )

    # 存 cache
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base_ppl": base_ppl,
            "multi_sensitivity": multi_sensitivity,
        },
        cache_file,
    )
    click.secho(
        f"[MultiLevel Sensitivity] save to: {cache_file}",
        fg="yellow",
    )

    return multi_sensitivity, base_ppl

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