import os
import torch
import torch.nn as nn
import click
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from modules.svd_linear import SVDLinear


def calib_fisher_info(model, calib_loader, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_fisher_info.pt"
    if os.path.exists(cache_file) and use_cache:
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        return
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = 0

    # get fisher info
    for batch in tqdm(calib_loader):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info += module.weight.grad.detach().pow(2).mean(0)
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info
    torch.save(all_fisher_info, cache_file)


@torch.no_grad()
def calib_input_distribution(model, calib_loader, method, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = (
        f"cache/asvd/{model_id.replace('/','_')}_calib_input_distribution_{method}.pt"
    )
    if os.path.exists(cache_file):
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.scaling_diag_matrix = all_scaling_diag_matrix[name].to(
                    module.weight.device
                )
        click.secho(f"[ASVD] Load scaling matrices from {cache_file}", fg="yellow")
        return
    click.secho(f"[ASVD] {cache_file} not found.", fg="red")
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )
        # abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
        # module.scaling_diag_matrix += abs_max

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix
    
    if not os.path.exists("cache/asvd"):
        os.makedirs("cache/asvd")
    torch.save(all_scaling_diag_matrix, cache_file)
    click.secho(
        f"[ASVD] Save scaling diag matrix to cache: {cache_file}", fg="yellow")

def compress_model_asvd(model, selection_result, args):
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
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
    
    for layername, rank in selection_result.items():
        print(layername, end=" ")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        
        if args.search_with_succinct:
            svd_linear = SVDLinear.from_linear_rank(
                raw_linear,
                rank=rank,
                name=layername,
                act_aware=True,
                sigma_fuse=args.sigma_fuse,
                succinct=True,
            ).cpu()
        else:
            svd_linear = SVDLinear.from_linear_rank(
                raw_linear,
                name=layername,
                rank=rank,
                alpha=args.alpha,
                act_aware=True,
                sigma_fuse=args.sigma_fuse,
            ).cpu()
        setattr(info["father"], info["name"], svd_linear)
        
def compress_model_svd(model, selection_result, args):
    # Compress the model
    module_dict = {name: module for name, module in model.named_modules()}
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
    
    for layername, rank in selection_result.items():
        print(layername, end=" ")
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        
        if args.search_with_succinct:
            svd_linear = SVDLinear.from_linear_rank(
                raw_linear,
                rank=rank,
                name=layername,
                act_aware=False,
                sigma_fuse=args.sigma_fuse,
                succinct=True,
            ).cpu()
        else:
            svd_linear = SVDLinear.from_linear_rank(
                raw_linear,
                name=layername,
                rank=rank,
                alpha=args.alpha,
                act_aware=False,
                sigma_fuse=args.sigma_fuse,
            ).cpu()
        setattr(info["father"], info["name"], svd_linear)