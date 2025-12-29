import torch
import copy
from utils.calc import rank_to_param_ratio


def binary_search_truncation_rank(module_dict, raw_sensitivity_dict, calib_loader, step_type, args):
    
    lst_copy = copy.deepcopy(raw_sensitivity_dict)
    
    if step_type == "rank":
        # Change rank to ratio in sensitivity list for searching
        # mapping: record the mapping of ratio to rank
        if args.search_with_succinct:
            sensitivity_dict, mapping = rank_to_param_ratio(module_dict, lst_copy, succinct=True)
        else:
            sensitivity_dict, mapping = rank_to_param_ratio(module_dict, lst_copy, succinct=False)
    elif step_type == "param_ratio":
        sensitivity_dict = lst_copy
    else:
        raise ValueError("Unvalid step type")

    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for ratio, ppl in v.items():
            sensitivity_list.append((layername, ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    # assert args.ppl_target > 0 or args.param_ratio_target > 0

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
        for layername, ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
        tot_params = 0
        compress_params = 0

        for layername, ratio in layers_min_ratio.items():
            raw_linear = module_dict[layername]
            tot_params += raw_linear.weight.numel()
            compress_params += raw_linear.weight.numel() * ratio
        param_ratio = compress_params / tot_params
        msg = f"low={low} mid={mid}, high={high}, param_ratio={param_ratio}({compress_params}/{tot_params})"
        print(msg)
        if param_ratio > args.param_ratio_target:
            high = mid
        else:
            low = mid + 1

    print(f"Searching finished, decomposing layers...")
    
    layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
    
    for layername, ratio, ppl in sorted_sensitive_list[mid:]:
        layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
    
    # Map the selected ratio to rank
    selection_rank = {}
    for layer, ratio in layers_min_ratio.items():
        raw_linear = module_dict[layer]
        if ratio >= 1.0:
            full_rank = min(raw_linear.in_features, raw_linear.out_features)
            selection_rank[layer] = full_rank
        else:
            if step_type == "rank":
                # Use mapping to translate the ratio to rank
                selection_rank[layer] = mapping[layer][ratio]
            elif step_type == "param_ratio":
                n_params = raw_linear.weight.numel()
                compressed_params = int(n_params * ratio)
                rank = compressed_params // (raw_linear.in_features + raw_linear.out_features)
                selection_rank[layer] = rank
            else:
                raise ValueError("Unvalid step type")
    
    return selection_rank