from linear_prog import (
    make_convex,
    calculate_slope,
    rank_selection,
    )
import copy
from utils.calc import rank_to_param_ratio

# Only search
def greedy_search_truncation_rank(module_dict, raw_sensitivity_dict, base_ppl, args, do_succinct_calib=False, param_ratio_target=0.9, step_type="param_ratio"):
    
    lst_copy = copy.deepcopy(raw_sensitivity_dict)
    
    if step_type == "rank":
        # Map ratio to rank
        if args.search_with_succinct:
            sensitivity_list, mapping = rank_to_param_ratio(module_dict, lst_copy, succinct=True)
        else:
            sensitivity_list, mapping = rank_to_param_ratio(module_dict, lst_copy, succinct=False)
    elif step_type == "param_ratio":
        sensitivity_list = lst_copy
    else:
        raise ValueError("Unvalid step type")
    
    for key in sensitivity_list.keys():
        if isinstance(sensitivity_list[key], dict):
            sensitivity_list[key] = list(sensitivity_list[key].items())
        else:
            raise ValueError("Raw sensitivity list is not a dict")

    # Remove lm_head
    if "lm_head" in sensitivity_list:
        sensitivity_list.pop("lm_head")
    
    # Make convex
    for layer, layer_list in sensitivity_list.items():
        sensitivity_list[layer] = make_convex(layer_list, base_ppl)
    
    # Calculate slope
    for layer, layer_list in sensitivity_list.items():
        layer_type = layer.split('.')[-1]
        # print(layer)
        sensitivity_list[layer] = calculate_slope(sensitivity_list[layer], layer_type, base_ppl, module_dict[layer])
    
    
    # Rank selection
    select_record = {}
    for layer in sensitivity_list.keys():
        select_record[layer] = -1
    
    selection_ratio, param_ratio = rank_selection(module_dict, sensitivity_list, select_record, target_ratio=param_ratio_target)
    print(f"param_ratio: {param_ratio}")
    
    selection_rank = copy.deepcopy(selection_ratio)
    
    # Map the selected ratio to rank
    for layer, ratio in selection_rank.items():
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
    
    return selection_rank, selection_ratio


