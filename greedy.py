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


def _count_layer_params_multilevel(raw_linear, outer_rank, inner_rank, include_bias=False):
    """Compute parameter count of a multilevel factorized linear layer.

    raw_linear: original nn.Linear layer (to read in/out features)
    outer_rank: rank after first SVD
    inner_rank: rank after second SVD on V
    """
    in_features = raw_linear.in_features
    out_features = raw_linear.out_features
    params = (
        in_features * inner_rank +
        inner_rank * outer_rank +
        outer_rank * out_features
    )
    if include_bias and raw_linear.bias is not None:
        params += out_features
    return params


def _count_total_params_multilevel(module_dict, config, include_bias=False):
    """Sum parameters over all layers for a given multilevel rank config.

    config: {layer_name: {"outer": r, "inner": r2}}
    """
    total = 0
    for layer_name, cfg in config.items():
        raw_linear = module_dict[layer_name]
        total += _count_layer_params_multilevel(
            raw_linear,
            cfg["outer"],
            cfg["inner"],
            include_bias=include_bias,
        )
    return total


def greedy_search_truncation_rank_multilevel(
    module_dict,
    multi_sensitivity_dict,
    base_ppl,
    param_ratio_target=0.9,
    include_bias=False,
    max_steps=10_000,
):
    """Greedy search for multilevel (outer, inner) ranks per layer.

    Args:
        module_dict:     {layer_full_name: nn.Linear}
        multi_sensitivity_dict:
            {layer_full_name: {
                "max_rank": int,
                "outer_step": int,
                "inner_step": int,
                "grid": {outer_rank: {inner_rank: ppl_float}}
            }}
        base_ppl:        Perplexity of the uncompressed model.
        param_ratio_target:
            Target total parameter ratio (compressed / dense).
        include_bias:    Whether to count bias parameters.
        max_steps:       Safety cap on greedy iterations.

    Returns:
        selection_rank:
            {layer_name: {"outer": r, "inner": r2}}
        selection_ratio:
            {layer_name: param_ratio_layer}  # compressed_params_layer / dense_params_layer
    """
    # 1. Build initial config: use the largest (outer, inner) available in grid.
    current_config = {}
    for layer_name, entry in multi_sensitivity_dict.items():
        grid = entry["grid"]
        if not grid:
            continue
        outer_ranks = sorted(grid.keys())
        outer_max = outer_ranks[-1]
        inner_ranks = sorted(grid[outer_max].keys())
        inner_max = inner_ranks[-1]
        current_config[layer_name] = {"outer": int(outer_max), "inner": int(inner_max)}

    # 2. Compute dense parameter baseline and target.
    original_params = 0
    for layer_name, raw_linear in module_dict.items():
        n_params_dense = raw_linear.weight.numel()
        if include_bias and raw_linear.bias is not None:
            n_params_dense += raw_linear.bias.numel()
        original_params += n_params_dense

    target_params = original_params * float(param_ratio_target)

    # current total params under multilevel factorization
    cur_params = _count_total_params_multilevel(
        module_dict,
        current_config,
        include_bias=include_bias,
    )

    # 用 base_ppl 當全域起點，只做近似的 Δppl 累積（目前沒回傳，只用來 debug）
    current_ppl = float(base_ppl)

    step = 0

    while step < max_steps:
        step += 1

        if cur_params <= target_params:
            break

        best_action = None
        best_score = None

        # 掃過所有 layer，對每層試兩個 action：outer↓ / inner↓
        for layer_name, cfg in current_config.items():
            if layer_name not in multi_sensitivity_dict:
                continue
            entry = multi_sensitivity_dict[layer_name]
            grid = entry["grid"]
            if not grid:
                continue

            raw_linear = module_dict[layer_name]
            outer_step = int(entry["outer_step"]) or 1
            inner_step = int(entry["inner_step"]) or 1

            outer_cur = cfg["outer"]
            inner_cur = cfg["inner"]

            # 確保目前這個 (outer_cur, inner_cur) 有在 grid 裡
            if outer_cur not in grid or inner_cur not in grid[outer_cur]:
                continue
            base_ppl_layer = float(grid[outer_cur][inner_cur])

            # ----- Action A: 降 outer rank -----
            new_outer = outer_cur - outer_step
            if new_outer in grid and new_outer > 0:
                # 在 new_outer 這一排選一個合法的 inner
                inner_keys = sorted(k for k in grid[new_outer].keys() if k > 0 and k <= new_outer)
                if inner_keys:
                    # 優先挑 <= inner_cur 且最接近的
                    candidates = [k for k in inner_keys if k <= inner_cur]
                    if candidates:
                        new_inner_for_outer = max(candidates)
                    else:
                        new_inner_for_outer = inner_keys[0]

                    if new_inner_for_outer in grid[new_outer]:
                        new_ppl = float(grid[new_outer][new_inner_for_outer])

                        old_params = _count_layer_params_multilevel(
                            raw_linear,
                            outer_cur,
                            inner_cur,
                            include_bias=include_bias,
                        )
                        new_params = _count_layer_params_multilevel(
                            raw_linear,
                            new_outer,
                            new_inner_for_outer,
                            include_bias=include_bias,
                        )
                        delta_params = old_params - new_params
                        delta_ppl = new_ppl - base_ppl_layer

                        if delta_params > 0:
                            score = delta_ppl / float(delta_params)
                            action = {
                                "layer": layer_name,
                                "type": "outer",
                                "old_outer": outer_cur,
                                "old_inner": inner_cur,
                                "new_outer": int(new_outer),
                                "new_inner": int(new_inner_for_outer),
                                "delta_ppl": float(delta_ppl),
                                "delta_params": int(delta_params),
                                "new_ppl": new_ppl,
                            }
                            if best_score is None or score < best_score:
                                best_score = score
                                best_action = action

            # ----- Action B: 降 inner rank -----
            new_inner = inner_cur - inner_step
            if new_inner > 0 and outer_cur in grid:
                inner_keys = sorted(k for k in grid[outer_cur].keys() if k > 0 and k <= outer_cur)
                candidates = [k for k in inner_keys if k <= new_inner]
                if candidates:
                    new_inner_adj = max(candidates)
                    if new_inner_adj in grid[outer_cur]:
                        new_ppl = float(grid[outer_cur][new_inner_adj])

                        old_params = _count_layer_params_multilevel(
                            raw_linear,
                            outer_cur,
                            inner_cur,
                            include_bias=include_bias,
                        )
                        new_params = _count_layer_params_multilevel(
                            raw_linear,
                            outer_cur,
                            new_inner_adj,
                            include_bias=include_bias,
                        )
                        delta_params = old_params - new_params
                        delta_ppl = new_ppl - base_ppl_layer

                        if delta_params > 0:
                            score = delta_ppl / float(delta_params)
                            action = {
                                "layer": layer_name,
                                "type": "inner",
                                "old_outer": outer_cur,
                                "old_inner": inner_cur,
                                "new_outer": outer_cur,
                                "new_inner": int(new_inner_adj),
                                "delta_ppl": float(delta_ppl),
                                "delta_params": int(delta_params),
                                "new_ppl": new_ppl,
                            }
                            if best_score is None or score < best_score:
                                best_score = score
                                best_action = action

        if best_action is None:
            # 沒有任何合法 move，可以停了
            break

        # 套用這一步選中的 action
        layer = best_action["layer"]
        cfg = current_config[layer]
        cfg["outer"] = best_action["new_outer"]
        cfg["inner"] = best_action["new_inner"]

        cur_params -= best_action["delta_params"]
        current_ppl += best_action["delta_ppl"]

    # 4. 整理輸出：每層 (outer, inner) 以及對 dense 參數的比例
    selection_rank = {}
    selection_ratio = {}
    final_params = 0

    for layer_name, cfg in current_config.items():
        raw_linear = module_dict[layer_name]
        dense_params = raw_linear.weight.numel()
        if include_bias and raw_linear.bias is not None:
            dense_params += raw_linear.bias.numel()

        compressed_params = _count_layer_params_multilevel(
            raw_linear,
            cfg["outer"],
            cfg["inner"],
            include_bias=include_bias,
        )

        selection_rank[layer_name] = {
            "outer": int(cfg["outer"]),
            "inner": int(cfg["inner"]),
        }
        selection_ratio[layer_name] = float(compressed_params) / float(dense_params)
        final_params += compressed_params

    if original_params > 0:
        global_ratio = float(final_params) / float(original_params)
    else:
        global_ratio = 0.0

    print(f"[MultiLevel Greedy] target_ratio={param_ratio_target:.4f}, final_ratio={global_ratio:.4f}")

    return selection_rank, selection_ratio
