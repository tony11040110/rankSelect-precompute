import copy
from typing import Dict, List, Tuple
import json

"""
sensitivity_list =
{
    'model.layers.0.self_attn.o_proj': {
        0.1: 5.987914562225342, 
        0.2: 5.986673831939697, 
        0.3: 5.986851692199707, 
        0.4: 5.987914562225342, 
        0.5: 5.987205982208252, 
        0.6: 5.986496925354004, 
        0.7: 5.986320495605469, 
        0.8: 5.986496925354004, 
        0.9: 5.985965728759766}}
    ...
}
"""

def binary_search(data, key):
    # Data is sorted in descending order
    low = 0
    upper = len(data) - 1
    while low <= upper:
        mid = (low + upper) // 2
        if data[mid] > key:
            low = mid + 1
        elif data[mid] < key:
            upper = mid - 1
        else:
            return mid
    return -1
    
def slope(a, b, xscale):
    try:
        s = (b[1] - a[1]) / ((b[0] - a[0])*xscale)
    except ZeroDivisionError:
        print("[linear_prog] Divided by zero:", a, b)
        exit()
    return s


def calculate_slope(param_ppl_pairs, layer_type, base_ppl, raw_linear):
    
    if layer_type not in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        raise ValueError(f"Unsupported layer type: {layer_type}")
    
    sorted_points = sorted(param_ppl_pairs, key=lambda x: -x[0])
    
    # (param_ratio, slope)
    new_points = []
    for i in range(len(sorted_points)):
        weight_param = raw_linear.weight.numel()
        if i == 0:
            s = slope((1, base_ppl), sorted_points[i], weight_param)
        else:
            s = slope(sorted_points[i], sorted_points[i-1], weight_param)
        
        new_points.append((sorted_points[i][0], -s))
    
    new_points.append((0.0, float('inf')))
    
    return new_points

def make_convex(param_ppl_pairs, base_ppl):
    """
    param_ppl_pairs = 
    [   
        (0.95, 5.875667572021484),
        (0.9, 5.986143112182617),
        (0.85, 5.875600814819336),
        (0.8, 5.986496925354004),
        (0.75, 5.875739097595215),
        ...
    ]
    """
    
    # Sort by ascending param_ratio
    sorted_points = sorted(param_ppl_pairs, key=lambda x: x[0])
    sorted_points.append((1.0, base_ppl))
    result_list = []

    def cross(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        # print(f"cross: {a[0]*b[1] - a[1]*b[0]}")
        return a[0]*b[1] - a[1]*b[0]
    
    def diff(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
        return (b[0]-a[0], b[1]-a[1])
    
    # Find the strictly decreasing line of the dots
    # Modified by Andrew's monotone chain
    
    result_list.append(sorted_points[0])
    
    # Force to go down first
    mid_pnt_id = 0
    for i in range(1, len(sorted_points)):
        if slope(result_list[-1], sorted_points[i], xscale=1) < 0:
            result_list.append(sorted_points[i])
            mid_pnt_id = i
            break
        elif slope(result_list[-1], sorted_points[i], xscale=1) == 0:
            result_list.append(sorted_points[i])
            # if there is no dots in list
            if i == len(sorted_points)-1:
                mid_pnt_id = i
        else:
            continue
            
    # print(result_list)
    
    for point in sorted_points[mid_pnt_id+1: len(sorted_points)]:
        if result_list[-1][0] > point[0]:
            print(f"Up convex: {result_list[-1][0]} > {point[0]}")
            break
        while len(result_list) > 2 and cross(diff(result_list[-2], result_list[-1]), diff(result_list[-2], point)) <= 0:
            # print(len(result_list), result_list)
            # print(f"pop: {result_list[-1]}")
            result_list.pop()
        result_list.append(copy.deepcopy(point))
    
    if result_list[-1][0] == 1.0:
        result_list.pop()
            
    return result_list


def calculate_param_ratio(module_dict, sen_list, select_record):
    remain_param = 0
    tot_params = 0
    for layer in select_record.keys(): 
        cursor = select_record[layer]
        if cursor == -1:
            ratio = 1.0
        else:
            ratio = sen_list[layer][cursor][0]

        raw_linear = module_dict[layer]
        tot_params += raw_linear.weight.numel()
        remain_param += raw_linear.weight.numel() * ratio
    
    # NOTE: About model total parameter calculation
    # Old code: lm_head params are added to the total parameters
    # lm_head size depends on the model
    # param_ratio = (remain_param + CALIBRATION) / TOTAL_PARAM
    # New code: lm_head params are not added to the total parameters
    param_ratio = remain_param / tot_params
    return param_ratio

def rank_selection(module_dict, sen_list, select_record, target_ratio=0.8, log=False):
    param_ratio = calculate_param_ratio(module_dict=module_dict, sen_list=sen_list, select_record=select_record)
    print("Init param_ratio: ", param_ratio)
    
    while param_ratio > target_ratio:
        # Select candidate
        candidate = ""
        min_slope = float('inf')
        for layer in select_record.keys():
            cursor = select_record[layer] + 1
            # print(layer, cursor, len(sen_list[layer]))
            slope = sen_list[layer][cursor][1]
            if sen_list[layer][cursor][1] < min_slope:
                candidate = layer
                min_slope = slope
        
        # Update the select record
        select_record[candidate] += 1
        
        # Calculate the parameter ratio
        param_ratio = calculate_param_ratio(module_dict=module_dict, sen_list=sen_list, select_record=select_record)
        # print(f"param_ratio: {param_ratio:.4f}")
        
        if param_ratio < target_ratio:
            # Rollback the select record
            select_record[candidate] -= 1
            break
        else:
            pass
            # print(f"select {candidate}, ratio {sen_list[candidate][select_record[candidate]][0]:.2f}, tot_param_ratio {param_ratio:.4f}")
    
    # Map the cursor to ratio
    for layer in select_record.keys():
        raw_linear = module_dict[layer]
        idx = select_record[layer]
        if idx >= 0:
            ratio = sen_list[layer][idx][0]
            select_record[layer] = ratio
        elif idx == -1:
            select_record[layer] = 1.0
        else:
            raise ValueError("Unvalid index")
            
    return select_record, param_ratio

def predict_ppl(select_record, raw_sen_list, base_ppl):
    
    sen_list = copy.deepcopy(raw_sen_list)
    
    # Preprocessing (dict to list)
    for key in sen_list.keys():
        if isinstance(sen_list[key], dict):
            sen_list[key] = list(sen_list[key].items())
    # select_recort item: (layer, cursor)
    acc_error = 0
    
    acc_errors = {}
    
    for layer, cursor in select_record.items():
        if cursor == -1.0:
            continue
        else:
            acc_error += sen_list[layer][cursor][1] - base_ppl
            acc_errors[layer] = (sen_list[layer][cursor][0], sen_list[layer][cursor][1], sen_list[layer][cursor][1] - base_ppl)
            # print(f"{layer}: {sen_list[layer][cursor][1]}, {sen_list[layer][cursor][1] - base_ppl}")
    
    return base_ppl + acc_error, acc_errors
    

def succinct_calib(module_dict, sen_list):
    # Coordination transformation (param ratio dimention)
    # Transform the param ratio to that of succinct version
    
    calib_list = copy.deepcopy(sen_list)
    # Extract a param ratio list
    for name, lst in sen_list.items():
        linear_type = name.split('.')[-1]
        param_ratios, ppls = zip(*lst)
        
        raw_linear = module_dict[name]
        
        n_params = raw_linear.weight.numel()
        in_plus_out = raw_linear.in_features + raw_linear.out_features

        # Mapping
        new_ratios = []
        for ratio in param_ratios:
            compressed_params = int(n_params * ratio)
            rank = compressed_params // in_plus_out
            new_ratio = (rank * in_plus_out - rank ** 2) / n_params
            new_ratios.append(new_ratio)
        

        calib_list[name] = list(zip(new_ratios, ppls))
    return calib_list
