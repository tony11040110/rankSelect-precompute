from modules.svd_linear import SVDLinear
import csv
import random
import string
import click

# TODO: Fix the parameter ratio calculation for flexible weight size
def rank_to_param_ratio(module_dict, sen_list, succinct=True):
    # Coordination transformation (rank dimention)
    
    calib_list = {}
    mapping = {}
    
    for name, lst in sen_list.items():
        linear_type = name.split('.')[-1]
        
        # if linear_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        #     n_params = QKVO_PARAM
        #     in_plus_out = HIDDEN_SIZE + HIDDEN_SIZE
        # elif linear_type in ['gate_proj', 'up_proj', 'down_proj']:
        #     n_params = MLP_PARAM
        #     in_plus_out = HIDDEN_SIZE + IMMEIDATE_SIZE
          
        raw_linear = module_dict[name]
        n_params = raw_linear.weight.numel()
        in_plus_out = raw_linear.in_features + raw_linear.out_features
        
        calib_list[name] = {}
        mapping[name] = {}  # ratio -> rank
        # Mapping
        for rank in lst.keys():
            if succinct:
                new_ratio = (rank * in_plus_out - rank ** 2) / n_params
            else:
                new_ratio = (rank * in_plus_out) / n_params
            calib_list[name][new_ratio] = lst[rank]
            mapping[name][new_ratio] = rank
        
    return calib_list, mapping

def catch_succinct_error(model, save_file=None):
    # Catch the transformer layer
    layers = model.model.layers
    module_dict = {}
    for name, module in model.named_modules():
        if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name:
            module_dict[name] = module
    
    # Init keys from module_dict to error_list
    error_list = {}
    for name, module in module_dict.items():
        if isinstance(module, SVDLinear):
            assert module.succinct_error is not None, f"{name} has no succinct error"
            error_list[name] = module.succinct_error
        else: # nn.Linear
            if "ALinear" in name or "BLinear" in name:
                pass
            else:
                error_list[name] = 0.0

    # Save error_list to csv
    if save_file is not None:
        save_csv = save_file + ".csv"
    else:
        save_csv = "succinct_error" . join(random.choices(string.ascii_uppercase + string.digits, k=5)) + ".csv"
    
    with open(save_csv, 'w+', newline='') as csvfile:
        fieldnames = ['module', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for name in error_list:
            writer.writerow({"module": name, "error": error_list[name]})
    click.secho(f"Error list is saved to {save_csv}", fg="yellow")

def set_uniform_truncation_rank(module_dict, linear_info, target_ratio):
    uniform_rank_list = {}
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        H, W = raw_linear.weight.shape
        rank = int((H * W * target_ratio) / (H + W))
        uniform_rank_list[info["full_name"]] = rank
    
    return uniform_rank_list