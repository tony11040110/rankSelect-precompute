from typing import Dict, Any
import copy

attention = ["q_proj", "k_proj", "v_proj", "o_proj"]
mlp = ["gate_proj", "up_proj", "down_proj"]

def init_config(num_layers) -> Dict[str, Any]:
    tmp = {
            "q_proj": 1.0,
            "k_proj": 1.0,
            "v_proj": 1.0,
            "o_proj": 1.0,
            "gate_proj": 1.0,
            "up_proj": 1.0,
            "down_proj": 1.0
        }
    config = {
        "blocks": []
    }
    
    for i in range(num_layers):
        config["blocks"].append(copy.deepcopy(tmp))
    
    return config


def translate_to_asvd(my_config, num_layers):
    asvd_config = {}
    asvd_config["lm_head"] = 1
    
    for idx in range(num_layers, -1, -1):
        block = my_config["blocks"][idx]
        asvd_config[f"model.layers.{idx}.self_attn.q_proj"] = block["q_proj"]
        asvd_config[f"model.layers.{idx}.self_attn.k_proj"] = block["k_proj"]
        asvd_config[f"model.layers.{idx}.self_attn.v_proj"] = block["v_proj"]
        asvd_config[f"model.layers.{idx}.self_attn.o_proj"] = block["o_proj"]
        asvd_config[f"model.layers.{idx}.mlp.gate_proj"] = block["gate_proj"]
        asvd_config[f"model.layers.{idx}.mlp.up_proj"] = block["up_proj"]
        asvd_config[f"model.layers.{idx}.mlp.down_proj"] = block["down_proj"]
    return asvd_config

def translate_rank_to_my_rank(rank_config, num_layers):
    config = init_config(num_layers)
    config['value'] = "rank"
    
    for name, rank in rank_config.items():
        if name == "lm_head":
            continue
        splits = name.split(".")
        proj = splits[4]
        block = int(splits[2])
        # print(name, rank)
        
        if proj == "q_proj":
            config["blocks"][block]["q_proj"] = rank
        elif proj == "k_proj":
            config["blocks"][block]["k_proj"] = rank
        elif proj == "v_proj":
            config["blocks"][block]["v_proj"] = rank
        elif proj == "o_proj":
            config["blocks"][block]["o_proj"] = rank
        elif proj == "gate_proj":
            config["blocks"][block]["gate_proj"] = rank
        elif proj == "up_proj":
            config["blocks"][block]["up_proj"] = rank
        elif proj == "down_proj":
            config["blocks"][block]["down_proj"] = rank
    return config