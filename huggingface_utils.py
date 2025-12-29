from modules.svd_linear import SVDLinear
import os

def dump_to_huggingface_repos(model, tokenizer, save_path, succinct=False):
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    config = model.config.to_dict()
    config["truncation_ranks"] = {}
    config["succinct_splits"] = {}
    for name, module in model.named_modules():
        if isinstance(module, SVDLinear):
            config["truncation_ranks"][name] = module.truncation_rank
            config["succinct_splits"][name] = module.succinct_split
    if succinct:
        config["architectures"] = ["SuccinctLlamaForCausalLM"]
        config["model_type"] = "succinctllama"
    else:
        config["architectures"] = ["ASVDLlamaForCausalLM"]
        config["model_type"] = "svdllama"
    import json

    json.dump(config, open(save_path + "/config.json", "w"), indent=2)
    print("Done building huggingface model")
    