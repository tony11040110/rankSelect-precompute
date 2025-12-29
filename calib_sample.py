import argparse
import torch
import json
import os
import click
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from transformers.models.opt.configuration_opt import OPTConfig
from evaluate import evaluate_model, evaluate_perplexity
from datautils import get_calib_data

def main():
    model_id = "meta-llama/Llama-2-7b-hf"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="flash_attention_2",
    )
    
    # Get the 256 samples from the calibration dataset
    asvd_n_sample = 256
    calib_loader = get_calib_data("wikitext2", tokenizer, model_id, 256)
    
    llama2_ppl = 5.47
    min_ppl = 6.061531066894531
    min_threshold = abs(min_ppl - llama2_ppl)
    
    # Generate random n samplese from the 256 samples
    n = 32
    seed = 3
    torch.manual_seed(seed)
    for i in range(100):
        sample_set_id = sorted(list(torch.randint(0, 256, (32,)).numpy()))
        input_ids = torch.cat([calib_loader[i]["input_ids"] for i in sample_set_id], 0)
        
        ppl = evaluate_perplexity(model, input_ids)
        diff = abs(ppl-llama2_ppl)
        # save = "X"
        if diff < min_threshold:
            min_ppl = ppl
            best_set = {
                "sample_set_id": sample_set_id,
                "sample_set": [calib_loader[i] for i in sample_set_id],
                "ppl": ppl
            }
            with open(f"output/best_{n}_in_{asvd_n_sample}.txt", "a+") as f:
                f.write(f"ppl={ppl}, sample_set_id={sample_set_id}\n")
                torch.save(best_set, f"cache/calib_sample/sets_{n}_in_{asvd_n_sample}_{seed}_{ppl:.4f}.pt")
                save = "O"
            min_threshold = diff
            
        print(f"{i}, ppl={ppl:.4f}, diff={diff:.4f}, save={save}")
            
            
        


if __name__ == "__main__":
    main()