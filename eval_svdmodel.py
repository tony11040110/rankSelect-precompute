import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from datasets import load_dataset
import time
import itertools
from torch.utils.data.dataset import Dataset
import click

@torch.no_grad()
def eval_ppl(model, tokenizer, seqlen=2048, batch_size=1):
    print(model)
    # Transpose the ALinear and BLinear weight in ASVDLinear in the model
    # if args.bits == 16:
    #     from svd_llama.modeling_asvd_llama import ASVDLinear
    #     for name, module in model.named_modules():
    #         if isinstance(module, ASVDLinear):
    #             print("Transpose ALinear and BLinear weight:", name)
    #             module.ALinear.weight = torch.nn.Parameter(module.ALinear.weight.T)
    #             module.BLinear.weight = torch.nn.Parameter(module.BLinear.weight.T)
    
    # Set the model sequence length
    model.seqlen = seqlen
    results = []
    # datasets = ["wikitext2", "c4"]
    ################### Eval wikitext2 ppl #################
    
    # Load dataset
    model_id = model.config._name_or_path.lower()
    # cache_testloader = (
    #     f"/tmp/wikitext2_testloader_{model_id.replace('/', '_')}_all.cache"
    # )
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    # torch.save(testenc, cache_testloader)
    
    # Get input IDs
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    nlls = []

    # Loop through each batch
    print(f"Eval wikitext2: {nsamples} samples")
    for i in tqdm(range(0,nsamples,batch_size), desc=f"(Eval) wikitext2"):
        # if i % 50 == 0:
        #     print(f"sample {i}")

        # Calculate end index
        j = min(i + batch_size, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(model.device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
        # loss_lst.append(loss.float())
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        
        # Empty CUDA cache to save memory
        torch.cuda.empty_cache()
    
    # for dataset, ppl_test, neg_loss in results:
    #     print(f"{dataset} perplexity {ppl_test}, loss (neg): {neg_loss}")
    return ppl.item(), float(torch.stack(nlls).sum() / (nsamples * model.seqlen))

@torch.no_grad()
def lm_eval_zero_shot(model, tokenizer, tasks, batch_size=1, peft=None, parallelize=False, report_to_wandb=False):
    """args

    Args:
        model : Huggingface model
        tasks: List of tasks to evaluate. E.g., ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada"],
        batch_size (int, optional): _description_. Defaults to 1.
        parallelize (bool, optional): Duplicate the model to multiple device. Defaults to False.
        report_to_wandb (bool, optional): Report result to wandb. Defaults to False.

    Returns:
        Dict: evalauation results
    """
    import lm_eval # 0.4.x
    from lm_eval import utils as lm_eval_utils
    from lm_eval.api.registry import ALL_TASKS
    from lm_eval.models.huggingface import HFLM
    import wandb
    import json
    import pprint

    
    # if distribute_model:
    #     utils.distribute_model(model)
    # else:
    #     model.to(utils.DEV)
    
    
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, peft=peft, parallelize=parallelize)
    
    results = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size=batch_size)['results']

    # for task, result in results.items():
    #     print(f"{task}: {result}")
    
    
    
    # if report_to_wandb:
    #     wandb.log(metric_vals)
    return results

@torch.no_grad()
def lm_eval_mmlu(model, tokenizer, batch_size=1, peft=None, parallelize=False, report_to_wandb=False):
    """args

    Args:
        model : Huggingface model
        tasks: List of tasks to evaluate. E.g., ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada"],
        batch_size (int, optional): _description_. Defaults to 1.
        parallelize (bool, optional): Duplicate the model to multiple device. Defaults to False.
        report_to_wandb (bool, optional): Report result to wandb. Defaults to False.

    Returns:
        Dict: evaluation results
    """
    import lm_eval # 0.4.x
    from lm_eval import utils as lm_eval_utils
    from lm_eval.api.registry import ALL_TASKS
    from lm_eval.models.huggingface import HFLM
    import wandb
    import json
    import pprint

    # if distribute_model:
    #     utils.distribute_model(model)
    # else:
    #     model.to(utils.DEV)
    
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, peft=peft, parallelize=parallelize)
    
    results = lm_eval.simple_evaluate(hflm, tasks="mmlu", batch_size=batch_size, num_fewshot=5)['results']

    # for task, result in results.items():
    #     print(f"{task}: {result}")
    
    ## Report results
    
    # print("================ ACC ================")
    # print("MMLU, acc: ", round(results["mmlu"]["acc,none"], 4))
    
    # if report_to_wandb:
    #     wandb.log(metric_vals)
    return results

@torch.no_grad()
def eval_decoding(model, tokenizer, original_len=4, generated_len=128, batch_size=1, device="cuda"):
    # Adopt from SVD-LLM
    # test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    
    # Transpose the ALinear and BLinear weight in ASVDLinear in the model
    if args.bits == 16:
        from svd_llama.modeling_asvd_llama import ASVDLinear
        for name, module in model.named_modules():
            if isinstance(module, ASVDLinear):
                print("Transpose ALinear and BLinear weight:", name)
                module.ALinear.weight = torch.nn.Parameter(module.ALinear.weight.T)
                module.BLinear.weight = torch.nn.Parameter(module.BLinear.weight.T)
    
    ## For batching the dataset
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    
    model.eval()
    latency = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    
    test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_dataset = process_data(test_data, tokenizer, original_len, 'text')
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print(type(test_loader), test_loader.dataset.tensors.shape) [166, 2048]
    
    ## Fetch the first 10 batches, and calculate the average throughput
    weight_memory = torch.cuda.memory_allocated()
    count = 0
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time_ns()
        try:
            generation_output = model.generate(
                    input_ids=batch,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    use_cache=True,     # use kv_cache
                    top_k=50,
                    max_new_tokens=generated_len, # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
                    top_p=0.95,
                    temperature=1,
            )
            torch.cuda.synchronize()
        except RuntimeError as e:
            click.secho(f"Warning: {e}", fg="red")
            exit()
        end_time = time.time_ns()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        # Get the number of generated tokens
        num_generated_tokens = generation_output.shape[1] - batch.shape[1]  # Subtract input length from output length
        if num_generated_tokens < generated_len:
            click.secho(f"Warning: early stop", fg="red")
            continue
        if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
            latency = end_time - start_time
            token_num = batch.shape[0] * generated_len
            count += 1
            print("count: {}, time: {}, num_generated_tokens: {}".format(count, end_time - start_time, num_generated_tokens))
        else:
            click.secho(f"Warning: generation_output contains nan", fg="red")
            # print("time: {}".format(end_time - start_time))
        if count == 2:
            break
    
    print("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
    print("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
    print("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
    print("Throughput: {} tokens/sec".format(token_num / (latency / (10 ** 9))))
    
    results = {
        "Total memory (GB)": end_memory/(1024 ** 3),
        "Weight memory (GB)": weight_memory/(1024 ** 3),
        "Activation memory (GB)": (end_memory - start_memory)/(1024 ** 3),
        "Throughput (tok/sec)": token_num / (latency / (10 ** 9)),
    }
    return results

@torch.no_grad()
def eval_ttft(model, tokenizer, original_len=2048, batch_size=1, device="cuda"):
    
    # Transpose the ALinear and BLinear weight in ASVDLinear in the model
    if args.bits == 16:
        from svd_llama.modeling_asvd_llama import ASVDLinear
        for name, module in model.named_modules():
            if isinstance(module, ASVDLinear):
                print("Transpose ALinear and BLinear weight:", name)
                module.ALinear.weight = torch.nn.Parameter(module.ALinear.weight.T)
                module.BLinear.weight = torch.nn.Parameter(module.BLinear.weight.T)
    
    ## For batching the dataset
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    
    model.eval()
    latency = 0
    num_batches_to_fetch = 10
    
    test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_dataset = process_data(test_data, tokenizer, original_len, 'text')
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print(type(test_loader), test_loader.dataset.tensors.shape) [166, 2048]
    
    ## Fetch the first 10 batches, and calculate the average throughput
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        torch.cuda.synchronize()
        start_time = time.time_ns()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,     # use kv_cache
                max_new_tokens=1, # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        )
        torch.cuda.synchronize()
        latency += time.time_ns() - start_time
    
    results = {
        "Prefill length": original_len,
        "TTFT (sec)": (latency / (10 ** 9)) / num_batches_to_fetch
    }

    return results

def main(args):
    import click
    import pprint
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import svd_llama
    # import succinct_llama
    from peft import PeftConfig, PeftModel
    import torch
    import json
    
    base_model = args.model_name
    
    if args.peft:
        click.secho(f"Eval PEFT model: {args.peft}", fg="yellow")
        config = PeftConfig.from_pretrained(args.model_name)
        base_model = config.base_model_name_or_path
        click.secho(f" -> Loading fixed pretrained: {config.base_model_name_or_path}", fg="yellow")
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cpu")
        
        if args.fake_quant:
            import rotation.hadamard_utils as hadamard_utils
            import rotation.rotation_utils as rotation_utils
            import quant_utils
            ## Quantization model init
            rotation_utils.fuse_layer_norms(model)
            quant_utils.add_actquant(model) #Add Activation Wrapper to the model
            
            qlayers = quant_utils.find_qlayers(model)
            for name in qlayers:
                # Find 'down_proj' or 'down_proj.BLinear'
                if 'down_proj' in name and 'ALinear' not in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K     # Register the buffer
                    qlayers[name].K = K
                    # qlayers[name].fp32_had = args.fp32_had
                if 'o_proj' in name and 'ALinear' not in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K     # Register the buffer
                    qlayers[name].K = K
                    qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
            
            # Load quantized model paramenters
            click.secho(f" -> Loading fake quant model: {args.fake_quant}", fg="yellow")
            save_dict = torch.load(args.fake_quant)
            model.load_state_dict(save_dict["model"])
        
        click.secho(f" -> Loading PEFT model: {args.model_name}", fg="yellow")
        model = PeftModel.from_pretrained(model, args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        click.secho(f"Eval model: {args.model_name}", fg="yellow")
        model = AutoModelForCausalLM.from_pretrained(base_model, 
                                                     torch_dtype=torch.float16, 
                                                     device_map="cpu", 
                                                    #  attn_implementation="flash_attention_2"
                                                     )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    
        if args.fake_quant:
            import rotation.hadamard_utils as hadamard_utils
            import rotation.rotation_utils as rotation_utils
            import quant_utils
        
            ## Quantization model init
            rotation_utils.fuse_layer_norms(model)
            quant_utils.add_actquant(model) # Add Activation Wrapper to the model
            
            qlayers = quant_utils.find_qlayers(model)
            for name in qlayers:
                # Find 'down_proj' or 'down_proj.BLinear'
                if 'down_proj' in name and 'ALinear' not in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K     # Register the buffer
                    qlayers[name].K = K
                    # qlayers[name].fp32_had = args.fp32_had
                if 'o_proj' in name and 'ALinear' not in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K     # Register the buffer
                    qlayers[name].K = K
                    qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
            
            # Load quantized model paramenters
            click.secho(f" -> Loading fake quant model: {args.fake_quant}", fg="yellow")
            save_dict = torch.load(args.fake_quant)
            model.load_state_dict(save_dict["model"])
    
    model.eval()
    model.to("cuda")
    torch.cuda.empty_cache()
    
    ##################################################### Evaluation #####################################################
    start = time.time()
   
    log = {}
    decode_results = {}
    ttft_results = {}
    ppl_results = {}
    zero_results = {}
    mmlu_results = {}
    
    ## Efficiency evaluation (Need to be run the first)
    if args.eval_decoding:
        click.secho(f"Generate token length for decoding evaluation: {args.generate_len}", fg="yellow")
        click.secho(f" -> Start decoding evaluation", fg="yellow")
        decode_results = eval_decoding(model, tokenizer, generated_len=args.generate_len, batch_size=args.speedup_bs, device="cuda")
    
    if args.eval_ttft:
        click.secho(f" -> Start prefill evaluation", fg="yellow")
        ttft_results = eval_ttft(model, tokenizer, original_len=args.prompt_len, batch_size=args.speedup_bs, device="cuda")
    
    ## PPL evaluation
    if args.ppl:
        ppl, neg_loss = eval_ppl(model, tokenizer, seqlen=2048, batch_size=1)
        ppl_results["wikitext2"] = {"ppl": ppl}
    
    ## Zero-shot evaluation
    # tasks = ["boolq", "winogrande", "hellaswag", "arc_easy", "arc_challenge", "openbookqa", "piqa"]
    if args.zero_shot:
        zero_results = lm_eval_zero_shot(
            model=model,
            tokenizer=tokenizer,
            tasks=args.tasks, 
            batch_size=args.batch_size, 
            peft=args.peft, 
            parallelize=args.parallelize, 
            report_to_wandb=args.report_to_wandb)
        
        model_tag = args.model_name.lstrip("./").replace("/", "_").replace(".pt", "")
        report_dir = "lmeval_report/peft" if args.peft else "lmeval_report"
        os.makedirs(report_dir, exist_ok=True)

        report = os.path.join(report_dir, f"{model_tag}_zero_shot_fake_quant.json")
        # report = f"lmeval_report/{'peft' if args.peft else ''}/{args.model_name.replace('/', '_').replace('.pt', '')}_zero_shot_fake_quant.json"
        json.dump(zero_results, open(report, "w"), indent=2)
    
    ## MMLU 5-shot
    if args.mmlu:
        ## MMLU 5-shot
        mmlu_results = lm_eval_mmlu(
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size, 
            peft=args.peft, 
            parallelize=args.parallelize, 
            report_to_wandb=args.report_to_wandb
        )
        
        model_tag = args.model_name.lstrip("./").replace("/", "_").replace(".pt", "")
        report_dir = "lmeval_report/peft" if args.peft else "lmeval_report"
        os.makedirs(report_dir, exist_ok=True)

        report = os.path.join(report_dir, f"{model_tag}_mmlu_fake_quant.json")
        # report = f"lmeval_report/{'peft' if args.peft else ''}/{args.model_name.replace('/', '_').replace('.pt', '')}_mmlu_fake_quant.json"
        json.dump(mmlu_results, open(report, "w"), indent=2)
    
    end = time.time()
    click.secho(f"Evaluation time: {(end - start)//60} min", fg="yellow")
    
    ## Report results
    if decode_results:
        print("================ Decoding ================")
        pprint.pprint(decode_results)
        with open(f"decoding.txt", "a+") as f:
            f.write(f"{args.model_name}, {args.speedup_bs}, {args.generate_len}, {decode_results['Throughput (tok/sec)']:.2f}, \
                {decode_results['Activation memory (GB)']:.2f}, {decode_results['Weight memory (GB)']:.2f}, {decode_results['Total memory (GB)']:.2f}\n")
    
    if ttft_results:
        print("================ Prefill ================")
        pprint.pprint(ttft_results)
        with open(f"ttft.txt", "a+") as f:
            f.write(f"{args.model_name}, {args.speedup_bs}, {ttft_results['Prefill length']}, {ttft_results['TTFT (sec)']:.4f}\n")
    
    if ppl_results:
        print("================ PPL ================")
        pprint.pprint(ppl_results)
        log["PPL"] = ppl_results
    
    if zero_results:
        metric_vals = {task: round(zero_results.get('acc,none', result['acc,none']), 4) for task, result in zero_results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        
        print("================ Zero-shot ACC ================")
        pprint.pprint(metric_vals)
        
        metric_vals_norm = {}
        for task, result in zero_results.items():
            if 'acc_norm,none' in result:
                metric_vals_norm[task] = round(result['acc_norm,none'], 4)
        print("================ Zero-shot ACC_NORM ================")
        pprint.pprint(metric_vals_norm)
        log["Zero-shot"] = metric_vals
        log["Zero-shot norm"] = metric_vals_norm

    if mmlu_results:
        print("================ MMLU ACC ===============")
        print("MMLU, acc: ", round(mmlu_results["mmlu"]["acc,none"], 4))
        log["MMLU, acc"] = round(mmlu_results["mmlu"]["acc,none"], 4)
        
    if args.log_to_file:
        if not os.path.exists("eval_results"):
            os.makedirs("eval_results")
        with open(f"eval_results/{args.model_name.replace('/', '_').replace('.pt', '')}_eval_results.txt", "a") as f:
            f.write(json.dumps(log, indent=2))
    

if __name__ == "__main__":
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--peft", action='store_true', help="Load peft model")
    parser.add_argument("--parallelize", action='store_true', help="Parallelize")
    parser.add_argument("--zero-shot", action='store_true', help="Zero-shot evaluation")
    parser.add_argument("--tasks", type=str, nargs='+', default=["boolq", "winogrande", "hellaswag", "arc_easy", "arc_challenge", "openbookqa", "piqa"], help="Tasks to evaluate")
    parser.add_argument("--mmlu", action='store_true', help="MMLU evaluation")
    parser.add_argument("--report_to_wandb", action='store_true', help="Report to wandb")
    parser.add_argument("--fake-quant", type=str, help="fake quant model")
    parser.add_argument("--ppl", action='store_true', help="PPL evaluation")
    parser.add_argument("--bits", type=int, default=16, choices=[4, 16], help="Quantization bits")
    # Speedup evaluation
    parser.add_argument("--eval_decoding", action='store_true', help="Decoding efficiency evaluation")
    parser.add_argument("--generate_len", type=int, default=128, help="Generated token length for efficiency evaluation")
    parser.add_argument("--eval_ttft", action='store_true', help="TTFT evaluation")
    parser.add_argument("--prompt_len", type=int, default=2048, help="Prompt length")
    parser.add_argument("--speedup_bs", type=int, default=1, help="Batch size for speedup evaluation")
    parser.add_argument("--log_to_file", action='store_true', help="Log task evaluation result to file")
    
    args = parser.parse_args()
    
    main(args)
