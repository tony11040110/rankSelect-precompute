import argparse
import torch
import torch.nn as nn
import json
import os
import click
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, AutoConfig
from transformers.models.opt.configuration_opt import OPTConfig
from evaluate_utils import evaluate_model, evaluate_perplexity, my_eval_ppl
from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, compress_model_asvd, compress_model_svd
from sensitivity import (
    get_calib_sensitivity_ratio,
    get_calib_sensitivity_step_rank,
    get_calib_sensitivity_step_rank_compressed_baseline,
    truncate_sensitivity_list,
)

# from quantization import rtn_quant_sequential
from binary_search import binary_search_truncation_rank
from greedy import greedy_search_truncation_rank
from spectrum_greedy import spectrum_greedy_search_truncation_rank
from config_translate import translate_rank_to_my_rank
from huggingface_utils import dump_to_huggingface_repos
from transformers import LlamaForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights
from utils.calc import set_uniform_truncation_rank
from pathlib import Path
from whiten_utils import insert_whiten_scale_matrix, compress_model_whiten, attach_nan_hooks_to_factorized_layers
    

def main(args):
    
    # --------------------------------------------------- Load model ---------------------------------------------------
    # model_id = args.model_id
    # tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # config = AutoConfig.from_pretrained(model_id)
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config)
    # # Do not split decoder layer
    # device_map = infer_auto_device_map(model, no_split_module_classes=["LlamaDecoderLayer"]) 
    # print(device_map)
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, device_map=device_map, torch_dtype=torch.float16, trust_remote_code=True
    # )
    # model.seqlen = 2048
    # model.eval()
    
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    import torch

    try:
        from utils.svd_logger import SVDLogger
        os.environ["SVDLOG_PATH"] = args.svdlog_path
        SVDLogger(args.svdlog_path).run_header({
            "model_id": args.model_id,
            "method": args.method,
            "step_type": args.step_type,
            "rank_step": args.rank_step,
            "search_method": args.search_method,
            "param_ratio_target": args.param_ratio_target,
            "calib_dataset": args.calib_dataset,
            "cmd": " ".join(os.sys.argv),
        })
    except Exception as e:
        print("[SVDLOG] header failed:", e)

    model_id = args.model_id

    # 1) Tokenizer（補上 pad_token，避免某些 dataset/eval 場景出錯）
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 生成/評估時較穩定

    # 2) 直接載模型：關閉自動切分/懶載入，避免 meta tensor / CPU offload
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=None,           # <== 關閉自動切分
        low_cpu_mem_usage=False,   # <== 關閉懶載入，拿到實體權重
        trust_remote_code=True,
    )

    # 3) 若有 Accelerate 裝置映射/Hook，先移除，避免 to() 被忽略
    try:
        from accelerate.hooks import remove_hook_from_module, remove_hook_from_submodules
        remove_hook_from_submodules(model)
        remove_hook_from_module(model)
    except Exception:
        pass

    if hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")  # 避免殘留映射

    # 4) 強制整個模型上 GPU
    model = model.to("cuda:0")
    model.eval()
    torch.set_grad_enabled(False)

    # 5) 設定序列長度（預設程式寫 2048；24GB 可能吃緊，可先改小）
    model.seqlen = getattr(args, "seqlen", 2048)  # 如果沒 seqlen 參數，先用 1024

    
    # --------------------------------------------------- Calibration data ---------------------------------------------------

    # sensitivity calibration
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, nsamples=256, seqlen=model.seqlen, seed=3)
    if args.method == "asvd":
        calib_input_distribution(model, calib_loader, args.scaling_method, args.use_cache)
    elif args.method == "whiten":
        insert_whiten_scale_matrix(model=model, calib_loader=calib_loader, calib_dataset=args.calib_dataset, dev=args.device)
    elif args.method == "svd":
        pass
    else:
        raise ValueError("Invalid method")

    # --------------------------------------------------- Sensitivity list ---------------------------------------------------
    # spectrum search is training-free; skip sensitivity list creation.
    if args.search_method == "spectrum":
        sensitivity, base_ppl = None, None
    else:
        # Optional warmup: if a baseline rank config is provided, do a local 7-point sweep
        # around that warmup rank for *every* layer, instead of a full grid.
        if (
            args.step_type == "rank"
            and getattr(args, "baseline_config", None) is not None
        ):
            try:
                with open(args.baseline_config, "r") as f:
                    raw_cfg = json.load(f)
                warmup_rank_config = {
                    k: int(v) for k, v in raw_cfg.items() if isinstance(v, (int, float))
                }
                # sensitivity.py reads these from args (no extra CLI needed)
                args.warmup_rank_dict = warmup_rank_config
                args.local_points = 7
                click.secho(
                    f"[Warmup->Sensitivity] Use local 7-point sweep around baseline_config={args.baseline_config}",
                    fg="yellow",
                )
            except Exception as e:
                click.secho(f"[Warmup->Sensitivity] baseline_config load failed: {e}", fg="red")

       
        if args.step_type == "rank":
            if (
                getattr(args, "baseline_config", None) is not None
                and str(getattr(args, "sens_baseline_mode", "dense")).lower() == "compressed"
            ):
                sensitivity, base_ppl = get_calib_sensitivity_step_rank_compressed_baseline(
                    model, calib_loader, args, args.use_cache, rankstep=args.rank_step
                )
            else:
                sensitivity, base_ppl = get_calib_sensitivity_step_rank(
                    model, calib_loader, args, args.use_cache, rankstep=args.rank_step
                )
        elif args.step_type == "param_ratio":
            sensitivity, base_ppl = get_calib_sensitivity_ratio(
                model, calib_loader, args, args.use_cache, step=args.ratio_step
            )

    if args.no_search:
        return
    
    #------------------------------------------------------ Searching --------------------------------------------------
    
    # Collect linear layers
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
    
    if (args.search_method != "spectrum"):
        sensitivity = truncate_sensitivity_list(module_dict, sensitivity, args.rank_step)
    
    # search best truncation rank for each layer
    click.secho(f"Search method: {args.search_method}", fg="yellow")
    
        
    if args.search_method == "greedy":
            ## Remove the high rank part of the sensitivity list
        select_result, _ = greedy_search_truncation_rank(module_dict=module_dict,
                                                            raw_sensitivity_dict=sensitivity, 
                                                            base_ppl=base_ppl,
                                                            do_succinct_calib=False, 
                                                            param_ratio_target=args.param_ratio_target,
                                                            step_type=args.step_type,
                                                            args=args)
    elif args.search_method == "spectrum":
        # A1: activation-aware spectrum greedy (weighted by diag input covariance)
        select_result = spectrum_greedy_search_truncation_rank(
            linear_info=linear_info,
            param_ratio_target=args.param_ratio_target,
            rank_step=int(args.rank_step),
            score_mode="energy_per_param",
            min_rank=max(1, int(args.rank_step)),
            #n_calib_samples=int(args.n_calib_samples),
            #act_aware=True,
            device=str(args.device),
            verbose=True,
        )
    elif args.search_method == "uniform":
        select_result = set_uniform_truncation_rank(module_dict, linear_info, args.param_ratio_target)
    else:
        raise ValueError("Invalid search method")
    
    if args.only_search:
        json.dump(select_result, open(f"config_dump.json", "w"))
        click.secho(f"Search result is saved to config_dump.json", fg="yellow")
        return
    
    # Compress model
    start_time = time.time()
    if args.method == "asvd":
        compress_model_asvd(model, select_result, args)
    elif args.method == "whiten":
        #TODO: compress by rank
        compress_model_whiten(model, select_result, args)
    elif args.method == "svd":
        compress_model_svd(model, select_result, args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    click.secho(f"Elapsed time for compression: {elapsed_time/60} minutes", fg="yellow")

    # Log succinct error
    # error_file = f"{args.method}_w2_{args.search_method}_{args.step_type}_{args.rank_step}_{int(args.param_ratio_target*100)}"
    # catch_succinct_error(model, save_file=error_file)
    # exit()
    
    attach_nan_hooks_to_factorized_layers(model)

    model.half()
    torch.cuda.empty_cache()
    model.cuda()
    
    # evaluate
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        eval_ppl="wikitext2",
        limit=-1,
    )
    test_ppl = result["wikitext2"]
    click.secho(f"Wikitext2 Test PPL: {test_ppl}", fg="yellow")
    
    # Log the result
    rec = Path(args.record_file)

    # 確保資料夾存在
    rec.parent.mkdir(parents=True, exist_ok=True)
    with open(f"output/{args.record_file}", "a+") as f:
    #with open(rec, "a+") as f:
       f.write(f"param_ratio={args.param_ratio_target}, method={args.method}, search={args.search_method}, succinct_calib={args.search_with_succinct}, ppl={test_ppl}\n")
       click.secho(f"Result is saved to output/{args.record_file}", fg="yellow")
    
    
    
    rank_config_for_translate = select_result

    # config = translate_to_rank(asvd_config=select_result)
    config = translate_rank_to_my_rank(
        rank_config=rank_config_for_translate,
        num_layers=len(model.model.layers),
    )
    # Example: ASVD_w2_STRS_param_ratio_0.05_80_8.86

    if args.search_with_succinct:
        search_succinct_str = "_succinct"
    else:
        search_succinct_str = ""

    

    if args.calib_dataset == "wikitext2":
        calib_dataset_str = "_w2"
    elif args.calib_dataset == "c4":
        calib_dataset_str = "_c4"
    else:
        calib_dataset_str = ""

    if args.step_type == "rank":
        config_name = (
            f"{args.method}{calib_dataset_str}_{args.search_method}_"
            f"{args.step_type}_{args.rank_step}"
            f"{search_succinct_str}_"
            f"{int(args.param_ratio_target*100)}_{round(test_ppl, 2):.2f}"
        )
    elif args.step_type == "param_ratio":
        config_name = (
            f"{args.method}{calib_dataset_str}_{args.search_method}_"
            f"{args.step_type}_{args.ratio_step}"
            f"{search_succinct_str}_"
            f"{int(args.param_ratio_target*100)}_{round(test_ppl, 2):.2f}"
        )

    if args.dump_config:
        if not os.path.exists(args.config_root):
            os.makedirs(args.config_root)
            click.secho(f"Create folder {args.config_root}", fg="green")

        # 原本的一維 rank config（給舊工具用，只有一個 rank）
        filename = f"{args.config_root}/{config_name}.json"
        with open(filename, "w") as f:
            json.dump(config, f)
        click.secho(f"Config is saved to {filename}", fg="yellow")
        with open("config_dump.json", "w") as f:
            json.dump(config, f)
        click.secho("Config is also saved to ./config_dump.json", fg="yellow")

        


    if args.dump_huggingface_model:
        save_folder = f"{args.save_folder}/{config_name}"
        if args.search_with_succinct:
            save_folder += "_succinct"
            dump_to_huggingface_repos(model, tokenizer, save_folder, succinct=True)
        else:
            dump_to_huggingface_repos(model, tokenizer, save_folder)
        click.secho(f"Huggingface model is saved to {save_folder}", fg="green")

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=-1,
        help="target param ratio",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="scaling method",
    )
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="ppl",
        choices=["ppl", "stable_rank"],
        help="search metric",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6"],
        help="weight quantization method",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method in SVD decomposition",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--step_type",
        type=str,
        default="param_ratio",
        choices=["param_ratio", "rank"],
    )
    parser.add_argument(
        "--ratio_step",
        type=float,
        default=0.1,
        help="step for param/rank ratio",
    )
    parser.add_argument(
        "--rank_step",
        type=int,
        default=128,
        help="step for rank",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="asvd",
        choices=["asvd", "whiten", "svd"],
        help="method",
    )
    parser.add_argument(
        "--succinct_calib",
        action="store_true",
        help="succinct calibration",
    )
    parser.add_argument(
        "--specify_sample",
        type=str,
        default="",
        help="specify samples in the ppl evaluation in sensitivity list creation.",
    )
    parser.add_argument(
        "--no_search",
        action="store_true",
        help="skip search",
    )
    parser.add_argument(
        "--search_method",
        type=str,
        default="STRS",
        choices=["STRS", "greedy", "uniform", "spectrum"],
        help="search method",
    )
    parser.add_argument(
        "--record_file",
        type=str,
        default="llm_rs_result.txt",
        help="Record search result",
    )
    parser.add_argument(
        "--dump_config",
        action="store_true",
        help="dump config",
    )
    parser.add_argument(
        "--config_root",
        type=str,
        default="./config",
        help="root directory for saving configs",
    )
    parser.add_argument(
        "--dump_huggingface_model",
        action="store_true",
        help="dump huggingface model",
    )
    parser.add_argument(
        "--search_with_succinct",
        action="store_true",
        help="[step rank] search with succinct-calibrated sensitivity list",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="./svd_models",
        help="folder for saving compressed models",
    )
    parser.add_argument(
        "--succint_split",
        type=str,
        choices=["A", "B"],
        default="A",
        help="The matrix to split in succinct SVD",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="default device",
    )
    parser.add_argument(
        "--only_search",
        action="store_true",
        help="only conduct rank searching and dump the result",
    )
    parser.add_argument(
        "--svdlog_path",
        type=str,
        default=None,
        help="Where to write SVD spectrum events (JSONL). If not set, use env SVDLOG_PATH; if none, disable logging."
    )    
    parser.add_argument(
        "--baseline_config",
        type=str,
        default=None,
        help="Path to a baseline rank config JSON (layer_name -> rank). Used for multilevel param-share experiments.",
    )
    parser.add_argument(
        "--sens_baseline_mode",
        type=str,
        default="dense",
        choices=["dense", "compressed"],
        help="For sensitivity (search_method=greedy) with --baseline_config: "
             "'dense' measures per-layer sensitivity on the dense model (default). "
             "'compressed' measures sensitivity on the baseline-compressed model (e.g. spectrum baseline).",
    )
    


    args = parser.parse_args()

    main(args)
