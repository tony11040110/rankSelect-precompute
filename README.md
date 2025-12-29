# LLM Rank Selection (LLM-RS)

LLM Rank Selection (LLM-RS) is a fine-grained rank selection methodology for SVD-based compression method on LLM.


## Support SVD-based compression method

**Compression method**
- ASVD (Yuan _et al._, 2023)
- SVD-LLM (Only whiten) (Wang, _et al._, 2024)

**Rank selection method**
- STRS (Yuan _et al._, 2023)
- greedy (ours)

**Note**: Now only support LLaMA family.

## Setup

Requirements:

- Python 3.10
- PyTorch 2.2
- CUDA 12.1
- [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.0 and above (recommend installing from source)

Make sure the requirements are all met, and then
```
pip install -r requirements.txt
```

environment.yml was created.

## Basic Usage

### 1. Compression

Compress the target LLM model.

```bash
CUDA_VISIBLE_DEVICES="0" python llm_rs.py \
--model_id="meta-llama/Llama-2-7b-hf" \
--method whiten \
--calib_dataset wikitext2 \
--step_type rank \
--rank_step 128 \
--search_method greedy \
--param_ratio_target 0.8 \
--use_cache \
--config_root "./config/" \
--record_file "./output/llm_rs_llama2_7b" \
--dump_huggingface_model \
--dump_config \
--save_folder "./svd_model" \
--svdlog_path ./output/svd_spectrum.jsonl \
--multilevel
# --search_with_succinct \
```

Options:
- `--no_search`: only the generate sensitivity list
- `--only_search`: only generate the rank search config to `./config_dump.json`
- `--step_type`: the spacing of sampling the sensitivity of layers. Support two types: `param_ratio` and `rank`
- `--ratio_step` / `--rank_step`: followed by `--step_type`, specify the amount of space

### 2. Evaluation

Evaluate the compressed model.

**Task performance**

- `ppl`: Perplexity (wikitext2)
- `zero-shot`: Common sense reasoning datasets (0-shot)
- `MMLU`: MMLU (5-shot)

Task evaluation are supported by [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness) framework (version >= 0.4.0, install from source).

- Single compressed model

```bash
CUDA_VISIBLE_DEVICES=0 python eval_svdmodel.py \
    --model_name <compressed model> \
    --ppl \
    --mmlu \
    --zero-shot \
```

- Evaluate the compressed model with fine-tuned LoRA adaptor

```bash
CUDA_VISIBLE_DEVICES=0 python eval_svdmodel.py \
    --peft \
    --model_name <lora adaptor> \
    --ppl \
    --mmlu \
    --zero-shot \
```

- Evaluate the fakequant model

```bash
CUDA_VISIBLE_DEVICES=0 python eval_svdmodel.py \
    --model_name <compressed model> \
    --fake-quant <fakequant_checkpoint>.pt \
    --mmlu \
    --ppl \
    --zero-shot \
```

**Inference speed**

- Time To First Token (TTFT) (sec)

```bash
CUDA_VISIBLE_DEVICES=0 python eval_svdmodel.py \
    --model_name <compressed model> \
    --eval_ttft \
    --speedup_bs 32 \
    --prompt_len 32
```

- Throughput (tok/sec)

```bash
CUDA_VISIBLE_DEVICES=0 python eval_svdmodel.py \
    --model_name <compressed model> \
    --eval_decoding \
    --speedup_bs 16 \
    --generate_len 128
```
