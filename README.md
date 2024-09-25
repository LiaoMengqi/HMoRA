# HMoRA
# Introduction

HMoRA is a fine-tuning method for LLMs that combines LoRA and MoE. The base models currently supported include:
- BLOOM
- LLaMA
- Qwen2

Here, we provide a sample HMoRA training script and a sample training data (each sample consists of a source text and a target text). The training script provided is written for Qwen2. Fine-tuning other base LLMs requires adjusting the input and output format according to the specific tokens of the LLM. We also provide code in `example_test.ipynb` that demonstrates how to load the saved HMoRA weights and perform testing.

# Environment

The main Python libraries and versions we use are as follows:
```
pytorch 2.0.1
transformers 4.44.2
datasets 2.18.0
```

# Quick Start

Fine-tuning script: `example_train.py`
train 
```shell
python example_train.py \
--model Qwen/Qwen2-1.5B \
--dataset ./dataset/example_data.parquet \
--max_steps 10000 \
--num_epochs 1 \
--target_modules_lora o_proj down_proj \
--lora_r 8 \
--lora_alpha 8 \
--top_k_routing_strategy \
--use_task_router \
--share_router_for_qkv \
--share_router_for_w_i \
--epsilon_alpha 2.0 \
--alpha_shift 0.0 \
--use_div_loss \
--gamma_div_certain_t 0.5 \
--gamma_div_balance_t 0.98 \
--gamma_div_certain_s 0.5 \
--gamma_div_balance_s 0.98 \
--lambda_auxiliary 0.005 \
--eta_b 1.2 \
--num_experts 8 \
--use_hydra_lora \
--dropout 0.1 \
--batch_size 1 \
--accumulation_steps 12 \
--lr 5e-5 \
--label_smoothing 0.1 \
--warmup_steps 500 \
--schedule_name constant_with_warmup \
```