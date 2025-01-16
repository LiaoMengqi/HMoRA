# HMoRA
# Introduction

HMoRA is a fine-tuning method for LLMs that combines LoRA and MoE.
The base models currently supported include:
- BLOOM
- LLaMA, LLaMA 2, LLaMA 3, LLaMA 3.1
- Qwen2

Here, we provide an example HMoRA training script `example_train.py` and a mini training data 
(each example in training data consists of a source text and a target text). 
The training script provided is written for Qwen2. 
Fine-tuning other base LLMs requires adjusting the input and output format 
according to the specific tokens of the LLM. 
We also provide code in `generate.ipynb` that demonstrates how 
to load the saved HMoRA checkpoint and perform generation.

# Environment

The main Python libraries and versions we use are as follows:
```
pytorch 2.0.1
transformers 4.44.2
datasets 2.18.0
```

# Quick Start

Example command for fine-tuning script: `example_train.py`

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

description of parameters:
- `model`: the model to be fine-tuned.
- `dataset`: the path to the training data.
- `max_steps`: the maximum number of training steps.
- `num_epochs`: the number of epochs. If `max_steps` is not specified, `num_epochs` will be used as the maximum number of epochs.`
- `target_modules_lora`: the target modules to be fine-tuned with LoRA only.
- `lora_r`: the rank of LoRA.
- `lora_alpha`: the alpha of LoRA.
- `top_k_routing_strategy`: whether to use the top-k routing strategy.
- `use_task_router`: whether to use the task router.
- `share_router_for_qkv`: whether to share the router for W_Q,W_K and W_V.
- `share_router_for_w_i`: whether to share the router for W_up and W_gate.
- `epsilon_alpha`: hyperparameter for the hybrid routing strategy.
- `alpha_shift`: hyperparameter for the hybrid routing strategy.
- `use_div_loss`: whether to use the GJS divergence loss.
- `gamma_div_certain_t`: hyperparameter for the GJS divergence loss for token router.
- `gamma_div_balance_t`: hyperparameter for the GJS divergence loss for token router.
- `gamma_div_certain_s`: hyperparameter for the GJS divergence loss for task router.
- `gamma_div_balance_s`: hyperparameter for the GJS divergence loss for task router.
- `lambda_auxiliary`: hyperparameter for the auxiliary loss.
- `eta_b`: speedup for lora_b update.
- `num_experts`: number of experts.
- `use_hydra_lora`: whether to use the Hydra LoRA.
- `dropout`: dropout rate.
- `batch_size`: batch size.
- `accumulation_steps`: gradient accumulation steps.
- `lr`: learning rate.
- `label_smoothing`: label smoothing.
- `warmup_steps`: warmup steps.
- `schedule_name`: learning rate schedule name.