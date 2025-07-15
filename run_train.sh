export PYTHONPATH=$PYTHONPATH:/xxxx/HMoRA

python train.py \
    --model /xxxx/models/Qwen-Qwen2-1_5B \
    --max_steps 100000 \
    --num_epochs 2 \
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
    --batch_size 8 \
    --accumulation_steps 2 \
    --lr 5e-5 \
    --label_smoothing 0.1 \
    --warmup_steps 500 \
    --schedule_name constant_with_warmup \
    --bf16 \ # use bfloat16 mixed precision training
    # --save_dir ./checkpoints