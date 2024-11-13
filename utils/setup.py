import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from hmora import HMoRAConfig, get_peft_model

DEFAULT_TARGET_MODULES = {
    'llama': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    'bloom': ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
    'qwen2': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
}


def setup(args, tokenizer):
    # base model
    model_config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.padding_side == 'right':
        tokenizer.padding_side = 'left'

    if args.target_modules is None:
        args.target_modules = DEFAULT_TARGET_MODULES[model_config.model_type]

    # number of parameters of base model
    vanilla_params = sum(p.numel() for p in model.parameters())

    peft_config = HMoRAConfig(
        target_modules=args.target_modules,
        target_modules_lora=args.target_modules_lora,
        dropout=args.dropout,
        # routing strategy
        top_k_routing_strategy=args.top_k_routing_strategy,
        top_k=args.top_k,
        # router sharing
        use_task_router=args.use_task_router,
        task_router_only=args.task_router_only,
        share_router_for_qkv=args.share_router_for_qkv,
        share_router_for_w_i=args.share_router_for_w_i,
        # router
        num_router_mlp_layers=args.num_router_mlp_layers,
        router_hidden_dim=args.router_hidden_dim,
        epsilon_alpha=args.epsilon_alpha,
        alpha_shift=args.alpha_shift,
        alpha_up_bound=args.alpha_up_bound,
        alpha_low_bound=args.alpha_low_bound,
        # loss
        use_load_balancing_loss=args.use_load_balancing_loss,
        use_div_loss=args.use_div_loss,
        gamma_div_certain_t=args.gamma_div_certain_t,
        gamma_div_balance_t=args.gamma_div_balance_t,
        gamma_div_certain_s=args.gamma_div_certain_s,
        gamma_div_balance_s=args.gamma_div_balance_s,
        lambda_lm=args.lambda_lm,
        lambda_auxiliary=args.lambda_auxiliary,
        # experts
        num_experts=args.num_experts,
        use_hydra_lora=args.use_hydra_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    peft_config.torch_dtype = torch.float32
    peft_config.padding_side = tokenizer.padding_side
    if peft_config.task_token is not None:
        peft_config.task_token_id = tokenizer.convert_tokens_to_ids(peft_config.task_token)
    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('model parameters :', vanilla_params,
          ' | trainable parameters :', trainable_params,
          ' | rate :', trainable_params / vanilla_params)

    return model
