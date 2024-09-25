import copy
from dataclasses import dataclass, asdict
from typing import Dict, List

import torch
from transformers.activations import ACT2FN

VALID_TREND = ("EQUAL", "INCREASE", "DECREASE")

SUPPORTED_SEQ2SEQ_MODELS = ('t5',)
SUPPORTED_CAUSAL_MODELS = ('llama', 'bloom', 'qwen2')

TARGET_MODULE_TYPE = {
    'llama': {'q': ['q_proj'],
              'k': ['k_proj'],
              'v': ['v_proj'],
              'o': ['o_proj'],
              'wi': ['gate_proj', 'up_proj'],
              'wo': ['down_proj'],
              'atte': 'self_attn',
              'ffn': 'mlp',
              'embed': 'embed_tokens',
              'decoders': 'model.layers'},
    'qwen2': {'q': ['q_proj'],
              'k': ['k_proj'],
              'v': ['v_proj'],
              'o': ['o_proj'],
              'wi': ['gate_proj', 'up_proj'],
              'wo': ['down_proj'],
              'atte': 'self_attn',
              'ffn': 'mlp',
              'embed': 'embed_tokens',
              'decoders': 'model.layers'},
    'bloom': {'q': ['query_key_value'],
              'v': [],
              'k': [],
              'o': ['dense'],
              'wi': ['dense_h_to_4h','gelu_impl'],
              'wo': ['dense_4h_to_h'],
              'atte': 'self_attention',
              'ffn': 'mlp',
              'embed': 'word_embeddings',
              'decoders': 'transformer.h'}
}


@dataclass
class HMoRAConfig:
    target_modules: List[str] = None
    peft_type: str = "hmora"
    hidden_size: int = None
    model_type: str = None
    torch_dtype: torch.dtype = torch.float32
    dropout: float = 0.1
    max_llm_layer: int = 0
    # routing strategy
    top_k_routing_strategy: bool = False
    top_k: int = 2
    # task router
    use_task_router: bool = False
    task_router_only: bool = False
    # router sharing
    share_router_for_qkv: bool = False
    share_router_for_w_i: bool = False
    # routers
    num_router_mlp_layers: int = 1
    router_hidden_dim: int = 32
    epsilon_alpha: float = 2.0
    alpha_shift: float = 0.0
    alpha_low_bound: float = 0.0
    alpha_up_bound: float = 1.0
    # loss
    use_load_balancing_loss: bool = False
    use_div_loss: bool = False
    gamma_div_certain_t: float = 0.0
    gamma_div_balance_t: float = 1.0
    gamma_div_certain_s: float = 0.0
    gamma_div_balance_s: float = 1.0
    lambda_auxiliary: float = 0.01
    lambda_lm: float = 1.0
    # lora
    target_modules_lora: List[str] = None
    use_rs_scaling: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    # experts
    use_hydra_lora: bool = False
    num_experts: int = 8
    # task encoder and task embedding
    use_causal_attention = False
    trainable_encoder: bool = True
    padding_side: str = "right"
    task_token: str = '?'
    task_token_id: int = None
    num_encoder_layer: int = 1

    @staticmethod
    def from_config(config: Dict[str, any]) -> "HMoRAConfig":
        config = HMoRAConfig(**config)
        return config

    def export(self) -> Dict[str, any]:
        config = asdict(self)
        return config
