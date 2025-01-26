import json
import os
import re
import types
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import PreTrainedModel

from .config import (
    HMoRAConfig,
    TARGET_MODULE_TYPE
)


class HMoRaModel:
    @staticmethod
    def from_pretrained(
            model: PreTrainedModel,
            name_or_path: Optional[str] = None,
    ) -> PeftModel:
        with open(name_or_path + "config.json") as f:
            config = json.load(f)
        config = HMoRAConfig.from_config(config)
        config.torch_dtype = model.dtype
        model = _apply_hmora(model, config=config)
        return model


class TaskRouter(nn.Module):
    def __init__(self, config: HMoRAConfig):
        super().__init__()
        self.torch_dtype = config.torch_dtype
        self.num_experts: int = config.num_experts
        self.dropout_prop = config.dropout
        if config.num_router_mlp_layers == 1:
            self.mlp = nn.Sequential(
                nn.Dropout(self.dropout_prop),
                nn.Linear(config.hidden_size, self.num_experts, dtype=config.torch_dtype)
            )
        else:
            self.mlp = nn.Sequential(nn.Dropout(self.dropout_prop),
                                     nn.Linear(config.hidden_size, config.router_hidden_dim,
                                               dtype=config.torch_dtype),
                                     nn.ReLU())
            for i in range(config.num_router_mlp_layers - 2):
                self.mlp.append(nn.Dropout(self.dropout_prop))
                self.mlp.append(nn.Linear(config.router_hidden_dim, config.router_hidden_dim,
                                          dtype=config.torch_dtype))
                self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(self.dropout_prop))
            self.mlp.append(nn.Linear(config.router_hidden_dim, self.num_experts,
                                      dtype=config.torch_dtype))
        self.gamma_div_balance = config.gamma_div_balance_s
        self.gamma_div_certain = config.gamma_div_certain_s

        self.task_weight: Optional[torch.Tensor] = None

    def forward(self, task_presentation: torch.Tensor):
        self.task_weight = F.softmax(self.mlp(task_presentation), dim=-1)

    def divergence_loss(self):
        # generalized Jensen-Shannon (GJS) divergence loss
        task_weight_batched = self.task_weight
        max_entropy = torch.log(torch.tensor(
            self.num_experts, dtype=task_weight_batched.dtype, device=task_weight_batched.device))
        max_entropy_m = self.gamma_div_balance * max_entropy
        min_entropy_p = self.gamma_div_certain * max_entropy
        max_div = max_entropy_m - min_entropy_p
        m = torch.mean(task_weight_batched, dim=0)
        hm = -torch.sum(m * torch.log(m + 1e-9), dim=-1)
        hm = torch.clamp(hm, max=max_entropy_m)
        hp = -torch.sum(task_weight_batched * torch.log(task_weight_batched + 1e-9), dim=-1)
        hp = torch.clamp(hp, min=min_entropy_p)
        hp = torch.mean(hp, dim=-1)
        loss = torch.relu(max_div - (hm - hp)) / max_entropy
        return loss

    def get_task_weight(self):
        return self.task_weight

    def clear(self):
        self.task_weight = None


class TokenRouter(nn.Module):
    def __init__(self, config: HMoRAConfig, input_dim: int,
                 layer_id: int, tag: Optional[str] = None):
        super().__init__()
        # alpha
        alpha = -config.epsilon_alpha + 2 * config.epsilon_alpha * (
                layer_id / config.max_llm_layer) + config.alpha_shift
        alpha = torch.tensor(alpha, dtype=config.torch_dtype)
        if config.use_task_router:
            # use task router
            if config.task_router_only:
                self.task_router = TaskRouter(config)
                self.task_router_only = True
                self.alpha = None
            else:
                self.task_router_only = False
                if torch.sigmoid(alpha) < config.alpha_low_bound:
                    # token router only
                    self.task_router = None
                    self.alpha = None
                elif torch.sigmoid(alpha) > config.alpha_up_bound:
                    # task router only
                    self.task_router = TaskRouter(config)
                    self.alpha = None
                    self.task_router_only = True
                else:
                    # combine token and task router
                    self.alpha = nn.Parameter(alpha)
                    self.task_router = TaskRouter(config)
        else:
            # token router only
            self.task_router = None
            self.task_router_only = False
            self.alpha = None

        self.num_experts = config.num_experts
        self.layer_id = layer_id
        self.input_dim = input_dim
        self.torch_dtype = config.torch_dtype
        self.tag = tag

        self.dropout_prop = config.dropout
        # routing strategy
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.top_k = config.top_k
        # routing function
        if not self.task_router_only:
            # token router parameters
            if config.num_router_mlp_layers == 1:
                self.mlp = nn.Sequential(
                    nn.Dropout(self.dropout_prop),
                    nn.Linear(input_dim, self.num_experts,
                              dtype=config.torch_dtype)
                )
            else:
                # mlp input
                self.mlp = nn.Sequential(nn.Dropout(self.dropout_prop),
                                         nn.Linear(input_dim, config.router_hidden_dim,
                                                   dtype=config.torch_dtype),
                                         nn.ReLU())
                # mlp hidden
                for i in range(config.num_router_mlp_layers - 2):
                    self.mlp.append(nn.Dropout(self.dropout_prop))
                    self.mlp.append(nn.Linear(config.router_hidden_dim, config.router_hidden_dim,
                                              dtype=config.torch_dtype))
                    self.mlp.append(nn.ReLU())
                # mlp output
                self.mlp.append(nn.Dropout(self.dropout_prop))
                self.mlp.append(nn.Linear(config.router_hidden_dim, self.num_experts,
                                          dtype=config.torch_dtype))
        else:
            self.mlp = None
        # gamma
        self.routing_weight: Optional[torch.Tensor] = None
        self.token_routing_weight: Optional[torch.Tensor] = None
        self.gamma_div_balance = config.gamma_div_balance_t
        self.gamma_div_certain = config.gamma_div_certain_t

    def forward(self, hidden_states: torch.Tensor):
        if self.task_router_only:
            routing_weight = self.task_router.get_task_weight().unsqueeze(-2)
            routing_weight = routing_weight.expand(hidden_states.shape[:-1] + (self.num_experts,))
            self.routing_weight = routing_weight
        else:
            routing_weight = F.softmax(self.mlp(hidden_states), dim=-1)
            self.token_routing_weight = routing_weight
            if self.task_router is not None:
                task_weight = self.task_router.get_task_weight().unsqueeze(-2)
                alpha = torch.sigmoid(self.alpha)
                self.routing_weight = (1 - alpha) * routing_weight + alpha * task_weight
            else:
                self.routing_weight = routing_weight
        if self.top_k_routing_strategy:
            top_k_values, top_k_indices = torch.topk(self.routing_weight, self.top_k, dim=-1)
            routing_weight = torch.full_like(self.routing_weight, torch.finfo(self.routing_weight.dtype).min)
            routing_weight.scatter_(-1, top_k_indices, top_k_values)
            routing_weight = torch.softmax(routing_weight, dim=-1)
            self.routing_weight = routing_weight
        return self.routing_weight

    def get_routing_weight(self):
        return self.routing_weight

    def divergence_loss(self, attention_mask):
        if self.task_router_only:
            return torch.tensor(0, dtype=self.routing_weight.dtype, device=self.routing_weight.device)
        # generalized Jensen-Shannon (GJS) divergence loss
        token_routing_weight = self.token_routing_weight
        mask = attention_mask.to(token_routing_weight.dtype).unsqueeze(-1)
        token_routing_weight = token_routing_weight * mask
        max_entropy = torch.log(torch.tensor(
            self.num_experts, dtype=token_routing_weight.dtype, device=token_routing_weight.device))
        max_entropy_m = self.gamma_div_balance * max_entropy
        min_entropy_p = self.gamma_div_certain * max_entropy
        max_div = max_entropy_m - min_entropy_p
        num_token = torch.sum(mask)
        token_routing_weight = token_routing_weight * mask
        m = torch.sum(token_routing_weight.view(-1, self.num_experts), dim=0) / num_token
        entropy_m = -torch.sum(m * torch.log(m + 1e-9), dim=-1)
        entropy_m = torch.clamp(entropy_m, max=max_entropy_m)
        entropy_p = -torch.sum(token_routing_weight * torch.log(token_routing_weight + 1e-9), dim=-1)
        entropy_p = torch.clamp(entropy_p, min=min_entropy_p) * mask.squeeze(-1)
        entropy_p = torch.sum(entropy_p) / num_token
        loss = torch.relu(max_div - (entropy_m - entropy_p)) / max_entropy
        return loss

    def load_balancing_loss(self, attention_mask):
        routing_weight = self.token_routing_weight
        mask = attention_mask.to(routing_weight.dtype)
        num_token = mask.sum()
        routing_weight = routing_weight * mask.unsqueeze(-1)
        count = torch.sign(self.routing_weight * mask.unsqueeze(-1))
        freq = torch.sum(count.view(-1, self.num_experts), dim=0) / (num_token * self.top_k)
        prop = torch.sum(routing_weight.view(-1, self.num_experts), dim=0) / num_token
        loss = torch.sum(prop * freq) * self.num_experts
        return loss.unsqueeze(0)

    def clear(self):
        if self.task_router is not None:
            self.task_router.clear()
        self.routing_weight = None
        self.token_routing_weight = None


class RouterManager(nn.Module):
    def __init__(self, config: HMoRAConfig,
                 task_routers: nn.ModuleList,
                 token_routers: nn.ModuleList):
        super().__init__()
        self.task_routers = task_routers
        self.token_routers = token_routers
        # loss
        self.top_k_routing_strategy = config.top_k_routing_strategy
        self.use_load_balancing_loss = config.use_load_balancing_loss
        self.use_div_loss = config.use_div_loss

        self.lambda_auxiliary = config.lambda_auxiliary
        self.lambda_lm = config.lambda_lm

    def set_task_weight(self, task_embedding: torch.Tensor):
        for task_router in self.task_routers:
            task_router(task_embedding)

    def clear(self):
        for router in self.token_routers:
            router.clear()

    def backward_auxiliary_loss_for_seq_router(self, reduce='sum'):
        if self.use_load_balancing_loss:
            return 0

        auxiliary_loss = []
        for router in self.task_routers:
            if self.use_div_loss:
                auxiliary_loss.append(router.divergence_loss())
            else:
                break

        if len(auxiliary_loss) == 0:
            return 0

        loss = torch.stack(auxiliary_loss, dim=0)
        if reduce == 'sum':
            loss = torch.sum(loss)
        elif reduce == 'mean':
            loss = torch.mean(loss)
        else:
            raise ValueError(f'reduce must be sum or mean, but got {reduce}')
        loss = loss * self.lambda_auxiliary
        loss.backward()
        return loss.item()

    def get_auxiliary_loss(self, loss, attention_mask, reduce='sum'):
        auxiliary_loss = []
        for router in self.token_routers:
            if self.use_load_balancing_loss:
                auxiliary_loss.append(router.load_balancing_loss(attention_mask))
            elif self.use_div_loss:
                auxiliary_loss.append(router.divergence_loss(attention_mask))
            else:
                break
        if len(auxiliary_loss) == 0:
            return loss
        auxiliary_loss = torch.stack(auxiliary_loss, dim=0)
        if reduce == 'sum':
            auxiliary_loss = torch.sum(auxiliary_loss)
        elif reduce == 'mean':
            auxiliary_loss = torch.mean(auxiliary_loss)
        else:
            raise ValueError(f'reduce must be sum or mean, but got {reduce}')
        loss = self.lambda_lm * loss + self.lambda_auxiliary * auxiliary_loss
        return loss


class MoRa(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: HMoRAConfig):
        super().__init__()
        self.out_features, self.in_features = base_layer.weight.shape
        self.dtype_ = config.torch_dtype

        self.num_experts = config.num_experts
        self.rank = config.lora_r

        self.dropout_tate = config.dropout
        self.dropout = nn.Dropout(p=config.dropout)
        self.use_hydra_lora = config.use_hydra_lora
        if config.use_hydra_lora:
            self.mora_a = nn.Parameter(
                torch.empty((self.rank, self.in_features), dtype=self.dtype_))
        else:
            self.mora_a = nn.Parameter(
                torch.empty((self.rank * self.num_experts, self.in_features), dtype=self.dtype_))
        self.mora_b = nn.Parameter(
            torch.empty((self.out_features, self.rank * self.num_experts), dtype=self.dtype_))
        # rs_lora scaling
        self.scaling = config.lora_alpha / math.sqrt(config.lora_r)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mora_a, a=math.sqrt(5))
        nn.init.zeros_(self.mora_b)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.linear(hidden_states, self.mora_a)
        target_shape = hidden_states.shape[:-1] + (self.num_experts, self.rank)
        if self.use_hydra_lora:
            hidden_states = hidden_states.unsqueeze(-2).expand(target_shape)
        else:
            hidden_states = hidden_states.view(target_shape)
        hidden_states = (hidden_states * gate.unsqueeze(-1)).view(hidden_states.shape[:-2] + (-1,))
        hidden_states = F.linear(hidden_states, self.mora_b) * self.scaling
        return hidden_states.to(residual.dtype) + residual


class LoRA(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: HMoRAConfig):
        super().__init__()
        self.out_features, self.in_features = base_layer.weight.shape
        self.dtype_ = config.torch_dtype
        self.dropout_tate = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        # rs_lora scaling
        self.scaling = config.lora_alpha / math.sqrt(config.lora_r)
        self.rank = config.lora_r
        self.lora_a = nn.Parameter(
            torch.empty((self.rank, self.in_features), dtype=self.dtype_))
        self.lora_b = nn.Parameter(
            torch.empty((self.out_features, self.rank), dtype=self.dtype_))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.linear(hidden_states, self.lora_a)
        hidden_states = F.linear(hidden_states, self.lora_b) * self.scaling
        return hidden_states + residual


class AdapterLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear,
                 config: HMoRAConfig,
                 router: Optional[TokenRouter],
                 use_cache: bool = False,
                 use_lora: bool = False):
        super().__init__()
        # linear
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.weight = base_layer.weight
        if hasattr(base_layer, "bias"):
            self.bias = base_layer.bias
        else:
            self.register_parameter('bias', None)

        self.use_lora = use_lora
        # mixture of lora experts
        if self.use_lora:
            self.lora = LoRA(base_layer, config)
        else:
            self.mora = MoRa(base_layer, config)
        # routers
        self.use_cache = use_cache # for router sharing
        self.router = router

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_lora:
            result = F.linear(hidden_states, self.weight, self.bias)
            return self.lora(hidden_states, result)
        else:
            if self.use_cache:
                # sharing router, the gate values has been calculated
                gate = self.router.get_routing_weight()
            else:
                # calculate gate values
                gate = self.router(hidden_states)
            result = F.linear(hidden_states, self.weight, self.bias)
            return self.mora(hidden_states, gate, result)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        # for left padding position encoding of task encoder
        return x + self.pe[:, -x.size(1):, :]


class TaskEncoder(nn.Module):
    def __init__(self, config: HMoRAConfig, task_embedding: Optional[torch.Tensor]):
        super(TaskEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_encoder_layer = config.num_encoder_layer
        self.pos_encoder = PositionalEncoding(self.hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(self.hidden_size, nhead=16,
                                                    dim_feedforward=self.hidden_size * 2, dropout=config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_encoder_layer)
        self.task_embedding = torch.nn.Embedding(1, config.hidden_size, dtype=config.torch_dtype)
        if task_embedding is not None:
            self.task_embedding.weight.data.copy_(task_embedding)

    def forward(self, src, src_key_padding_mask):
        # src shape: (batch_size, seq_len, hidden_size)
        # src_key_padding_mask shape: (batch_size, seq_len)

        task_embedding = self.task_embedding(torch.tensor([0], device=src.device))
        task_embedding = task_embedding.expand(src.shape[0], -1, -1)
        src = self.pos_encoder(src)
        src = torch.cat([src, task_embedding], dim=1)
        src = src.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        src_key_padding_mask = torch.cat([src_key_padding_mask, torch.ones(src_key_padding_mask.shape[0], 1,
                                                                           device=src_key_padding_mask.device,
                                                                           dtype=src_key_padding_mask.dtype)], dim=1)
        # attention_mask = self.gen_causal_attention_mask(hidden_states, padding_mask)
        src_key_padding_mask = src_key_padding_mask == 0
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        sentence_embedding = output[-1]
        # take the output of the last token as task presentation
        return sentence_embedding


def get_peft_model(model: PreTrainedModel, config: HMoRAConfig) -> PeftModel:
    config.hidden_size = model.config.hidden_size
    config.model_type = model.config.model_type
    if model.config.model_type == "bloom":
        config.share_router_for_qkv = True
    model = _apply_hmora(model, config)
    return model


def _get_module(model: nn.Module, target_name: str):
    for name, module in model.named_modules():
        if name == target_name:
            return module
    return None


def _apply_for_layer(layer_module: nn.Module,
                     layer_id: int,
                     config: HMoRAConfig):
    token_router_list = []
    task_router_list = []

    def get_router(input_dim, tag: Optional[str] = None):
        router = TokenRouter(config, input_dim, layer_id, tag=tag)
        token_router_list.append(router)
        if router.task_router is not None:
            task_router_list.append(router.task_router)
        return router

    def get_target_modules(target: list[str]) -> list[str]:
        res = []
        for t in target:
            res += TARGET_MODULE_TYPE[config.model_type][t]
        return res

    def set_mora(module, targets: list):
        """
        applay hmora for a list of linear, share the same router
        :param module:
        :param targets:
        :return:
        """
        token_router = None
        use_cache = False
        for target_name in targets:
            if target_name not in config.target_modules:
                continue
            target_model = _get_module(module, target_name)
            if not isinstance(target_model, nn.Linear):
                continue
            if config.target_modules_lora is not None and target_name in config.target_modules_lora:
                target_model = AdapterLinear(target_model, config, router=None, use_cache=False, use_lora=True)
            else:
                if token_router is None:
                    token_router = get_router(target_model.in_features, tag=target_name)
                target_model = AdapterLinear(target_model, config, router=token_router, use_cache=use_cache,
                                             use_lora=False)
                use_cache = True  # besides the first linear, others use the cached routing weight
            setattr(module, target_name, target_model)

    # apply for attention block
    atte_name = TARGET_MODULE_TYPE[config.model_type]['atte']
    atte_module = _get_module(layer_module, atte_name)
    if config.share_router_for_qkv:
        target_modules = get_target_modules(['q', 'k', 'v'])
        set_mora(atte_module, target_modules)
        target_modules = get_target_modules(['o'])
        for target_module in target_modules:
            set_mora(atte_module, [target_module])
    else:
        target_modules = get_target_modules(['q', 'k', 'v', 'o'])
        for target_module in target_modules:
            set_mora(atte_module, [target_module])

    # apply for ffn block
    ffn_name = TARGET_MODULE_TYPE[config.model_type]['ffn']
    ffn_module = _get_module(layer_module, ffn_name)
    if config.share_router_for_w_i:
        target_modules = get_target_modules(['wi'])
        set_mora(ffn_module, target_modules)
        target_modules = get_target_modules(['wo'])
        for target_module in target_modules:
            set_mora(ffn_module, [target_module])
    else:
        target_modules = get_target_modules(['wi', 'wo'])
        for target_module in target_modules:
            set_mora(ffn_module, [target_module])

    return token_router_list, task_router_list


def _apply_hmora(model, config: HMoRAConfig) -> PeftModel:
    """
    inject hmora into base model
    :param model: pretrain model
    :param config: hmora config
    :return: peft model
    """

    def _extract_layer_id(name: str):
        """
        extract layer id from module name
        :param name:
        :return:
        """
        # modules in module list end with digit
        match = re.search(r'\.\d+$', name)
        if match:
            return int(name.split('.')[-1])
        return None

    layer_list = dict()  # {layer_id : layer_module}
    for module_name, module in model.named_modules():
        layer_id = _extract_layer_id(module_name)
        if layer_id is not None:
            if layer_id > config.max_llm_layer:
                # record the max layer id
                config.max_llm_layer = layer_id
            # record decoder layers
            layer_list[layer_id] = module

    token_router_list = nn.ModuleList()
    task_router_list = nn.ModuleList()
    # apply for each layer
    for layer_id in sorted(layer_list.keys()):
        module = layer_list[layer_id]
        token_routers, task_routers = _apply_for_layer(module, layer_id, config)
        for router in token_routers:
            token_router_list.append(router)
        for router in task_routers:
            task_router_list.append(router)

    if config.use_task_router and len(task_router_list) > 0:
        embed = _get_module(model, model.base_model_prefix + '.' + TARGET_MODULE_TYPE[config.model_type]['embed'])
        task_embedding = embed.weight.data[config.task_token_id]
        task_encoder = TaskEncoder(config, task_embedding)
    else:
        task_encoder = None
    model.task_encoder = task_encoder
    # router manager
    router_manager = RouterManager(config, task_router_list, token_router_list)
    model.router_manager = router_manager

    trainable_modules = ['router', 'mora', 'lora', 'task_encoder']
    # freeze parameters
    for param_name, param in model.named_parameters():
        if any(target in param_name for target in trainable_modules):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # overwrite save_pretrained
    model.save_pretrained = types.MethodType(_save_pretrained, model)
    # model.peft_config = config
    setattr(model, 'peft_config', config)
    return model


def _save_pretrained(self: nn.Module, path):
    if not os.path.exists(path):
        os.makedirs(path)
    trainable_params = dict()
    for name, param in self.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param.detach().cpu()
    config = self.peft_config.export()
    torch.save(trainable_params, path + '/' + 'adapter_model.safetensors')
    config['torch_dtype'] = None
    json.dump(config, open(path + '/' + 'config.json', 'w'))
