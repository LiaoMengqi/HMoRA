import json
import os
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.utils import is_bitsandbytes_available

from .config import HMoRAConfig


# from .model import MoRa, MoRaLinear, Router


def load_adapter_weights(
        name_or_path: str,
        adapter_name: str,
        device: str,
        dtype: torch.dtype,
):
    if not os.path.exists(name_or_path):
        name_or_path = snapshot_download(repo_id=name_or_path, repo_type="model")

    with open(
            name_or_path + os.sep + "adapter_config.json", "r", encoding="utf8"
    ) as fp:
        config = HMoRAConfig(adapter_name_=adapter_name, torch_dtype=dtype).from_config(
            json.load(fp)
        )

    weights = torch.load(
        name_or_path + os.sep + "adapter_model.bin", map_location=device
    )

    return config, weights


def inject_pretrained(
        model: PreTrainedModel,
        config: HMoRAConfig,
        device: str,
        weights: Optional[Dict[str, torch.Tensor]] = None,
):
    config.hidden_size = model.config.hidden_size
    config.model_type = model.config.model_type
    model._mixlora_config = config
    # for idx, layer in enumerate(model.layers):
    #     router = RouterLayer(config, device)
    #     hmora = HMoRaLayer(layer, router, config, device)
    #     layer.forward = hmora.forward_fn
    #     _inject_attn_module(idx, layer.self_attn, config, weights, router, device)
    #     _inject_mlp_module(idx, layer.mlp, config, weights, router, device)


def _inject_mlp_module(
        layer_idx: int,
        mlp: torch.nn.Module,
        config: HMoRAConfig,
        weights: Dict[str, torch.Tensor],
        # router: RouterLayer,
        device: str,
):
    for proj_name, inject in config.target_modules.items():
        if not inject or not hasattr(mlp, proj_name):
            continue
        linear = getattr(mlp, proj_name)
        # mora = MoRaLinear(linear, MoRa(linear, config, device), router, 'mlp')
        # if weights is not None:
        #     mora.mora_layer.reset_parameters(
        #         (weights[f"{layer_idx}_{proj_name}_down"], weights[f"{layer_idx}_{proj_name}_up"])
        #     )
        # else:
        #     mora.mora_layer.reset_parameters()
        # setattr(mlp, proj_name, mora)


def _inject_attn_module(
        layer_idx: int,
        self_attn: torch.nn.Module,
        config: HMoRAConfig,
        weights: Dict[str, torch.Tensor],
        # router: RouterLayer,
        device: str,
):
    for proj_name, inject in config.target_modules.items():
        if not inject or not hasattr(self_attn, proj_name):
            continue
        linear = getattr(self_attn, proj_name)

        # mora = MoRaLinear(linear, MoRa(linear, config, device), router, 'attn')
        # if weights is not None:
        #     mora.mora_layer.reset_parameters(
        #         (weights[f"{layer_idx}_{proj_name}_down"], weights[f"{layer_idx}_{proj_name}_up"])
        #     )
        # else:
        #     mora.mora_layer.reset_parameters()
        # setattr(self_attn, proj_name, mora)
