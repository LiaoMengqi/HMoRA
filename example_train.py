import datasets
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import transformers

from utils.func import set_seed, set_device, save_check_point
from utils.setup import setup
import torch
import re
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import get_scheduler
from peft import PeftModel

import numpy as np
from hmora import SUPPORTED_SEQ2SEQ_MODELS, SUPPORTED_CAUSAL_MODELS, TARGET_MODULE_TYPE

transformers.logging.set_verbosity_error()


def _extract_layer_id(name: str):
    """
    extract layer id from module name
    :param name:
    :return:
    """
    names = name.split('.')

    for item in names:
        match = re.search(r'\d+', item)
        if match:
            return int(item)
    return None


def clip_gradients(model, args):
    if args.max_grad_norm is None:
        return
    parameters_to_clip = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(parameters_to_clip, max_norm=args.max_grad_norm)


def create_scheduler(optimizer, args):
    if args.schedule_name == 'polynomial':
        specific_kwargs = {'power', args.polynomial_power}
    elif args.schedule_name == 'cosine_with_restarts':
        specific_kwargs = {'num_cycles', args.num_cycles}
    else:
        specific_kwargs = None

    scheduler = get_scheduler(
        args.schedule_name,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        scheduler_specific_kwargs=specific_kwargs
    )
    return scheduler


def format_source(source):
    prefix = '<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>human\n'
    source = prefix + source + "\n<|im_end|>\n"
    return source


def format_target(target):
    target = '<|im_start|>assistant\n' + target + '<|endoftext|>'
    return target


def train(model: PeftModel,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          tokenizer: PreTrainedTokenizer,
          train_dataloader: DataLoader,
          device: Optional[str],
          args):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)
    loss_list = []
    bar = tqdm(total=args.max_steps, ncols=150)

    def take_step(batched_source):

        # div loss for task router
        if model.task_encoder is not None and model.router_manager.use_div_loss:
            prefix_tensors = tokenizer(batched_source, padding=True, truncation=True,
                                       return_tensors='pt', max_length=1000, add_special_tokens=False).to(device)
            embedding = getattr(model.base_model, TARGET_MODULE_TYPE[model.config.model_type]['embed'])
            hidden_states = embedding(prefix_tensors.input_ids)
            task_embed = model.task_encoder(hidden_states, prefix_tensors.attention_mask)
            model.router_manager.set_task_weight(task_embed)
            loss = model.router_manager.backward_auxiliary_loss_for_seq_router()
            loss_list[-1] = loss_list[-1] + loss
        clip_gradients(model, args)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        bar.update(1)
        bar.set_postfix(loss=loss_list[-1])

    def loop():
        step = 0
        accumulated_step = 0
        accumulated_loss = 0
        finished = False
        batched_source = []

        for epoch in range(args.num_epochs):
            for batch in train_dataloader:
                model.train()
                if model.task_encoder is not None:
                    batched_source = batched_source + batch['source']
                    prefix_tensors = tokenizer(batch['source'], padding=True, truncation=True,
                                               return_tensors='pt', max_length=1024, add_special_tokens=False).to(
                        device)
                    embedding = getattr(model.base_model, TARGET_MODULE_TYPE[model.config.model_type]['embed'])
                    hidden_states = embedding(prefix_tensors.input_ids)
                    task_embed = model.task_encoder(hidden_states, prefix_tensors.attention_mask)
                    model.router_manager.set_task_weight(task_embed)

                input_tensors = tokenizer([format_source(i) for i in batch['source']],
                                          [format_target(i) for i in batch['target']],
                                          padding=True,
                                          truncation=True, return_token_type_ids=True,
                                          return_tensors='pt', max_length=1024
                                          , add_special_tokens=False).to(device)

                labels = input_tensors.input_ids.clone()
                # ignore padding and input
                labels[input_tensors.token_type_ids == 0] = -100
                labels = labels[:, 1:]

                output = model(input_ids=input_tensors.input_ids,
                               attention_mask=input_tensors.attention_mask,
                               use_cache=False)

                prop = output.logits[:, :-1]
                loss = loss_fct(prop.reshape(-1, output.logits.shape[-1]), labels.reshape(-1))
                loss = model.router_manager.get_auxiliary_loss(loss, input_tensors.attention_mask)
                loss.backward()
                model.router_manager.clear()
                # step
                if args.accumulation_steps is not None and args.accumulation_steps > 1:
                    accumulated_step += 1
                    accumulated_loss += (loss / args.accumulation_steps).item()
                    if accumulated_step == args.accumulation_steps:
                        loss_list.append(accumulated_loss)
                        accumulated_loss = 0
                        accumulated_step = 0
                        take_step(batched_source)
                        step = step + 1
                        batched_source = []
                else:
                    loss_list.append(loss.item())
                    take_step(batched_source)
                    step = step + 1
                    batched_source = []
                if args.max_steps is not None and step > args.max_steps:
                    finished = True
                    break
                del prop
                del loss
            if finished:
                break

    loop()
    save_check_point(model, args, tokenizer)
    bar.close()
    return


def main(args):
    device = set_device(args.gpu, args.exclude_gpu)
    set_seed(args.seed)

    # train data
    train_dataset = datasets.load_dataset('parquet', split='train', data_files=args.dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # model
    model = setup(args, tokenizer).to(device)
    # optimizer
    speedup_param_name = ['lora_b', 'mora_b']
    norm_param = []
    speedup_param_param = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(spd_name in name for spd_name in speedup_param_name):
            speedup_param_param.append(param)
        else:
            norm_param.append(param)
    # apply higher learning rate for LoRA B
    optimizer = AdamW([{'params': norm_param, 'lr': args.lr},
                       {'params': speedup_param_param, 'lr': args.eta_b * args.lr}],
                      weight_decay=args.weight_decay)
    # learning rate scheduler
    scheduler = create_scheduler(optimizer, args)

    train(model, optimizer, scheduler, tokenizer,
          train_dataloader, device, args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training')
    # model
    parser.add_argument('--model', type=str, required=True)
    # peft
    parser.add_argument('--target_modules', nargs='+', type=str, default=None)
    parser.add_argument('--target_modules_lora', nargs='+', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    # lora
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=8)
    # hmora
    # hmora routing strategy
    parser.add_argument('--top_k_routing_strategy', action='store_true', default=False)
    parser.add_argument('--top_k', type=int, default=2)
    # hmora router sharing
    parser.add_argument('--use_task_router', action='store_true', default=False)
    parser.add_argument('--task_router_only', action='store_true', default=False)
    parser.add_argument('--share_router_for_qkv', action='store_true', default=False)
    parser.add_argument('--share_router_for_w_i', action='store_true', default=False)
    # hmora router config
    parser.add_argument('--num_router_mlp_layers', type=int, default=1)
    parser.add_argument('--router_hidden_dim', type=int, default=32)
    parser.add_argument('--epsilon_alpha', type=float, default=2.0)
    parser.add_argument('--alpha_shift', type=float, default=0.0)
    parser.add_argument('--alpha_up_bound', type=float, default=0.8)
    parser.add_argument('--alpha_low_bound', type=float, default=0.2)
    # hmora loss
    parser.add_argument('--use_load_balancing_loss', action='store_true', default=False)
    parser.add_argument('--use_div_loss', action='store_true', default=False)
    parser.add_argument('--gamma_div_certain_t', type=float, default=0.5)
    parser.add_argument('--gamma_div_balance_t', type=float, default=1)
    parser.add_argument('--gamma_div_certain_s', type=float, default=0.5)
    parser.add_argument('--gamma_div_balance_s', type=float, default=1)
    parser.add_argument('--lambda_auxiliary', type=float, default=0.01)
    parser.add_argument('--lambda_lm', type=float, default=1.0)

    # hmora experts
    parser.add_argument('--eta_b', type=float, default=1.0)
    parser.add_argument('--num_experts', type=int, default=8)
    parser.add_argument('--use_hydra_lora', action='store_true', default=False)

    # dataset
    parser.add_argument('--dataset', type=str, required=True)
    # training
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=-2)
    parser.add_argument('--exclude_gpu', nargs='+', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    # important training config
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    # schedule
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--schedule_name', type=str, default='constant', choices=['constant',
                                                                                  'linear',
                                                                                  'cosine',
                                                                                  'cosine_with_restarts',
                                                                                  'polynomial',
                                                                                  'constant',
                                                                                  'constant_with_warmup'])
    parser.add_argument('--polynomial_power', type=int, default=2)
    parser.add_argument('--num_cycles', type=int, default=5)

    args_parsed = parser.parse_args()
    main(args_parsed)
