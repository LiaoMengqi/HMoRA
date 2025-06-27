import datasets
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoConfig)
import transformers
from peft import PeftModel
import torch

from hmora import SUPPORTED_CAUSAL_MODELS, SUPPORTED_SEQ2SEQ_MODELS, HMoRaModel
from utils.evaluator import evaluate
from utils.func import set_seed, set_device, to_json

transformers.logging.set_verbosity_error()


def main(args):
    device = set_device(args.gpu, args.exclude_gpu)
    set_seed(args.seed)

    model_config = AutoConfig.from_pretrained(args.model)
    # model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if args.peft_model is not None:
        if args.peft_model in ['lora']:
            model = PeftModel.from_pretrained(model, args.peft_model_path)
        elif args.peft_model == 'hmora':
            model = HMoRaModel.from_pretrained(model, args.peft_model_path)
            peft_weights = torch.load(args.peft_model_path + '/' + 'adapter_model.safetensors')
            model.load_state_dict(peft_weights, strict=False)
    model.to(device)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if model_config.model_type in SUPPORTED_CAUSAL_MODELS:
        if tokenizer.padding_side == 'right':
            print('change padding side to left.')
            tokenizer.padding_side = 'left'

    # data
    if args.valid_data_type == 'mmlu':
        from utils.data_process import format_mmlu
        valid_dataset = datasets.load_dataset('parquet', split=args.split, data_files=args.dataset)
        valid_dataset = valid_dataset.map(format_mmlu)
    elif args.valid_data_type == 'mmlu_pro':
        from utils.data_process import format_mmlu_pro
        valid_dataset = datasets.load_dataset('parquet', split=args.split, data_files=args.dataset)
        valid_dataset = valid_dataset.map(format_mmlu_pro)
    elif args.valid_data_type == 'arc_e' or args.valid_data_type == 'arc_c':
        from utils.data_process import format_arc
        valid_dataset = datasets.load_dataset('parquet', split=args.split, data_files=args.dataset)
        valid_dataset = valid_dataset.map(format_arc)
    else:
        raise ValueError(f'valid_data_type {args.valid_data_type} not supported!')
    from utils.data_process import remove_column
    valid_dataset = remove_column(valid_dataset)

    # few shot data
    if args.num_few_shot > 0:
        few_shot = valid_dataset.select(range(args.num_few_shot))
        valid_dataset = valid_dataset.select(range(args.num_few_shot, len(valid_dataset)))
    else:
        few_shot = None

    dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=True)
    res = evaluate(model, dataloader, tokenizer, args, data_type=args.valid_data_type,
                   device=device, display=True)
    print('acc : ', res[0])
    path = './evaluate_result/' + model.config.model_type + '/'
    if args.peft_model is not None:
        path = path + args.peft_model + '/'
    else:
        path = path + 'base/'
    path = path + args.valid_data_type + '/'
    if args.save_dir is not None:
        path = path + args.save_dir + '/'
    else:
        path = path + 'default/'
    to_json(res, path, 'eval')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training')
    # model
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--peft_model', type=str, default=None)
    parser.add_argument('--peft_model_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    # dataset
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--valid_data_type', type=str, default=None)
    # evaluation
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_few_shot', type=int, default=5)

    parser.add_argument('--gpu', type=int, default=-2)
    parser.add_argument('--exclude_gpu', nargs='+', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args_parsed = parser.parse_args()
    main(args_parsed)
