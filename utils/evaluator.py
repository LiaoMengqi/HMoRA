from transformers import PreTrainedModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

from .data_process import (
    gen_few_shot_prompt,
    format_source
)
from hmora import TARGET_MODULE_TYPE

warnings.filterwarnings("ignore")


@torch.no_grad()
def evaluate(model: PreTrainedModel,
             eval_dataloader: DataLoader,
             tokenizer,
             args,
             data_type,
             device=None,
             display=False,
             few_shot=None):
    model.eval()

    right_count_for_subject = dict()
    all_count_for_subject = dict()
    right_count = 0
    all_count = 0

    few_shot_prompt = gen_few_shot_prompt(few_shot, args.num_few_shot)
    bar = None
    """
    format the question as:
    question text 
    A. option A
    B. option B
    C. option C
    D. option D
    ...
    we take the logits of the last token of the option and the question text, and then take the argmax to get the answer.
    """
    option_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    num_option = {'mmlu': 4, 'mmlu_pro': 10, 'arc_e': 4, 'arc_c': 4, 'openbookqa': 4, 'swag': 4, 'commonsenseqa': 5}

    option_index = tokenizer(option_list[:num_option[data_type]],
                             return_tensors='pt', add_special_tokens=False).input_ids.squeeze().to(device)

    if display:
        bar = tqdm(eval_dataloader, ncols=120)
    for batch in eval_dataloader:
        model_input = tokenizer(
            [format_source(i, model.config.model_type, few_shot_prompt, training=False) for i in batch['source']],
            padding=True, return_tensors='pt', add_special_tokens=False)
        model_input.to(device)
        if args.peft_model == 'hmora' and model.task_encoder is not None:
            prefix_tensors = tokenizer(batch['source'], padding=True, return_tensors='pt', add_special_tokens=False).to(
                device)
            embedding = getattr(model.base_model, TARGET_MODULE_TYPE[model.config.model_type]['embed'])
            hidden_states = embedding(prefix_tensors.input_ids)
            task_embed = model.task_encoder(hidden_states, prefix_tensors.attention_mask)
            model.router_manager.set_task_weight(task_embed)
        res = model(input_ids=model_input.input_ids, attention_mask=model_input.attention_mask, use_cache=False)
        answer = torch.argmax(res.logits[:, -1, option_index], dim=-1)
        label = torch.LongTensor(batch['target_id'])
        if data_type == 'mmlu' or data_type == 'mmlu_pro':
            for i in range(len(answer)):
                if answer[i].item() == label[i].item():
                    if batch['subject'][i] in right_count_for_subject:
                        right_count_for_subject[batch['subject'][i]] = right_count_for_subject[batch['subject'][i]] + 1
                    else:
                        right_count_for_subject[batch['subject'][i]] = 1
                if batch['subject'][i] in all_count_for_subject:
                    all_count_for_subject[batch['subject'][i]] = all_count_for_subject[batch['subject'][i]] + 1
                else:
                    all_count_for_subject[batch['subject'][i]] = 1
        else:
            for i in range(len(answer)):
                if answer[i].item() == label[i].item():
                    right_count += 1
                all_count += 1
        if args.peft_model == 'hmora':
            model.router_manager.clear()

        if bar is not None:
            bar.update(1)

    if data_type == 'mmlu' or data_type == 'mmlu_pro':
        # Macro Average
        acc = [
            (right_count_for_subject[subject] if subject in right_count_for_subject else 0) / all_count_for_subject[
                subject]
            for subject in all_count_for_subject.keys()]
        acc = sum(acc) / len(acc)
    else:
        acc = right_count / all_count
    return acc, right_count_for_subject, all_count_for_subject
