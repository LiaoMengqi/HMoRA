import os
import json
import torch
import subprocess
import re
import random
import numpy as np
from transformers import set_seed as transformers_seed
from typing import Optional


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_json(data, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + file_name, 'w') as f:
        json.dump(data, f)


def save_check_point(model, args, tokenizer=None):
    import warnings
    warnings.simplefilter("ignore")
    if args.save_dir is None:
        path = './checkpoint/' + model.config.model_type + '/'
    else:
        path = args.save_dir + model.config.model_type + '/'

    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    model.save_pretrained(path)


def select_gpu(exclude_gpu):
    try:
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
    except UnicodeDecodeError:
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode("gbk")
    used_list = re.compile(r"(\d+)MiB\s+/\s+\d+MiB").findall(nvidia_info)
    used = [(idx, int(num)) for idx, num in enumerate(used_list)]
    sorted_used = sorted(used, key=lambda x: x[1])
    for idx, _ in sorted_used:
        if idx not in exclude_gpu:
            return idx
    return sorted_used[0][0]


def set_device(gpu, exclude_gpu=None):
    if exclude_gpu is None:
        exclude_gpu = []
    if gpu != -1:
        # use gpu
        if not torch.cuda.is_available():
            # gpu not available
            print('No GPU available. Using CPU.')
            device = 'cpu'
        else:
            # gpu available
            if gpu < -1:
                # auto select gpu
                gpu_id = select_gpu(exclude_gpu)
                print('Auto select gpu:%d' % gpu_id)
                device = 'cuda:%d' % gpu_id
            else:
                # specify gpu id
                if gpu >= torch.cuda.device_count():
                    gpu_id = select_gpu(exclude_gpu)
                    print('GPU id is invalid. Auto select gpu:%d' % gpu_id)
                    device = 'cuda:%d' % gpu_id
                else:
                    print('Using gpu:%d' % gpu)
                    device = 'cuda:%d' % gpu
    else:
        print('Using CPU.')
        device = 'cpu'
    return device
