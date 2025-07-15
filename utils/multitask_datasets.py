import logging
from typing import List, Optional, Dict

import datasets as hf_datasets
import torch

from .data_process import *


# Code modified from https://github.com/TUDB-Labs/MoE-PEFT.git


arc_path = "/path/to/your/data/arc"
boolq_path = "/path/to/your/data/boolq"
obqa_path = "/path/to/your/data/openbookqa"
piqa_path = "/path/to/your/data/piqa"
siqa_path = "/path/to/your/data/siqa"
hellaswag_path = "/path/to/your/data/hellaswag"
winogrande_path = "/path/to/your/data/winogrande/winogrande_debiased"
commonsenseqa_path = "/path/to/your/data/commonsenseqa"
pubmedqa_path = "/path/to/your/data/pubmedqa"


class ARC(object):
    def __init__(self, subject: str) -> None:
        self.labels_ = ["1", "2", "3", "4", "A", "B", "C", "D", "E"]
        self.labels2id_ = {text: idx for idx, text in enumerate(self.labels_)}
        self.label_dtype_ = torch.int
        assert subject in ["ARC-Easy", "ARC-Challenge"]
        self.subject_ = subject

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[dict]:
        data = hf_datasets.load_dataset(
            "allenai/ai2_arc" if arc_path is None else arc_path, self.subject_
        )["train" if is_train else "test"]
        logging.info(f"Preparing data for {self.subject_}")
        ret: List[dict] = []
    
        for data_point in data:
            example = {}
            example["question"] = data_point["question"]
            example["choices"] = data_point["choices"]
            example["answerKey"] = data_point["answerKey"]
            
            example = format_arc(example)
            ret.append(example)

        return ret


class BoolQ(object):
    def __init__(self) -> None:
        self.labels_ = ["A", "B"]
        self.labels2id_ = {text: idx for idx, text in enumerate(self.labels_)}
        self.label_dtype_ = torch.int

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[dict]:
        data = hf_datasets.load_dataset("google/boolq" if boolq_path is None else boolq_path)[
            "train" if is_train else "validation"
        ]
        logging.info("Preparing data for BoolQ")
        ret: List[dict] = []
        for data_point in data:
            # Use the same format as evaluation
            prompt = "Please answer the following question with true or false:\n"
            prompt += "Question: " + data_point["question"] + "\n"
            prompt += "Options:\n"
            prompt += "A. true\n"
            prompt += "B. false\n"
            prompt += "Answer: "
            
            # Convert boolean answer to A/B format, consistent with evaluation
            answer = "A" if data_point["answer"] else "B"
                
            example = {
                "source": prompt,
                "target": answer,
            }
            
            ret.append(example)

        return ret


class OpenBookQA(object):
    def __init__(self) -> None:
        self.labels_ = ["A", "B", "C", "D"]
        self.labels2id_ = {text: idx for idx, text in enumerate(self.labels_)}
        self.label_dtype_ = torch.int

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[dict]:
        data = hf_datasets.load_dataset(
            "allenai/openbookqa" if obqa_path is None else obqa_path, "main"
        )["train" if is_train else "test"]
        logging.info("Preparing data for OpenBookQA")
        ret: List[dict] = []
        for data_point in data:
            example = {}
            # Use the same format as evaluation
            prompt = "Please choose the correct answer to the question: "
            prompt += data_point["question_stem"] + "\n"
            prompt += "Options:\n"
            choices = data_point["choices"]
            for i, choice in enumerate(choices["text"]):
                prompt += f"{choices['label'][i]}. {choice}\n"
            prompt += "Answer: "
            example["source"] = prompt
            example["target"] = data_point["answerKey"]
            ret.append(example)

        return ret


class PIQA(object):
    def __init__(self, labels: List[str]=["A", "B"]) -> None:
        self.labels_ = labels
        self.labels2id_ = {text: idx for idx, text in enumerate(self.labels_)}
        self.label_dtype_ = torch.int

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[dict]:
        data = hf_datasets.load_dataset(
            "piqa" if piqa_path is None else piqa_path, trust_remote_code=True
        )["train" if is_train else "validation"]
        logging.info("Preparing data for PIQA")
        ret: List[dict] = []
        for data_point in data:
            example = {}
            # Use the same format as evaluation
            prompt = "Below is a common task along with two possible solutions. Please select the appropriate solution to achieve the task:\n"
            prompt += "Task: " + data_point["goal"] + "\n"
            prompt += "Options:\n"
            prompt += "A. " + data_point["sol1"] + "\n"
            prompt += "B. " + data_point["sol2"] + "\n"
            prompt += "Answer: "
            answer = self.labels_[data_point["label"]]
            example["source"] = prompt
            example["target"] = answer
            ret.append(example)

        return ret


def update_task_dict(task_dict):
    task_dict.update(
        {
            "arc-e": ARC("ARC-Easy"),
            "arc-c": ARC("ARC-Challenge"),
            "boolq": BoolQ(),
            "obqa": OpenBookQA(),
            "piqa": PIQA(),
        }
    )

class MultiTaskDatasets(torch.utils.data.Dataset):
    def __init__(
        self, 
        task_names: List[str], 
        is_train: bool = True,
        sample_strategy: str = "proportional",  # proportional, uniform, or custom
        task_weights: Optional[Dict[str, float]] = None,
        max_samples_per_task: Optional[int] = None,
        seed: int = 42
    ):
        """
        Multi-task dataset class
        
        Args:
            task_names: List of task names, e.g. ["arc-e", "boolq", "piqa"]
            is_train: Whether this is training data
            sample_strategy: Sampling strategy
                - "proportional": Sample proportionally to each task's data size
                - "uniform": Sample equal amounts from each task
                - "custom": Use task_weights for custom sampling weights
            task_weights: Sampling weights for each task when sample_strategy is "custom"
            max_samples_per_task: Maximum number of samples per task limit
            seed: Random seed
        """
        super().__init__()
        self.task_names = task_names
        self.is_train = is_train
        self.sample_strategy = sample_strategy
        self.task_weights = task_weights or {}
        self.max_samples_per_task = max_samples_per_task
        self.seed = seed
        
        # Initialize task dictionary
        self.task_dict = {}
        update_task_dict(self.task_dict)
        
        # Load data for all tasks
        self.task_data = {}
        self.task_sizes = {}
        self.all_data = []
        self.task_labels = []  # Record which task each sample belongs to
        
        self._load_all_tasks()
        self._mix_data()
        
    def _load_all_tasks(self):
        """Load data for all tasks"""
        logging.info(f"Loading {len(self.task_names)} tasks for {'training' if self.is_train else 'evaluation'}")
        
        for task_name in self.task_names:
            if task_name not in self.task_dict:
                raise ValueError(f"Unknown task: {task_name}")
            
            task = self.task_dict[task_name]
            data = task.loading_data(is_train=self.is_train)
            
            # If max_samples_per_task is set, truncate the data
            if self.max_samples_per_task is not None and len(data) > self.max_samples_per_task:
                import random
                random.seed(self.seed)
                data = random.sample(data, self.max_samples_per_task)
            
            self.task_data[task_name] = data
            self.task_sizes[task_name] = len(data)
            logging.info(f"Loaded {len(data)} samples from {task_name}")
    
    def _mix_data(self):
        """Mix data from all tasks according to sampling strategy"""
        import random
        random.seed(self.seed)
        
        if self.sample_strategy == "proportional":
            # Proportional mixing: directly combine all data
            for task_name, data in self.task_data.items():
                for sample in data:
                    sample_with_task = sample.copy()
                    sample_with_task['task'] = task_name  # Add task identifier
                    self.all_data.append(sample_with_task)
                    self.task_labels.append(task_name)
                    
        elif self.sample_strategy == "uniform":
            # Uniform sampling: sample equal amounts from each task
            min_size = min(self.task_sizes.values())
            for task_name, data in self.task_data.items():
                sampled_data = random.sample(data, min_size)
                for sample in sampled_data:
                    sample_with_task = sample.copy()
                    sample_with_task['task'] = task_name
                    self.all_data.append(sample_with_task)
                    self.task_labels.append(task_name)
                    
        elif self.sample_strategy == "custom":
            # Custom weight sampling
            if not all(task in self.task_weights for task in self.task_names):
                raise ValueError("task_weights must contain weights for all tasks when using custom strategy")
            
            # Calculate the number of samples to draw from each task
            total_weight = sum(self.task_weights[task] for task in self.task_names)
            total_samples = sum(self.task_sizes.values())
            
            for task_name, data in self.task_data.items():
                weight = self.task_weights[task_name]
                n_samples = int(total_samples * weight / total_weight)
                n_samples = min(n_samples, len(data))  # Cannot exceed actual data size
                
                sampled_data = random.sample(data, n_samples)
                for sample in sampled_data:
                    sample_with_task = sample.copy()
                    sample_with_task['task'] = task_name
                    self.all_data.append(sample_with_task)
                    self.task_labels.append(task_name)
        
        # Shuffle the data
        if self.is_train:
            indices = list(range(len(self.all_data)))
            random.shuffle(indices)
            self.all_data = [self.all_data[i] for i in indices]
            self.task_labels = [self.task_labels[i] for i in indices]
        
        logging.info(f"Total samples after mixing: {len(self.all_data)}")
        
        # Count samples for each task
        task_counts = {}
        for task in self.task_labels:
            task_counts[task] = task_counts.get(task, 0) + 1
        logging.info(f"Task distribution: {task_counts}")
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        """Returns format consistent with mini_batch in training code"""
        return self.all_data[idx]
    
    def get_task_distribution(self):
        """Get task distribution statistics"""
        task_counts = {}
        for task in self.task_labels:
            task_counts[task] = task_counts.get(task, 0) + 1
        return task_counts
    
    def get_collate_fn(self):
        """Return a collate function for DataLoader"""
        def collate_fn(batch):
            """Organize samples in batch into format required by training code"""
            sources = [item['source'] for item in batch]
            targets = [item['target'] for item in batch]
            tasks = [item.get('task', 'unknown') for item in batch]
            
            return {
                'source': sources,
                'target': targets,
                'task': tasks
            }
        return collate_fn


# Usage example
if __name__ == "__main__":
    # Create multi-task dataset
    task_names = ["arc-e", "arc-c", "boolq", "obqa", "piqa"]
    
    # Method 1: Proportional mixing
    train_dataset = MultiTaskDatasets(
        task_names=task_names,
        is_train=True,
        sample_strategy="proportional"
    )
    
    # Method 2: Uniform sampling
    train_dataset_uniform = MultiTaskDatasets(
        task_names=task_names,
        is_train=True,
        sample_strategy="uniform"
    )
    
    # Method 3: Custom weights
    train_dataset_custom = MultiTaskDatasets(
        task_names=task_names,
        is_train=True,
        sample_strategy="custom",
        task_weights={
            "arc-e": 0.3,
            "arc-c": 0.2,
            "boolq": 0.2,
            "obqa": 0.15,
            "piqa": 0.15
        }
    )
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=train_dataset.get_collate_fn()
    )
    
    # Test data loading
    for batch in train_dataloader:
        print(f"Batch sources: {len(batch['source'])}")
        print(f"Batch targets: {len(batch['target'])}")
        print(f"Batch tasks: {batch['task']}")
        break