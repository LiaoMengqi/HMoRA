import datasets
from torch.utils.data import DataLoader

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def load_dataset(dataset: list[str], data_type: list[str], eval_batch_size: int):
    dataset_list = []
    for i in range(len(dataset)):
        if data_type[i] == "mmlu":
            data = datasets.load_dataset(
                "parquet", split="train", data_files=dataset[i]
            )
            data = data.map(format_mmlu)
        elif data_type[i] == "mmlu_pro":
            data = datasets.load_dataset(
                "parquet", split="train", data_files=dataset[i]
            )
            data = data.map(format_mmlu_pro)
        elif data_type[i] == "arc_e" or data_type[i] == "arc_c":
            data = datasets.load_dataset(
                "parquet", split="train", data_files=dataset[i]
            )
            data = data.map(format_arc)
        elif data_type[i] == "swag":
            data = datasets.load_dataset(
                "parquet", split="train", data_files=dataset[i]
            )
            data = data.map(format_swag)
        elif data_type[i] == "commonsenseqa":
            data = datasets.load_dataset(
                "parquet", split="train", data_files=dataset[i]
            )
            data = data.map(format_commonsenseqa)
        elif data_type[i] == "openbookqa":
            data = datasets.load_dataset(
                "parquet", split="train", data_files=dataset[i]
            )
            data = data.map(format_openbookqa)
        else:
            raise ValueError(f"valid_data_type not supported!")
        data = remove_column(data)
        data = DataLoader(data, batch_size=eval_batch_size, shuffle=True)
        dataset_list.append(data)
    return dataset_list


def cat_options(options):
    return "\n".join([f"{choices[i]}. {option}" for i, option in enumerate(options)])


def format_mmlu(example):
    def format_subject(subject):
        return subject.replace("_", " ").lower()

    prompt = (
        f'Below is a multiple-choice question about {format_subject(example["subject"])}. '
        f"Please choose the correct answer.\n"
    )
    prompt += example["question"] + "\nOptions:"
    for j in range(len(example["choices"])):
        prompt += "\n{}. {}".format(choices[j], example["choices"][j])
    prompt += "\nAnswer: "
    example["source"] = prompt
    example["target"] = choices[example["answer"]]
    example["target_id"] = example["answer"]
    return example


def format_mmlu_pro(example):
    prompt = (
        f'Below is a multiple-choice question about {format_subject(example["category"])}.'
        " Please choose the correct answer.\n"
    )
    prompt += example["question"] + "\nOptions:"
    for j in range(len(example["options"])):
        prompt += "\n{}. {}".format(choices[j], example["options"][j])
    prompt += "\nAnswer: "
    example["subject"] = example["category"]
    example["source"] = prompt
    example["target"] = example["answer"]
    example["target_id"] = example["answer_index"]
    return example


def format_arc(example):
    prompt = "Below is a multiple-choice question. Please choose the correct answer.\n"
    prompt += example["question"] + "\nOptions:"
    for j in range(len(example["choices"]["text"])):
        prompt += "\n{}. {}".format(choices[j], example["choices"]["text"][j])
    prompt += "\nAnswer: "
    example["source"] = prompt
    example["target"] = example["answerKey"]
    example["target_id"] = example["choices"]["label"].index(example["answerKey"])
    return example


def format_swag(example):
    options = ["A", "B", "C", "D"]
    prompt = "Choose the sentence that best completes the start phrase below:\n"
    prompt += "start phrase: " + example["startphrase"] + "\nOptions:\n"
    prompt += "A. " + example["ending0"] + "\n"
    prompt += "B. " + example["ending1"] + "\n"
    prompt += "C. " + example["ending2"] + "\n"
    prompt += "D. " + example["ending3"] + "\n"
    prompt += "Answer: "
    example["source"] = prompt
    example["target"] = options[example["label"]]
    example["target_id"] = choices.index(example["target"])
    return example


def format_commonsenseqa(example):
    prompt = "Below is a multiple-choice question. Please choose the correct answer.\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, choice in enumerate(example["choices"]["text"]):
        prompt += choices[i] + ". " + choice + "\n"
    prompt += "Answer: "
    example["source"] = prompt
    example["target"] = example["answerKey"]
    example["target_id"] = choices.index(example["answerKey"])
    return example


def format_openbookqa(example):
    prompt = "Consider the fact and answer the following question by selecting the correct option:\n"
    prompt += "Fact: " + example["fact1"] + ".\n"
    prompt += "Question: " + example["question_stem"] + "\n"
    prompt += "Options:\n"
    for i, choice in enumerate(example["choices"]["text"]):
        prompt += choices[i] + ". " + choice + "\n"
    prompt += "Answer: "
    example["source"] = prompt
    example["target"] = example["answerKey"]
    example["target_id"] = choices.index(example["answerKey"])
    return example


def remove_column(dataset):
    save_column = ["source", "target", "target_id", "subject"]
    columns_to_remove = []
    for column_name in dataset.column_names:
        if column_name not in save_column:
            columns_to_remove.append(column_name)
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def format_source(source, model_type, few_shot_prompt=None, training=True):
    source = format_source_for_qwen2(source, training)
    return source


def format_source_for_qwen2(source, training=True):
    prefix = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>human\n"
    source = prefix + source + "\n<|im_end|>\n"
    if training:
        return source
    return source + "<|im_start|>assistant\n"


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(example, include_answer=True):
    prompt = example["source"]
    if include_answer:
        prompt += "{}\n\n".format(example["target"])
    return prompt


def gen_few_shot_prompt(few_shot_df, k):
    prompt = ""
    if few_shot_df is None:
        return prompt
    for i in range(k):
        prompt += format_example(few_shot_df[i])
    return prompt
