import os
import sys

pdj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("--pdj:", pdj)
sys.path.append(pdj)

from fastchat.train.train import *


def my_make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_path
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print("Loading data...")

    train_json = json.load(open(data_path, "r"))

    train_dataset = LazySupervisedDataset(train_json, tokenizer=tokenizer)

    eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


# model_name_or_path = '/mnt/cephfs/hjh/huggingface_models/nlp/fastchat/vicuna-7b-v1.5'
model_name_or_path = "lmsys/vicuna-7b-v1.5"
cache_dir = "/tmp/fastchat"
data_path = "/home/huangjiahong.dracu/hjh/pycharm_projects/nlp/FastChat/data/dummy_conversation.json"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path,
    cache_dir=cache_dir,
    model_max_length=2048,
    padding_side="right",
    use_fast=False,
)

tokenizer.pad_token = tokenizer.unk_token

data_module = my_make_supervised_data_module(tokenizer=tokenizer, data_path=data_path)

for data in data_module['train_dataset']:
    print('====--==', data)
