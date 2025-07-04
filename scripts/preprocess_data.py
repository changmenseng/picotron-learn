# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--input_file", type=str, required=True)
import itertools
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/etc/ssd1/jiangzhongtao/base_models/DeepSeek-R1-Distill-Qwen-7B")


dataset = load_dataset(
    "json", 
    split="train",
    data_files="/etc/ssd1/jiangzhongtao/qwen-mla-moba/data/Chinese-DeepSeek-R1-Distill-data-110k/train.jsonl"
)


dataset = dataset.map(
    lambda e: {"text": e["input"] + e["reasoning_content"] + e["content"]},
    num_proc=16,
    desc="Concatenating field"
)
column_names_to_remove = dataset.column_names.copy()
column_names_to_remove.remove('text')
dataset = dataset.remove_columns(column_names_to_remove)

dataset = dataset.map(
    lambda e: tokenizer(
        e["text"], 
        max_length=192000,
        truncation=True,
        return_attention_mask=False,
        add_special_tokens=False
    ),
    remove_columns=dataset.column_names,
    batched=True,
    batch_size=128,
    num_proc=16,
    desc="Running tokenizer on dataset"
)

chunk_size = 5120
def group_texts(examples):
    input_ids = sum(examples["input_ids"], [])
    total_length = len(input_ids)
    total_length = (total_length // chunk_size) * chunk_size
    input_ids = [input_ids[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
    return {"input_ids": input_ids}
dataset = dataset.map(
    group_texts,
    remove_columns=dataset.column_names,
    batched=True,
    num_proc=16,
    desc=f"Grouping texts in chunks of {chunk_size}"
)

dataset = dataset.train_test_split(test_size=0.01)

dataset.cleanup_cache_files()


