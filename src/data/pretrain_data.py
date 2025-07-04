from datasets import load_from_disk
import torch

class InputIdsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        target_field=None,
        chunk_size=5120,
        **hf_datasets_load_kwargs
    ):
        self._dataset = load_from_disk(**hf_datasets_load_kwargs)
        if target_field is not None:
            assert target_field in self._dataset.column_names
            self.delete_column_names_except(target_field)
            self._dataset = self._dataset.rename_column(target_field, 'text')
        else:
            # TODO: 将不同的field组合成text
            target_field = 'text'
            raise NotImplementedErro()
        self.tokenizer = tokenizer
        
        self._dataset = self._dataset.map(
            lambda e: self.tokenizer(
                e[target_field], 
                max_length=192000,
                truncation=True,
                return_attention_mask=False,
                add_special_tokens=False
            ),
            remove_columns=self._dataset.column_names,
            batched=True,
            batch_size=500,
            num_proc=32,
            desc="Running tokenizer on dataset"
        )
        
        def group_texts(examples):
            chunks = []
            chunk = []
            chunk_left_size = chunk_size
            for input_ids in examples["input_ids"]:
                input_start_id = 0
                input_ids_len = len(input_ids)
                while input_start_id < input_ids_len:
                    chunk.extend(input_ids[input_start_id: input_start_id + chunk_left_size])
                    input_start_id += chunk_left_size
                    chunk_left_size = chunk_size - len(chunk)
                    if chunk_left_size == 0:
                        chunks.append(chunk)
                        chunk = []
                        chunk_left_size = chunk_size
            return {"input_ids": chunks}
        self._dataset = self._dataset.map(
            group_texts,
            remove_columns=self._dataset.column_names,
            batched=True,
            num_proc=32,
            desc=f"Grouping texts in chunks of {chunk_size}"
        )

    def delete_column_names_except(self, column_name):
        column_names_to_remove = self._dataset.column_names.copy()
        column_names_to_remove.remove(column_name)
        self._dataset = self._dataset.remove_columns(column_names_to_remove)
    
    def __len__(self):
        return len(self._dataset)
    
    def __item__(self, idx):
        return self._dataset[idx]["input_ids"]
