import os
import random
# from dataclasses import dataclass
import torch.distributed as dist
import torch
import jinja2

class CLSCollator:

    def __init__(
        self,
        tokenizer,
        user_prompt_template,
        label_dict,
        system_prompt: str = None,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.label_dict = label_dict

        self.not_verbose = True
    
    def __call__(self, raw_batch):

        prompts = []
        labels = []
        for item in raw_batch:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_template.format(**item)}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            label = self.label_dict[item['label']]

            prompts.append(prompt)
            labels.append(label)
        

        if self.not_verbose and dist.get_rank() == 0:
            print(prompts[0])
            self.not_verbose = False

        encoding = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = torch.tensor(labels)
        
        batch = {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "labels": labels
        }
        return batch

class LMCollator:

    def __init__(
        self,
        tokenizer,
        prompt_path,
        max_length: int = 512,
        force_padding_to_max_length: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.force_padding_to_max_length = force_padding_to_max_length

        if os.path.exists(os.path.join(prompt_path, 'system_prompt.txt')):
            with open(os.path.join(prompt_path, 'system_prompt.txt'), 'r') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = None
        
        if os.path.exists(os.path.join(prompt_path, 'input_template.txt')):
            with open(os.path.join(prompt_path, 'input_template.txt'), 'r') as f:
                self.input_template = jinja2.Template(f.read())
            self.user_prompt_template = None
        else:
            self.input_template = None
            assert os.path.exists(os.path.join(prompt_path, 'user_prompt_template.txt'))
            with open(os.path.join(prompt_path, 'user_prompt_template.txt'), 'r') as f:
                self.user_prompt_template = jinja2.Template(f.read())
        
        with open(os.path.join(prompt_path, 'output_template.txt'), 'r') as f:
            self.output_template = jinja2.Template(f.read())
        
        self.verbose = True
        if dist.is_initialized():
            self.verbose = dist.get_rank() == 0

    def __call__(self, raw_batch):

        pairs = []
        for example in raw_batch:
            if self.input_template is not None:
                prompt = self.input_template.render(
                    system_prompt=self.system_prompt,
                    **example
                )
            else:
                messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt is not None else []
                messages.append({"role": "user", "content": self.user_prompt_template.render(**example)})
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            output = self.output_template.render(**example) + self.tokenizer.eos_token
            pairs.append([prompt, output])

        if self.verbose:
            print(pairs[0])
            self.verbose = False


        encoding = self.tokenizer(
            pairs,
            padding='max_length' if self.force_padding_to_max_length else True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=True,
            add_special_tokens=False
        )

        labels = torch.where(
            encoding.token_type_ids.bool(),
            encoding.input_ids, -100
        )


        batch = {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "labels": labels
        }

        return batch

class InputIdsCollator:
    
    def __call__(self, raw_batch):
        input_ids = []
        for example in raw_batch:
            input_ids.append(example['input_ids'])
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        batch = {
            "input_ids": input_ids,
            "labels": input_ids
        }
        return batch
