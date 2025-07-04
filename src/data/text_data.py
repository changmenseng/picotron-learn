import os
import random
# from dataclasses import dataclass
import torch.distributed as dist
from typing import List, Tuple
import json
import torch
from tqdm import tqdm
import jinja2
import math

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_file=None, 
        data_dir=None,
        reference_file=None
    ):
        """
            Args:
                data_file： 单个数据文件
                data_dir：文件夹，该文件夹下的所有文件都是数据文件
                reference_file：该文件的每一行存储了一个数据文件路径和该数据文件需要采样的数据量，以`\t`隔开。若数据量为-1，则取全量数据。
        """
        assert not (data_file is None and data_dir is None and reference_file is None)
        data_info = []
        if data_file is not None:
            assert data_dir is None and reference_file is None
            data_info.append((data_file, -1))

        if data_dir is not None:
            assert data_file is None and reference_file is None
            for fname in os.listdir(data_dir):
                data_info.append((os.path.join(data_dir, fname), -1))
        
        if reference_file is not None:
            with open(reference_file, 'r') as f:
                for line in f:
                    data_file, data_size = line.strip().split('\t')
                    data_size = int(data_size)
                    data_info.append((data_file, data_size))
        
        pbar_disable = False
        if dist.is_initialized():
            pbar_disable = dist.get_rank() != 0
        pbar = tqdm(desc='Loading data', smoothing=0, disable=pbar_disable)

        self.dataset = []
        for data_file, data_size in data_info:
            start_index = len(self.dataset)
            if data_size == -1: # 全部都需要添加并训练
                for example in self._load_data(data_file):
                    self.dataset.append(example)
                    pbar.update(1)
            else: # 蓄水池采样
                self.dataset.extend([None for _ in range(data_size)])
                for i, example in enumerate(self._load_data(data_file)):
                    if i < data_size:
                        self.dataset[start_index + i] = example
                    else:
                        if random.random() < data_size / (i + 1): # 替换元素
                            self.dataset[start_index + random.randint(0, data_size - 1)] = example
                for i in range(start_index, len(self.dataset)): # 验证是否正确，第二次运行时可删除
                    pbar.update(1)
                    try:
                        assert self.dataset[i] is not None
                    except Exception:
                        raise RuntimeError(data_file)    
        pbar.close()

    def _load_data(self, fname):
        with open(fname, 'r') as f:
            for line in f:
                line = json.loads(line)
                yield line # (query, rewrite, intent)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[str, List[str]]:
        example = self.dataset[idx]
        return example
