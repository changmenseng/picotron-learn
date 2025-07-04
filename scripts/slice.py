import argparse
from transformers import Qwen2ForCausalLM

import sys
import os
import torch
import warnings

sys.path.append('')
from src.convert import slice_model

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src_path', type=str, required=True)
parser.add_argument('-t', '--target_path', type=str, default=None)
parser.add_argument('--tp_size', type=int, default=4)
parser.add_argument('--pp_size', type=int, default=2)
parser.add_argument('--dtype', type=str, default='bfloat16')
args = parser.parse_args()

if args.target_path is None:
    args.target_path = os.path.join(args.src_path, f'picotron/tp-{args.tp_size}_pp-{args.pp_size}')
    warnings.warn(f'--target_path not specified, default: {args.target_path}')

try:
    os.makedirs(args.target_path)
except Exception:
    pass

model = Qwen2ForCausalLM.from_pretrained(args.src_path, torch_dtype=getattr(torch, args.dtype))

for i, shard in enumerate(slice_model(model, args.tp_size, args.pp_size, True)):
    pp_rank = i // args.tp_size
    tp_rank = i % args.tp_size
    shard.save_pretrained(os.path.join(args.target_path, f'tp-{tp_rank}_pp-{pp_rank}'))
