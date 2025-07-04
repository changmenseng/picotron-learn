import os
import argparse
import warnings

import sys
sys.path.append('')

from src.model import Qwen2ForCausalLMDistributedShard
from src.convert import combine_shards

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src_path', type=str, required=True)
parser.add_argument('-t', '--target_path', type=str, default=None)
parser.add_argument('--dtype', type=str, default='bfloat16')
args = parser.parse_args()

if args.target_path is None:
    args.target_path = os.path.join(args.src_path, 'hf')
    warnings.warn(f'--target_path not specified, default: {args.src_path}')


shards = []
for shard_dir in os.listdir(args.src_path):
    shard_path = os.path.join(args.src_path, shard_dir)
    shard = Qwen2ForCausalLMDistributedShard.from_pretrained(shard_path)
    shards.append(shard)

model = combine_shards(shards, True)
try:
    os.makedirs(args.target_path)
except Exception:
    pass
model.save_pretrained(args.target_path)
