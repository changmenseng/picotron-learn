import os
import random
import numpy as np
import torch
import torch.distributed as dist
import argparse
import datetime
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk, concatenate_datasets

import sys
import sys
sys.path.append('')
from src.model import Qwen2ForCausalLMDistributedShard
from src.process_group_manager import ProcessGroupManager
from src.data import InputIdsCollator
from src.main_loop import train

def move_micro_batch_to_device(micro_batch, device):
    for key, value in micro_batch.items():
        micro_batch[key] = value.to(device)
    return micro_batch

def set_all_seed(seed):
    random.seed(seed)
    np.random.rand(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

def compute_params(m):
    cnt = 0
    for p in m.parameters():
        cnt += p.numel()
    return cnt

def estimate_activation_memory(params, bsz, seq_len, dtype='bfloat16'):
    num_bytes = (4.6894e-4 * params + 1.8494e6) * bsz * seq_len * 2
    if dtype == 'float32':
        return num_bytes * 2
    elif dtype == 'bfloat16' or dtype == 'float16':
        return num_bytes
    elif dtype == 'float8':
        return num_bytes / 2

def estimate_model_memory(params, dtype='bfloat16'):
    num_bytes = params * 2
    if dtype == 'float32':
        return num_bytes * 2
    elif dtype == 'bfloat16' or dtype == 'float16':
        return num_bytes
    elif dtype == 'float8':
        return num_bytes / 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Seed arguments
    parser.add_argument("--seed", type=int, default=1116)

    # Environment arguments
    # parser.add_argument("--omp_num_threads", type=str, default="1")
    # parser.add_argument("--tokenizers_parallelism", type=str, default="false")

    # Communicate arguments
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--timeout", type=int, default=30)


    # Model arguments
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default='bfloat16')

    # Parallel arguments
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--pp_size", type=int, default=2)

    # Data arguments
    parser.add_argument("--chunk_size", type=int, default=5120)
    parser.add_argument("--train_hf_dataset_dirs", type=str, default=None)
    parser.add_argument("--train_micro_batch_size", type=int, default=32)
    parser.add_argument("--eval_hf_dataset_dirs", type=str, default=None)
    parser.add_argument("--eval_micro_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)

    # Train arguments
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_micro_batches_per_batch", type=int, default=2)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--num_epoches", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--save_epoch_intervals", type=int, default=None)
    parser.add_argument("--save_step_intervals", type=int, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # wandb arguments
    parser.add_argument("--wandb_entity", type=str, default="kuaishou-search-tech")
    parser.add_argument("--wandb_project", type=str, default=None)
    
    args = parser.parse_args()

    args.train_hf_dataset_dirs = args.train_hf_dataset_dirs.split(',')
    try:
        args.eval_hf_dataset_dirs = args.eval_hf_dataset_dirs.split(',')
    except Exception:
        pass


    # os.environ["OMP_NUM_THREADS"] = 1
    # os.environ["TOKENIZERS_PARALLELISM"] = False

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl"
    # torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = getattr(torch, args.dtype)

    dist.init_process_group(
        rank=global_rank, 
        world_size=world_size, 
        backend=args.backend, 
        init_method=f"env://", 
        timeout=datetime.timedelta(minutes=args.timeout)
    )

    # 进程管理
    process_group_manager = ProcessGroupManager(args.tp_size, 1, args.pp_size, 1)

    # 载入wandb
    wandb_logger = None
    if process_group_manager.global_rank == process_group_manager.world_size - 1:
        if args.wandb_entity is not None and args.wandb_project is not None:
            import wandb
            wandb_logger = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                config=args.__dict__
            )


    # 载入模型
    shard_path = os.path.join(args.model_path, f'tp-{process_group_manager.tp_rank}_pp-{process_group_manager.pp_rank}')
    shard = Qwen2ForCausalLMDistributedShard.from_pretrained(
        shard_path, 
        torch_dtype=dtype,
        _attn_implementation='flash_attention_2'
    )
    shard.distributed_init(process_group_manager)
    shard.to(dtype)
    if args.gradient_checkpointing:
        shard.gradient_checkpointing_enable({"use_reentrant": False})
    
    if process_group_manager.global_rank == process_group_manager.world_size - 1:
        for name, param in shard.named_parameters():
            if param.requires_grad:
                print(name)

    if process_group_manager.global_rank == process_group_manager.world_size - 1:
        model_memory = estimate_model_memory(compute_params(shard), args.dtype) / (1024 ** 3)
        activation_memory = estimate_activation_memory(compute_params(shard), args.train_micro_batch_size, args.chunk_size, args.dtype) / (1024 ** 3)
        print("Estimated memory:")
        print('================================================')
        # Fixed
        print('Fixed')
        print(f"Model: {model_memory} GB")
        print(f"AdamW 1-moment: {model_memory * 2} GB") # fp32
        print(f"AdamW 2-moment: {model_memory * 2} GB")
        print('------------------------------------------------')
        # Dynamic
        print('Dynamic')
        print(f"Activations: {activation_memory * args.num_micro_batches_per_batch} GB")
        print(f"Gradient: {model_memory} GB")
        print(f"Optimizer intermediates: {model_memory} GB")
        print('------------------------------------------------')
        print(f"Runtime max: {model_memory * 5 + max(model_memory * 2, activation_memory * args.num_micro_batches_per_batch)} GB")

    shard.to(device)
    shard.train()
    

    # dist.barrier()
    print(process_group_manager.global_rank, compute_params(shard))

    set_all_seed(args.seed)
    # 载入数据
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.padding_side = 'right'
    train_dataset = []
    for train_hf_dataset_dir in args.train_hf_dataset_dirs:
        _train_dataset = load_from_disk(train_hf_dataset_dir)
        train_dataset.append(_train_dataset)
    if len(train_dataset) == 1:
        train_dataset = train_dataset[0]
    else:
        train_dataset = concatenate_datasets(train_dataset)
    collator = InputIdsCollator()
    collator.force_padding_to_max_length = True
    collator.max_length = args.chunk_size
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collator
    )
    # TODO: 引入eval数据的载入流程
    eval_dataloader = None

    # 训练
    train(
        shard=shard,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=args.learning_rate,
        num_micro_batches_per_batch=args.num_micro_batches_per_batch,
        accumulate_steps=args.accumulate_steps,
        num_epoches=args.num_epoches,
        num_steps=args.num_steps,
        save_epoch_intervals=args.save_epoch_intervals,
        save_step_intervals=args.save_step_intervals,
        save_path=args.save_path,
        warmup_ratio=args.warmup_ratio,
        wandb_logger=wandb_logger
    )
    
    dist.destroy_process_group()
