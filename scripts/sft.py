import os
import random
import numpy as np
import torch
import torch.distributed as dist
import argparse
import datetime
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import sys
sys.path.append('')
from src.process_group_manager import ProcessGroupManager
from src.model import Qwen2ForCausalLMDistributedShard
from src.data import Dataset, LMCollator
from src.main_loop import train

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
    parser.add_argument("--train_data_file", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--train_reference_file", type=str, default=None)
    parser.add_argument("--train_micro_batch_size", type=int, default=32)
    parser.add_argument("--eval_data_file", type=str, default=None)
    parser.add_argument("--eval_data_dir", type=str, default=None)
    parser.add_argument("--eval_reference_file", type=str, default=None)
    parser.add_argument("--eval_micro_batch_size", type=int, default=64)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--force_padding_to_max_length", action="store_true")

    # Train arguments
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_micro_batches_per_batch", type=int, default=2)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--num_epoches", type=int, default=3)
    parser.add_argument("--save_epoch_intervals", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # wandb arguments
    parser.add_argument("--wandb_entity", type=str, default="kuaishou-search-tech")
    parser.add_argument("--wandb_project", type=str, default=None)
    
    args = parser.parse_args()

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
    shard = Qwen2ForCausalLMDistributedShard.from_pretrained(shard_path, torch_dtype=dtype)
    shard.distributed_init(process_group_manager)
    shard.to(dtype)
    if args.gradient_checkpointing:
        shard.gradient_checkpointing_enable()

    if process_group_manager.global_rank == process_group_manager.world_size - 1:
        model_memory = estimate_model_memory(compute_params(shard), args.dtype) / (1024 ** 3)
        activation_memory = estimate_activation_memory(compute_params(shard), args.train_micro_batch_size, args.max_length, args.dtype) / (1024 ** 3)
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

    # dist.barrier()

    set_all_seed(args.seed)
    # 载入数据
    collator = LMCollator(
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_path),
        prompt_path=args.prompt_path,
        max_length=args.max_length,
        force_padding_to_max_length=args.force_padding_to_max_length
    )
    train_dataset = Dataset(
        data_file=args.train_data_file,
        data_dir=args.train_data_dir,
        reference_file=args.train_reference_file
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collator
    )
    if args.eval_data_file is None and args.eval_data_dir is None and args.eval_reference_file is None:
        eval_dataloader = None
    else:
        eval_dataset = Dataset(
            data_file=args.eval_data_file,
            data_dir=args.eval_data_dir,
            reference_file=args.eval_reference_file
        )
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=args.eval_micro_batch_size if args.eval_micro_batch_size is not None else args.train_micro_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=collator
        )

    # 训练
    train(
        shard=shard,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=args.learning_rate,
        num_micro_batches_per_batch=args.num_micro_batches_per_batch,
        accumulate_steps=args.accumulate_steps,
        num_epoches=args.num_epoches,
        save_epoch_intervals=args.save_epoch_intervals,
        save_path=args.save_path,
        warmup_ratio=args.warmup_ratio,
        wandb_logger=wandb_logger
    )
    
    dist.destroy_process_group()
