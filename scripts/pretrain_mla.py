import os
import random
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import torch.optim as optim
import torch.distributed as dist
import argparse
import datetime
import functools
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from datasets import load_from_disk, concatenate_datasets

import sys
sys.path.append('/etc/ssd1/jiangzhongtao/code_framework/train/picotron')
from src.process_group_manager import ProcessGroupManager
from src.data import InputIdsCollator
from src.main_loop import train


sys.path.append('')
from models.qwen2_mla_moba.picotron.modeling import Qwen2MLAMoBAShardForCausalLM

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

# Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
def gradient_checkpointing_enable(shard, gradient_checkpointing_kwargs=None):
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}
    gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
    shard._gradient_checkpointing_func = gradient_checkpointing_func
    shard.gradient_checkpointing = True

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
    parser.add_argument("--kv_a_layernorm_type", type=str, default='rmsnorm', choices=["rmsnorm", "dyt", "none"])

    # Parallel arguments
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--dp_size", type=int, default=2)

    # Data arguments
    parser.add_argument("--chunk_size", type=int, default=5120)
    parser.add_argument("--train_hf_dataset_dirs", type=str, default=None)
    parser.add_argument("--train_micro_batch_size", type=int, default=32)
    parser.add_argument("--eval_hf_dataset_dirs", type=str, default=None)
    parser.add_argument("--eval_micro_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)

    # Train arguments
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--attn_learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_learning_ratio", type=float, default=None)
    parser.add_argument("--learning_rate_num_cycles", type=float, default=None)
    parser.add_argument("--use_zero_optimizer", action="store_true")
    parser.add_argument("--num_micro_batches_per_batch", type=int, default=2)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--num_epoches", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--save_epoch_intervals", type=int, default=None)
    parser.add_argument("--save_step_intervals", type=int, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--only_train_attn", action="store_true")

    # wandb arguments
    parser.add_argument("--wandb_entity", type=str, default="kuaishou-search-tech")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_runname", type=str, default=None)
    
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
    torch.cuda.set_device(local_rank)
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
    process_group_manager = ProcessGroupManager(args.tp_size, 1, args.pp_size, args.dp_size)
    
    # if process_group_manager.global_rank == process_group_manager.world_size - 1:
    #     torch.cuda.memory._record_memory_history(max_entries=10000000)

    # 载入wandb
    wandb_logger = None
    if process_group_manager.global_rank == process_group_manager.world_size - 1:
        if args.wandb_entity is not None and args.wandb_project is not None:
            import wandb
            wandb_logger = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                group=args.wandb_group,
                name=args.wandb_runname,
                config=args.__dict__,
            )

    # 载入模型
    shard_path = os.path.join(args.model_path, f'tp-{process_group_manager.tp_rank}_pp-{process_group_manager.pp_rank}')
    shard = Qwen2MLAMoBAShardForCausalLM.from_pretrained(
        shard_path, 
        torch_dtype=dtype,
        _attn_implementation='flash_attention_2',
        kv_a_layernorm_type=args.kv_a_layernorm_type
    )
    shard.distributed_init(process_group_manager)
    shard.train()
    shard.to(dtype)
    # # 这里需要把norm搞成32位的，避免下溢
    # for layer in shard.model.layers:
    #     layer.self_attn.kv_a_layernorm.to(torch.float32)
    # shard.model.norm.to(torch.float32)
    if args.gradient_checkpointing:
        # shard.gradient_checkpointing_enable({"use_reentrant": False})
        gradient_checkpointing_enable(shard, {"use_reentrant": False})
    shard.to(device)
    
    if process_group_manager.dp_world_size > 1:
        # set `gradients_as_bucket_view=True` to reduce memory usage (But I found not useful...). 
        # https://discuss.pytorch.org/t/memory-consumption-for-the-model-get-doubled-after-wrapped-with-ddp/130837/5
        # https://stackoverflow.com/questions/68949954/model-takes-twice-the-memory-footprint-with-distributed-data-parallel
        shard = DDP(
            shard, 
            device_ids=[local_rank], 
            output_device=local_rank,
            process_group=process_group_manager.dp_group,
            gradient_as_bucket_view=True
        )

        # print(shard._get_ddp_logging_data().get("can_set_static_graph"))
        shard.process_group_manager = process_group_manager
        shard.config = shard.module.config
        shard.dtype = shard.module.dtype
        torch.cuda.empty_cache()

    # There are some higher ops, so we need to disable DDPOptimizer, which disable the bucket communication.
    # However, if we use accumulate grad, this hardly affect the performance.
    # https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860
    # https://discuss.pytorch.org/t/how-should-i-use-torch-compile-properly/179021
    import torch._dynamo
    torch._dynamo.config.optimize_ddp = False
    # both `mode="reduce-overhead"` and `mode="max-autotune"` will cause NaN issues.
    shard = torch.compile(shard)

    param_groups = None
    attn_module_substrs = {
        "q_proj", "kv_a_proj", "k_rope_proj", "kv_a_layernorm", 
        "kv_b_proj", "o_proj", "input_layernorm", "post_attention_layernorm"
    }
    def get_attn_params():
        for name, param in shard.named_parameters():
            for module_substr in attn_module_substrs:
                if module_substr in name:
                    yield param
                    break
    def get_nonattn_params():
        for name, param in shard.named_parameters():
            for module_substr in attn_module_substrs:
                if module_substr in name:
                    break
            else: # 没有break，说明名字中没有attn的参数中缀，因此，直接yield
                yield param

    if args.only_train_attn:
        for param in get_nonattn_params():
            param.requires_grad = False
    else:
        if args.learning_rate != args.attn_learning_rate:
            param_groups = [
                {
                    "params": get_attn_params(), 
                    "lr": torch.tensor(args.attn_learning_rate)
                },
                {
                    "params": get_nonattn_params(), 
                    "lr": torch.tensor(args.learning_rate)
                }
            ]
        
    # if process_group_manager.global_rank == process_group_manager.world_size - 1:
    #     for name, param in shard.named_parameters():
    #         if param.requires_grad:
    #             print(name)

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
    
    sampler = None
    # if process_group_manager.dp_world_size > 1:
    #     sampler = DistributedSampler(
    #         train_dataset,
    #         num_replicas=process_group_manager.dp_world_size,
    #         rank=process_group_manager.dp_rank,
    #         shuffle=True,
    #         seed=args.seed,
    #         drop_last=True
    #     ) # 限制采样的样本数量为原始的1/dp_world_size
    collator = InputIdsCollator()
    collator.force_padding_to_max_length = True
    collator.max_length = args.chunk_size
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_micro_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collator,
        pin_memory=True
    )
    # TODO: 引入eval数据的载入流程
    eval_dataloader = None

    # ########
    # # test #
    # ########
    # args.save_path = None
    # args.num_steps = 2
    # args.num_micro_batches_per_batch = 2

    # args.save_path = None
    # args.num_steps = 4
    # args.num_micro_batches_per_batch = 1

    # 训练
    train(
        shard=shard,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=args.learning_rate,
        use_zero_optimizer=args.use_zero_optimizer,
        param_groups=param_groups,
        min_learning_ratio=args.min_learning_ratio,
        learning_rate_num_cycles=args.learning_rate_num_cycles,
        num_micro_batches_per_batch=args.num_micro_batches_per_batch,
        accumulate_steps=args.accumulate_steps,
        max_grad_norm=args.max_grad_norm,
        num_epoches=args.num_epoches,
        num_steps=args.num_steps,
        save_epoch_intervals=args.save_epoch_intervals,
        save_step_intervals=args.save_step_intervals,
        save_path=args.save_path,
        warmup_ratio=args.warmup_ratio,
        wandb_logger=wandb_logger
    )
    # if process_group_manager.global_rank == process_group_manager.world_size - 1:
    #     torch.cuda.memory._dump_snapshot(f"scripts/profile-{process_group_manager.global_rank}.pkl")
    #     torch.cuda.memory._record_memory_history(enabled=None)

    dist.destroy_process_group()
