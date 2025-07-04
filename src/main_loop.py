import torch
import torch.optim as optim
import torch.distributed as dist
import torch.distributed.optim as disted_optim
from tqdm import tqdm
import math
import os
from transformers import get_cosine_schedule_with_warmup
import numpy as np

from .pipeline_parallel import (
    shard_forward, 
    shard_forward_backward_1f1b as shard_forward_backward
)

class SpikeDetector:
    
    def __init__(
        self, 
        min_window_size=50, 
        max_window_size=100,
        std_multiple=4,
        add_value_when_spike_happens=False,
        clear_window_when_spike_happens=False
    ):
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.std_multiple = std_multiple
        self.add_value_when_spike_happens = add_value_when_spike_happens
        self.clear_window_when_spike_happens = clear_window_when_spike_happens
        self.window = []
    
    def _predict_mean_std(self):
        T = len(self.window)
        if T == 1:
            return self.window[0], 0
        else:
            S = sum(self.window)
            M = sum([(j + 1) * v for j, v in enumerate(self.window)])
            b = (2 * (2 * T + 1) * S - 6 * M) / (T * (T - 1))
            k = 2 * (S - b * T) / (T * (T + 1))
            means = k * np.arange(T) + b
            predict_mean = k * (T + 1) + b
            predict_std = np.sqrt(np.power(self.window - means, 2).mean())
            return predict_mean, predict_std

    def is_spike(self, value):
        if len(self.window) < self.min_window_size:
            self.window.append(value)
            return None

        mu, std = self._predict_mean_std()
        if value > mu + self.std_multiple * std:
            if self.clear_window_when_spike_happens:
                self.window.clear()
            if self.add_value_when_spike_happens:
                self.window.append(value)
            return True
        else:
            self.window.append(value)
            if len(self.window) == self.max_window_size + 1:
                self.window.pop(0)
            return False

def move_micro_batch_to_device(micro_batch, device):
    for key, value in micro_batch.items():
        micro_batch[key] = value.to(device)
    return micro_batch

def yield_batch(
    dataloader, 
    num_micro_batches_per_batch,
    device,
    dp_rank=0,
    dp_size=1
):
    batch = []
    for micro_batch in dataloader: # 每一个micro_batch
        micro_batch = move_micro_batch_to_device(micro_batch, device)
        batch.append(micro_batch)
        if len(batch) == num_micro_batches_per_batch * dp_size:
            yield batch[
                num_micro_batches_per_batch * dp_rank: 
                num_micro_batches_per_batch * (dp_rank + 1)
            ]
            batch = []
    # drop last
    # if batch:
    #     yield batch

@torch.no_grad()
def get_total_grad_norm(shard):
    total_norm = 0
    tp_module_substrs = {
        "q_proj", "kv_b_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    } # 有一点问题，就是k_rope_proj也可能是tp的，但是这里不能判断k_rope_proj也是tp的，因此，可能会
    for param_name, param in shard.named_parameters():
        if not param.requires_grad:
            continue
        for tp_module_substr in tp_module_substrs:
            if tp_module_substr in param_name:
                param_norm = param.grad.detach().data.pow(2).sum()
                total_norm += param_norm
                break
        else: # 不是tp参数，每个tp_group的第一个rank只记录一次
            if shard.process_group_manager.global_rank == shard.process_group_manager.tp_first_rank:
                param_norm = param.grad.detach().data.pow(2).sum()
                total_norm += param_norm
    dist.all_reduce(
        tensor=total_norm, 
        op=dist.ReduceOp.SUM,
        group=shard.process_group_manager.tp_pp_group
    )
    total_norm = total_norm.sqrt().item()
    return total_norm

@torch.no_grad()
def clip_shard_grad_norm(
    shard, 
    total_grad_norm,
    max_grad_norm
):
    if total_grad_norm <= max_grad_norm:
        return
    ratio = max_grad_norm / total_grad_norm
    for param in shard.parameters():
        param.grad.data *= ratio

@torch.no_grad()
def gather_loss(
    shard,
    loss
):
    if shard.process_group_manager.dp_world_size > 1:
        loss = torch.tensor(loss, device=shard.device)
        dist.all_reduce(
            tensor=loss,
            op=dist.ReduceOp.AVG,
            group=shard.process_group_manager.dp_group
        )
        return loss.item()
    return loss

def train_epoch(
    shard,
    optimizer,
    lr_scheduler,
    dataloader,
    num_micro_batches_per_batch,
    accumulate_steps,
    spike_detector=None,
    max_grad_norm=10.0,
    not_clip_grad_when_warmup=False
):
    # Compile the optimizer and lr_schedular. But somehow, performance is slower. Ref:
    # https://pytorch.org/tutorials/recipes/compiling_optimizer_lr_scheduler.html
    # @torch.compile(fullgraph=False)
    # def step():
    #     optimizer.step()
    #     lr_scheduler.step()
    
    def step():
        optimizer.step()
        lr_scheduler.step()

    shard.train()
    loss = 0
    for batch_id, batch in enumerate(
        yield_batch(
            dataloader, 
            num_micro_batches_per_batch, 
            shard.device,
            dp_rank=shard.process_group_manager.dp_rank,
            dp_size=shard.process_group_manager.dp_world_size,
        )
    ): # 每一个batch
        batch_step = batch_id + 1
        batch_loss = shard_forward_backward(
            shard=shard, 
            micro_batches=batch, 
            accumulate_steps=accumulate_steps, 
            max_length_force_padded=dataloader.collate_fn.max_length if dataloader.collate_fn.force_padding_to_max_length else None
        ) # 一个batch的loss，pp组只有last_stage才有，每个tp组一样，每个dp组不一样，所以需要对dp组进行平均
        loss += batch_loss if batch_loss is not None else 0
        if batch_step % accumulate_steps == 0:
            loss = gather_loss(shard, loss)
            in_warmup = lr_scheduler._step_count <= lr_scheduler.lr_lambdas[0].keywords['num_warmup_steps']
            grad_norm = get_total_grad_norm(shard)
            if not (not_clip_grad_when_warmup and in_warmup):
                clip_shard_grad_norm(shard, grad_norm, max_grad_norm)
            
            # if spike_detector is not None:
            #     is_spike = spike_detector.is_spike(grad_norm)
            # else: is_spike = False
            # if not is_spike:
            #     optimizer.step()

            step()
            shard.zero_grad() # ZeroRedundancyOptimizer only zero param grads it covers. So use model.zero_grad()

            yield loss, grad_norm
            loss = 0

    # if batch_step % accumulate_steps != 0:
    #     grad_norm = get_total_grad_norm(shard)
    #     is_spike = False
    #     if spike_detector is not None:
    #         is_spike = spike_detector.is_spike(grad_norm)
    #     if not is_spike:
    #         optimizer.step()
    #     # optimizer.zero_grad()
    #     shard.zero_grad()
    #     lr_scheduler.step()
    #     yield loss, grad_norm
    #     loss = 0

@torch.no_grad()
def eval_epoch(
    shard,
    dataloader
):
    shard.eval()
    # TODO: 和dp兼容
    for i, micro_batch in enumerate(dataloader):
        micro_batch = move_micro_batch_to_device(micro_batch, shard.device)
        num_micro_batch_predict_tokens = (micro_batch['labels'] != -100).long().sum()
        _, micro_batch_loss = shard_forward(
            shard=shard,
            micro_batch=micro_batch,
            max_length_force_padded=dataloader.collate_fn.max_length if dataloader.collate_fn.force_padding_to_max_length else None
        )
        yield micro_batch_loss / num_micro_batch_predict_tokens, num_micro_batch_predict_tokens

def train(
    shard,
    train_dataloader,
    eval_dataloader=None,
    learning_rate=2e-5,
    use_zero_optimizer=False,
    param_groups=None,
    min_learning_ratio=None,
    learning_rate_num_cycles=None, # 0.5
    num_micro_batches_per_batch=4,
    accumulate_steps=1,
    max_grad_norm=10.0,
    num_epoches=None,
    num_steps=None,
    save_epoch_intervals=None,
    save_step_intervals=None,
    save_path=None,
    warmup_ratio=0.05,
    wandb_logger=None
):
    train_epoch_steps = math.ceil(math.ceil(len(train_dataloader) / num_micro_batches_per_batch) / accumulate_steps)
    if save_epoch_intervals is not None:
        assert save_step_intervals is None
        save_step_intervals = save_epoch_intervals * train_epoch_steps
    else:
        assert save_step_intervals is not None
    
    if num_epoches is not None:
        num_steps = train_epoch_steps * num_epoches
    else:
        assert num_steps is not None
        num_epoches = math.ceil(num_steps / train_epoch_steps)

    pbar = tqdm(total=num_steps, disable=shard.process_group_manager.global_rank != shard.process_group_manager.world_size - 1)
    
    if use_zero_optimizer:
        if param_groups is None:
            # To use `overlap_with_ddp=True`, ref:
            # https://discuss.pytorch.org/t/how-to-use-torch-distributed-optim-zeroredundancyoptimizer-with-overlap-with-ddp-true/151523
            optimizer = disted_optim.ZeroRedundancyOptimizer(
                shard.parameters(),
                optimizer_class=torch.optim.AdamW,
                lr=learning_rate,
                process_group=shard.process_group_manager.dp_group,
                parameters_as_bucket_view=True
            )
        else:
            # If param_groups is not None, then Should initialize optimizer in the following way:
            # https://discuss.pytorch.org/t/adamw-zeroredundancyoptimizer-weight-decay-dictionary/141516
            # https://github.com/pytorch/pytorch/issues/71347
            optimizer = disted_optim.ZeroRedundancyOptimizer(
                **param_groups[0],
                optimizer_class=optim.AdamW,
                process_group=shard.process_group_manager.dp_group,
                parameters_as_bucket_view=True
            )
            for param_group in param_groups[1:]:
                optimizer.add_param_group(param_group)
    else:
        if param_groups is None:
            optimizer = optim.AdamW(
                shard.parameters(),
                lr=torch.tensor(learning_rate),
                fused=True
            )
        else:
            optimizer = optim.AdamW(
                param_groups, 
                fused=True
            )

    if learning_rate_num_cycles is None:
        learning_rate_num_cycles = math.acos(2 * min_learning_ratio - 1) / (2 * math.pi)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=int(warmup_ratio * num_steps), 
        num_training_steps=num_steps,
        num_cycles=learning_rate_num_cycles
    )
    
    # spike_detector = SpikeDetector()
    spike_detector = None
    global_step = 0
    for epoch in range(1, num_epoches + 1):
        # train
        for id_, (loss, grad_norm) in enumerate(train_epoch(
            shard=shard,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=train_dataloader,
            num_micro_batches_per_batch=num_micro_batches_per_batch,
            accumulate_steps=accumulate_steps,
            spike_detector=spike_detector,
            max_grad_norm=max_grad_norm,
            not_clip_grad_when_warmup=False
        )):
            pbar.update(1)
            local_step = id_ + 1
            global_step += 1

            if shard.process_group_manager.global_rank == shard.process_group_manager.world_size - 1:
                print(f'[Train] epoch: {epoch}, local_step: {local_step}, global_step: {global_step}, loss: {loss}')
                if wandb_logger is not None:
                    lrs = lr_scheduler.get_lr()
                    logging_info = {
                        "train_loss": loss, 
                        "grad_norm": grad_norm,
                        "step": global_step
                    }
                    for j, lr in enumerate(lrs):
                        logging_info[f'lr_{j}'] = lr
                    wandb_logger.log(logging_info)
            
            if shard.process_group_manager.dp_rank == 0:
                if save_path is not None and (global_step % save_step_intervals == 0 or global_step == num_steps):
                    ckpt_path = os.path.join(save_path, f'step-{global_step}/tp-{shard.process_group_manager.tp_rank}_pp-{shard.process_group_manager.pp_rank}')
                    try:
                        os.makedirs(ckpt_path)
                    except Exception:
                        pass
                    if shard.process_group_manager.dp_world_size == 1:
                        shard.save_pretrained(ckpt_path)
                    else:
                        shard.module.save_pretrained(ckpt_path)
                    print(f'[Save] to {ckpt_path}')
            
            if global_step == num_steps:
                break
        
        if global_step == num_steps:
            break

        # eval
        if eval_dataloader is not None:
            # eval_pbar = tqdm(total=len(eval_dataloader), disable=not shard.process_group_manager.pp_is_last_stage)

            total_loss = 0
            total_num_predict_tokens = 0
            for micro_batch_loss, num_micro_batch_predict_tokens in eval_epoch(shard, eval_dataloader):
                if shard.process_group_manager.global_rank == shard.process_group_manager.world_size - 1:
                    total_loss += micro_batch_loss.item() * num_micro_batch_predict_tokens.item()
                    total_num_predict_tokens += num_micro_batch_predict_tokens.item()
                    # loss = (loss * id_ + micro_batch_loss.item()) / (id_ + 1)
            
            if shard.process_group_manager.global_rank == shard.process_group_manager.world_size - 1:
                loss = total_loss / total_num_predict_tokens
                print(f'[Eval] epoch: {epoch}, loss: {loss}')
                if wandb_logger is not None:
                    wandb_logger.log({"eval_loss": loss})
    
    if wandb_logger is not None:
        wandb_logger.finish()
