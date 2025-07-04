import torch
import torch.distributed as dist
from .all_forward_all_backward import (
    shard_forward, 
    shard_backward,
    recv_input_for_forward,
    send_output_for_forward,
    recv_output_grad_for_backward,
    send_input_grad_for_backward,
    _shard_forward,
    send_3D_tensor,
    recv_3D_tensor
)

def send_output_for_forward_and_recv_output_grad_for_backward(
    shard,
    output_to_send_for_forward, # send_output_for_forward
    output_recv_grad_for_backward, # recv_output_grad_for_backward
):
    output_grad = None
    ops = []
    if not shard.process_group_manager.pp_is_last_stage:
        _, send_op = send_3D_tensor(
            tensor=output_to_send_for_forward, 
            dst_rank=shard.process_group_manager.pp_next_rank,
            return_op=True
        )
        ops.append(send_op)
    if not shard.process_group_manager.pp_is_last_stage:
        output_grad, recv_op = recv_3D_tensor(
            recv_shape=output_recv_grad_for_backward.shape,
            src_rank=shard.process_group_manager.pp_next_rank,
            dtype=shard.dtype, 
            device=shard.device,
            requires_grad=False,
            return_op=True
        )
        ops.append(recv_op)
    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        [req.wait() for req in reqs]
    # torch.cuda.synchronize()
    return output_grad

def recv_input_for_forward_and_send_input_grad_for_backward(
    shard,
    micro_batch_for_forward,
    input_with_grad_to_send_for_backward
):
    ops = []
    micro_batch_inputs = dict()
    micro_batch_inputs.update(micro_batch_for_forward)

    if shard.process_group_manager.pp_world_size > 1:
        if not shard.process_group_manager.pp_is_first_stage: # middle, last pp_rank
            hidden_states, recv_op = recv_3D_tensor(
                recv_shape=list(micro_batch_inputs["input_ids"].shape) + [shard.config.hidden_size],
                src_rank=shard.process_group_manager.pp_prev_rank,
                dtype=shard.dtype, 
                device=shard.device,
                requires_grad=shard.training,
                return_op=True
            )
            ops.append(recv_op)
    
    if not shard.process_group_manager.pp_is_first_stage:
        _, send_op = send_3D_tensor(
            tensor=input_with_grad_to_send_for_backward.grad,
            dst_rank=shard.process_group_manager.pp_prev_rank,
            return_op=True
        )
        ops.append(send_op)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        [req.wait() for req in reqs]
    # torch.cuda.synchronize()

    if shard.process_group_manager.pp_world_size > 1:
        if not shard.process_group_manager.pp_is_first_stage: # middle, last pp_rank
            micro_batch_inputs["input_ids"] = None
            micro_batch_inputs["inputs_embeds"] = hidden_states
        if not shard.process_group_manager.pp_is_last_stage:
            micro_batch_inputs["labels"] = None
    
    return micro_batch_inputs

def shard_forward_backward(
    shard, 
    micro_batches, 
    accumulate_steps=1,
    max_length_force_padded=None
):
    """
        forward and backward on one batch, i.e., num_micro_batch_per_batch * micro_batches.
        Args:
            shard: Model shard
            micro_batches: list of dict, each consisting `input_ids`, `attention_mask`, `labels`
            accumulate_steps:
    """
    # loss_discounts = []
    # for micro_batch in micro_batches:
    #     num_micro_batch_predict_tokens = (micro_batch['labels'] != -100).long().sum().item()
    #     loss_discounts.append(num_micro_batch_predict_tokens)
    # num_batch_predict_tokens = sum(loss_discounts)
    # loss_discounts = [l / num_batch_predict_tokens / accumulate_steps for l in loss_discounts]

    with torch.no_grad():
        loss_discounts = torch.zeros(len(micro_batches), device=shard.device)
        for i, micro_batch in enumerate(micro_batches):
            num_micro_batch_predict_tokens = (micro_batch['labels'] != -100).long().sum()
            loss_discounts[i] = num_micro_batch_predict_tokens
        loss_discounts /= loss_discounts.sum() * accumulate_steps

    assert len(micro_batches) >= shard.process_group_manager.pp_world_size
    startup_num_micro_batches = shard.process_group_manager.pp_world_size - shard.process_group_manager.pp_rank - 1

    if shard.process_group_manager.pp_is_last_stage:
        loss = 0

    # startup
    input_output_pairs = []
    for i in range(startup_num_micro_batches):
        pair = shard_forward(
            shard=shard,
            micro_batch=micro_batches[i],
            loss_discount=loss_discounts[i],
        )
        input_output_pairs.append(pair)
    
    if shard.process_group_manager.dp_world_size > 1:
        shard.require_backward_grad_sync = False
    # steady
    for i in range(startup_num_micro_batches, len(micro_batches)):

        # 通信：当前的输入（带梯度）传往上一个pp_stage，同时从上一个pp_stage接受输入
        if i == startup_num_micro_batches: # 如果是第一个，则只从上一个pp_stage接受输入
            micro_batch_inputs = recv_input_for_forward(
                shard=shard,
                micro_batch=micro_batches[i]
            )
        else:
            micro_batch_inputs = recv_input_for_forward_and_send_input_grad_for_backward(
                shard=shard,
                micro_batch_for_forward=micro_batches[i],
                input_with_grad_to_send_for_backward=input_to_backward
            )

        # 1f
        input_this_step, output_this_step = _shard_forward(
            shard=shard,
            micro_batch_inputs=micro_batch_inputs,
            loss_discount=loss_discounts[i]
        )
        if shard.process_group_manager.pp_is_last_stage:
            loss += output_this_step.item()

        input_output_pairs.append((input_this_step, output_this_step))
        input_to_backward, output_to_backward = input_output_pairs.pop(0)

        # 通信：当前的输出传往下一个pp_stage，同时从下一个pp_stage接受需要backward的输出的grad
        output_to_backward_grad = send_output_for_forward_and_recv_output_grad_for_backward(
            shard=shard,
            output_to_send_for_forward=output_this_step,
            output_recv_grad_for_backward=output_to_backward
        )

        # 最后一个rank的最后一个micro_batch的backward需要把梯度同步打开
        if shard.process_group_manager.dp_world_size > 1 \
            and startup_num_micro_batches == 0 \
            and i == len(micro_batches) - 1:
            shard.require_backward_grad_sync = True

        # 1b: input_to_backward 会有梯度
        if output_to_backward_grad is not None:
            torch.autograd.backward(output_to_backward, output_to_backward_grad)
        else:
            output_to_backward.backward()

        if i == len(micro_batches) - 1: # 还需要把带梯度的输入传到上一个pp_stage
            send_input_grad_for_backward(
                shard=shard,
                input_with_grad=input_to_backward
            )


    # cooldown
    assert len(input_output_pairs) == startup_num_micro_batches
    for i, pair in enumerate(input_output_pairs):
        shard_backward(
            shard=shard,
            input_output_pair=pair,
            is_last_iter=i == startup_num_micro_batches - 1
        )
    
    if shard.process_group_manager.pp_is_last_stage:
        return loss
