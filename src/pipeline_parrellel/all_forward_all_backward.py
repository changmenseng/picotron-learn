import torch
import torch.distributed as dist
import warnings

# def send_3D_tensor(
#     tensor, 
#     dst_rank,
#     send_shape=True
# ):
#     if send_shape:
#         dist.send(torch.tensor(tensor.shape).to(tensor.device), dst=dst_rank)
#     dist.send(tensor, dst=dst_rank)

def send_3D_tensor(
    tensor,
    dst_rank,
    return_op=False
):
    if return_op:
        return None, dist.P2POp(dist.isend, tensor, dst_rank)
    dist.send(tensor, dst=dst_rank)

# def recv_3D_tensor(
#     src_rank,
#     recv_shape=None, # list
#     **tensor_kwargs
# ):
#     if recv_shape is None:
#         recv_shape = torch.zeros(3, dtype=torch.long, device=tensor_kwargs.get('device', None))
#         dist.recv(recv_shape, src=src_rank)
#         recv_shape = recv_shape.tolist()
#     tensor = torch.empty(*recv_shape, **tensor_kwargs)
#     dist.recv(tensor, src=src_rank)
#     return tensor

def recv_3D_tensor(
    recv_shape, # list
    src_rank,
    return_op=False,
    **tensor_kwargs
):
    tensor = torch.empty(*recv_shape, **tensor_kwargs)
    if return_op:
        op = dist.P2POp(dist.irecv, tensor, src_rank)
        return tensor, op
    dist.recv(tensor, src=src_rank)
    return tensor

def recv_input_for_forward(
    shard,
    micro_batch
):
    micro_batch_inputs = dict()
    micro_batch_inputs.update(micro_batch)
    if shard.process_group_manager.pp_world_size > 1:
        if not shard.process_group_manager.pp_is_first_stage: # middle, last pp_rank
            hidden_states = recv_3D_tensor(
                recv_shape=list(micro_batch_inputs["input_ids"].shape) + [shard.config.hidden_size],
                src_rank=shard.process_group_manager.pp_prev_rank,
                dtype=shard.dtype, 
                device=shard.device,
                requires_grad=shard.training
            )
            micro_batch_inputs["input_ids"] = None
            micro_batch_inputs["inputs_embeds"] = hidden_states
        if not shard.process_group_manager.pp_is_last_stage:
            micro_batch_inputs["labels"] = None
    return micro_batch_inputs

def send_output_for_forward(
    shard,
    output
):
    if not shard.process_group_manager.pp_is_last_stage:
        send_3D_tensor(
            tensor=output, 
            dst_rank=shard.process_group_manager.pp_next_rank
        )

def _shard_forward(
    shard,
    micro_batch_inputs,
    loss_discount=1.0,
):
    outputs = shard(
        **micro_batch_inputs,
        output_hidden_states=not shard.process_group_manager.pp_is_last_stage,
        use_cache=False
    )
    if shard.process_group_manager.pp_is_last_stage:
        try:
            return micro_batch_inputs.get("inputs_embeds", None), outputs.loss * loss_discount
        except Exception as e:
            print(outputs)
            print(outputs.keys())
            print(micro_batch_inputs.keys())
            raise e
    return micro_batch_inputs.get("inputs_embeds", None), outputs.hidden_states[-1].to(shard.dtype)

def shard_forward(
    shard,
    micro_batch,
    loss_discount=1.0,
):
    """
        if pp:
            first pp_rank:
                recv: None
                input: input_ids, attention_mask, output_hidden_states=True
                output: hidden_states
                send: hidden_states
            middle pp_rank:
                recv: hidden_states
                input: inputs_embeds, attention_mask, output_hidden_states=False
                output: hidden_states
                send: hidden_states
            last pp_rank:
                recv: hidden_states
                input: inputs_embeds, attention_mask, labels, output_hidden_states=False
                output: loss
        else:
            input: input_ids, attention_mask, labels, output_hidden_states=False
    """
    micro_batch_inputs = recv_input_for_forward(
        shard=shard,
        micro_batch=micro_batch,
    )

    input_, output = _shard_forward(
        shard=shard,
        micro_batch_inputs=micro_batch_inputs,
        loss_discount=loss_discount
    )

    send_output_for_forward(
        shard=shard,
        output=output,
    )
    
    return input_, output

def recv_output_grad_for_backward(
    shard,
    output,
):
    if not shard.process_group_manager.pp_is_last_stage:
        output_grad = recv_3D_tensor(
            recv_shape=output.shape,
            src_rank=shard.process_group_manager.pp_next_rank,
            dtype=shard.dtype, 
            device=shard.device,
            requires_grad=False
        )
        return output_grad

def send_input_grad_for_backward(
    shard,
    input_with_grad,
):
    if not shard.process_group_manager.pp_is_first_stage:
        send_3D_tensor(
            tensor=input_with_grad.grad,
            dst_rank=shard.process_group_manager.pp_prev_rank
        )

def shard_backward(
    shard,
    input_output_pair,
    is_last_iter
):
    input_, output = input_output_pair

    output_grad = recv_output_grad_for_backward(
        shard=shard,
        output=output,
    )
    
    if shard.process_group_manager.dp_world_size > 1:
        shard.require_backward_grad_sync = is_last_iter
    
    if output_grad is not None:
        torch.autograd.backward(output, output_grad)
    else:
        output.backward()
    
    send_input_grad_for_backward(
        shard=shard,
        input_with_grad=input_
    )

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
    with torch.no_grad():
        loss_discounts = torch.zeros(len(micro_batches), device=shard.device)
        for i, micro_batch in enumerate(micro_batches):
            num_micro_batch_predict_tokens = (micro_batch['labels'] != -100).long().sum()
            loss_discounts[i] = num_micro_batch_predict_tokens
        loss_discounts /= loss_discounts.sum() * accumulate_steps
        
    # forward
    input_output_pairs = []
    for micro_batch, loss_discount in zip(micro_batches, loss_discounts):
        pair = shard_forward(
            shard=shard,
            micro_batch=micro_batch,
            loss_discount=loss_discount,
        )
        input_output_pairs.append(pair)
    
    # backward
    for i, pair in enumerate(input_output_pairs):
        shard_backward(
            shard=shard,
            input_output_pair=pair,
            is_last_iter=i == len(input_output_pairs) - 1
        )

    if shard.process_group_manager.pp_is_last_stage:
        with torch.no_grad():
            losses = [pair[1] for pair in input_output_pairs]
            loss = sum(losses).item()
            return loss
