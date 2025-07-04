import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class Copy(torch.autograd.Function):
    """copy in the first step of column-wise parallel linear"""
    @staticmethod
    def forward(ctx, input_, process_group_manager):
        ctx.process_group_manager = process_group_manager
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.process_group_manager.tp_world_size == 1:
            return grad_output, None
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.process_group_manager.tp_group)
        return grad_output, None

class AllGather(torch.autograd.Function):
    """All-gather in the last step of column-wise parallel linear"""
    # @staticmethod
    # def forward(ctx, input_, process_group_manager):
    #     ctx.process_group_manager = process_group_manager
    #     if ctx.process_group_manager.tp_world_size == 1:
    #         return input_
    #     input_ = input_.contiguous()
    #     tensor_list = [torch.empty_like(input_) for _ in range(ctx.process_group_manager.tp_world_size)]
    #     tensor_list[ctx.process_group_manager.tp_rank] = input_
    #     dist.all_gather(tensor_list, input_, group=ctx.process_group_manager.tp_group)
    #     output = torch.cat(tensor_list, -1).contiguous()
    #     return output

    @staticmethod
    def forward(ctx, input_, process_group_manager, async_op=False):
        ctx.process_group_manager = process_group_manager
        if ctx.process_group_manager.tp_world_size == 1:
            return input_
        input_ = input_.contiguous()
        input_shape = list(input_.shape)
        input_shape[-1] *= process_group_manager.tp_world_size
        output = torch.empty(input_shape, dtype=input_.dtype, device=input_.device)
        handle = dist.all_gather_into_tensor(output, input_, group=ctx.process_group_manager.tp_group, async_op=async_op)
        if async_op:
            return handle.get_future().then(lambda f: f.wait()[0])
        return output.contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.process_group_manager.tp_world_size == 1:
            return grad_output, None
        assert grad_output.shape[-1] % ctx.process_group_manager.tp_world_size == 0
        dim_per_tp_stage = grad_output.shape[-1] // ctx.process_group_manager.tp_world_size
        tensor_list = torch.split(grad_output, dim_per_tp_stage, -1)
        grad_input = tensor_list[ctx.process_group_manager.tp_rank].contiguous()
        return grad_input, None, None

class AllReduceSum(torch.autograd.Function):
    """All-reduce in the last step of row-wise parallel linear"""
    @staticmethod
    def forward(ctx, input_, process_group_manager, async_op=False):
        ctx.process_group_manager = process_group_manager
        if ctx.process_group_manager.tp_world_size == 1:
            return input_
        handle = dist.all_reduce(input_, op=dist.ReduceOp.SUM, group=ctx.process_group_manager.tp_group, async_op=async_op)
        if async_op:
            return handle.get_future().then(lambda f: f.wait()[0])
        return input_.contiguous()
            
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

# class AllReduceMax(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_, process_group_manager, async_op=False):
#         ctx.process_group_manager = process_group_manager
#         if ctx.process_group_manager.tp_world_size == 1:
#             return input_
#         output = input_.clone()
#         handle = dist.all_reduce(output, op=dist.ReduceOp.MAX, group=ctx.process_group_manager.tp_group, async_op=async_op)
#         ctx.save_for_backward(input_ == output)
#         return output, handle
    
#     @staticmethod
#     def backward(ctx, grad_output, grad_handle_placeholder):
#         max_mask = ctx.saved_tensors[0]
#         return max_mask * grad_output, None, None

class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        *args,
        process_group_manager=None,
        gather_output=False,
        async_op=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.process_group_manager = process_group_manager
        self.gather_output = gather_output
        self.async_op = async_op
        

    @property
    def distributed_inited(self):
        return self.process_group_manager is not None

    @classmethod
    def from_linear(cls, linear, process_group_manager, gather_output=False, async_op=False):
        self = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=next(linear.parameters()).device,
            dtype=next(linear.parameters()).dtype,
            process_group_manager=process_group_manager,
            gather_output=gather_output,
            async_op=async_op
        )
        self.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            self.bias.data.copy_(linear.bias.data)

        return self
    
    def to_linear(self):
        linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype
        )
        linear.weight.data.copy_(self.weight.data)
        if self.bias is not None:
            linear.bias.data.copy_(self.bias.data)
        return linear


    def distributed_init(self, process_group_manager, gather_output=False):
        raise NotImplementedError()

    def forward(self, input_):
        if not self.distributed_inited:
            raise Exception("Not init yet! Please call .distributed_init method first.")
        input_ = Copy.apply(input_, self.process_group_manager)
        output = F.linear(input_, self.weight, self.bias)
        if self.gather_output:
            output = AllGather.apply(output, self.process_group_manager, self.async_op)
        return output

class RowParallelLinear(nn.Linear):
    """Sharding the input dim, i.e., the second dim of self.weight."""

    def __init__(
        self,
        *args,
        process_group_manager=None,
        async_op=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.process_group_manager = process_group_manager
        self.async_op = async_op

    @property
    def distributed_inited(self):
        return self.process_group_manager is not None

    @classmethod
    def from_linear(cls, linear, process_group_manager, async_op=False):
        self = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=next(linear.parameters()).device,
            dtype=next(linear.parameters()).dtype,
            process_group_manager=process_group_manager,
            async_op=async_op
        )
        self.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            self.bias.data.copy_(linear.bias.data)

        return self
    
    def to_linear(self):
        linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype
        )
        linear.weight.data.copy_(self.weight.data)
        if self.bias is not None:
            linear.bias.data.copy_(self.bias.data)
        return linear

    def distributed_init(self, process_group_manager):
        raise NotImplementedError()
    
    def forward(self, input_):
        if not self.distributed_inited:
            raise Exception("Not sharded yet! Please call .distributed_inited method first.")
        output = F.linear(input_, self.weight)
        output = AllReduceSum.apply(output, self.process_group_manager, self.async_op)
        if self.bias is not None:
            if self.async_op:
                output = output.then(lambda f: f.wait()[0])
                return output.then(lambda f: self.bias + f.wait())
            return output + self.bias
        return output

class EmbeddingParallel(nn.Embedding):

    def __init__(
        self,
        *args,
        process_group_manager=None,
        async_op=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.process_group_manager = process_group_manager
        self.async_op = async_op
        self.vocab_offset = process_group_manager.tp_rank * self.num_embeddings

    @property
    def distributed_inited(self):
        return self.process_group_manager is not None

    @classmethod
    def from_embedding(cls, embedding, process_group_manager, async_op=False):
        self = cls(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
            max_norm=embedding.max_norm,
            norm_type=embedding.norm_type,
            scale_grad_by_freq=embedding.scale_grad_by_freq,
            sparse=embedding.sparse,
            _weight=embedding.weight.data,
            _freeze=not embedding.weight.requires_grad,
            device=next(embedding.parameters()).device,
            dtype=next(embedding.parameters()).dtype,
            process_group_manager=process_group_manager,
            async_op=async_op
        )
        self.vocab_offset = process_group_manager.tp_rank * embedding.num_embeddings
        self.process_group_manager = process_group_manager
        self.async_op = async_op

        return self
    
    def to_embedding(self):
        embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
            _weight=self.weight.data,
            _freeze=not self.weight.requires_grad,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype
        )
        return embedding

    def distributed_init(self, process_group_manager):
        raise NotImplementedError()
    
    def forward(self, input_):
        if not self.distributed_inited:
            raise Exception("Not sharded yet! Please call .distributed_init method first.")
        input_ -= self.vocab_offset
        vocab_mask = (input_ >= 0) & \
                     (input_ < self.num_embeddings) # (batch_size, seq_length)
        input_[~vocab_mask] = 0
        output = super().forward(input_) # (batch_size, seq_length, embedding_dim)
        output *= vocab_mask.unsqueeze(-1) # (batch_size, seq_length, embedding_dim)
        output = AllReduceSum.apply(output, self.process_group_manager, self.async_op)
        return output

# class VocabParallelCrossEntropy(nn.Module):

#     def __init__(self, process_group_manager, async_op=False):
#         super().__init__()
#         self.process_group_manager = process_group_manager
#         self.async_op = async_op

#     def forward(self, logits, labels, *args, **kwargs):
#         logits = logits[:, :-1]
#         labels = labels[:, 1:]

#         target_word_mask = (labels != -100) # (bsz, seq_len - 1)
#         _, _, tp_vocab_size = logits.shape
#         vocab_offset = tp_vocab_size * self.process_group_manager.tp_rank

#         # Upcast to float if we need to compute the loss to avoid potential precision issues
#         # logits = logits.float()
#         max_logits = logits.max(-1).values.detach() # (bsz, seq_len - 1)
#         handle = dist.all_reduce(
#             max_logits, 
#             op=dist.ReduceOp.MAX, 
#             group=self.process_group_manager.tp_group
#         )
#         logits = logits - max_logits[:, :, None] # (bsz, seq_len - 1, tp_vocab_size)
#         denominator = logits.exp().sum(-1) # (bsz, seq_len - 1)
#         denominator = AllReduceSum.apply(denominator, self.process_group_manager) # (bsz, seq_len - 1)
        
#         labels = labels - vocab_offset
#         tp_word_mask = (labels >= 0) & \
#                      (labels < tp_vocab_size) # (batch_size, seq_len - 1)
#         # labels[~tp_word_mask] = 0
#         labels = labels * tp_word_mask

#         logits = torch.gather(
#             logits, -1, labels.unsqueeze(-1),
#         ).squeeze(-1) # (bsz, seq_len - 1)
#         # logits[~tp_word_mask] = 0 # (bsz, seq_len - 1)
#         logits = logits * tp_word_mask

#         logits = AllReduceSum.apply(logits, self.process_group_manager)
#         loss = denominator.log() - logits # (bsz, seq_len - 1)

#         loss = (target_word_mask * loss).sum() / target_word_mask.sum()
#         return loss

class VocabParallelCrossEntropy(nn.Module):

    def __init__(self, process_group_manager=None, async_op=True):
        super().__init__()
        self.process_group_manager = process_group_manager
        self.async_op = async_op

    def forward(self, logits, labels,  *args, **kwargs):
        if self.process_group_manager.tp_world_size == 1:
            return self.forward_single(logits, labels, *args, **kwargs)
        return self.forward_tp(logits, labels, *args, **kwargs)

    def forward_single(self, logits, labels,  *args, **kwargs):
        loss = F.cross_entropy(
            input=logits[:, :-1].flatten(0, 1),
            target=labels[:, 1:].flatten(0, 1),
            ignore_index=-100
        )
        return loss

    def forward_tp(self, logits, labels, *args, **kwargs):
        logits = logits[:, :-1] # (bsz, seq_len - 1, tp_vocab_size)
        labels = labels[:, 1:]

        # Upcast to float if we need to compute the loss to avoid potential precision issues
        # logits = logits.float()
        max_logits = logits.max(-1).values.detach() # (bsz, seq_len - 1)
        # start to communicate to get max_logits
        handle = dist.all_reduce(
            max_logits, 
            op=dist.ReduceOp.MAX, 
            group=self.process_group_manager.tp_group,
            async_op=self.async_op
        )

        target_word_mask = (labels != -100) # (bsz, seq_len - 1)
        _, _, tp_vocab_size = logits.shape
        vocab_offset = tp_vocab_size * self.process_group_manager.tp_rank
        labels = labels - vocab_offset
        tp_word_mask = (labels >= 0) & \
                     (labels < tp_vocab_size) # (batch_size, seq_len - 1)
        # labels[~tp_word_mask] = 0
        labels = labels * tp_word_mask

        # max_logits should be done!
        if self.async_op:
            max_logits = handle.get_future().wait()[0]
        logits = logits - max_logits[:, :, None] # (bsz, seq_len - 1, tp_vocab_size)
        denominator = logits.exp().sum(-1) # (bsz, seq_len - 1)
        # start to communicate to get denominator
        denominator = AllReduceSum.apply(
            denominator, 
            self.process_group_manager,
            self.async_op
        ) # (bsz, seq_len - 1)

        logits = torch.gather(
            logits, -1, labels.unsqueeze(-1),
        ).squeeze(-1) # (bsz, seq_len - 1)
        # logits[~tp_word_mask] = 0 # (bsz, seq_len - 1)
        logits = logits * tp_word_mask
        # start to communicate to get logits
        logits = AllReduceSum.apply(
            logits, 
            self.process_group_manager,
            self.async_op
        )

        # denominator should be done!
        if self.async_op:
            denominator = denominator.wait()[0]
            logits = logits.wait()[0]
        
        loss = denominator.log() - logits # (bsz, seq_len - 1)
        loss = (target_word_mask * loss).sum() / target_word_mask.sum()
        return loss
