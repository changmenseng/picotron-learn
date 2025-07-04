import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Config
from typing import Optional
from tqdm import tqdm
import copy
import warnings

from .tensor_parallel import (
    ColumnParallelLinear, 
    RowParallelLinear, 
    EmbeddingParallel
)

from .utils import (
    linear_column_slice,
    linear_row_slice,
    embedding_slice,
    linear_complete_to_shard_column_copy,
    linear_complete_to_shard_row_copy,
    embedding_complete_to_shard_copy,
    linear_shard_to_complete_column_copy,
    linear_shard_to_complete_row_copy,
    embedding_shard_to_complete_copy
)

class Qwen2ForCausalLMShardConfig(Qwen2Config):
    model_type = "qwen2_for_causal_lm_shard"

    def __init__(
        self,
        tp_rank=None,
        tp_size=None,
        pp_rank=None,
        pp_size=None,
        **kwargs
    ):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        super().__init__(**kwargs)

class Qwen2ForCausalLMDistributedShard(Qwen2ForCausalLM):
    config_class = Qwen2ForCausalLMShardConfig
    """
        The shard class of Qwen2ForCausalLM. You must use this class as follows. 
        Say you want to split the model into 8 parts, with tp_size=4, and pp_size=2. 
        You should enumerate tp_size and pp_size, and instantiate this class using
        the config:
    """

    def __init__(self, config):
        self.process_group_manager = None
        
        # config中的元素需要是完整config
        assert config.hidden_size % config.tp_size == 0
        assert config.intermediate_size % config.tp_size == 0
        assert config.num_attention_heads % config.tp_size == 0
        assert config.num_key_value_heads % config.tp_size == 0
        assert config.num_hidden_layers % config.pp_size == 0

        init_config = copy.deepcopy(config)

        if init_config.pp_rank != 0 and init_config.pp_rank != init_config.pp_size == -1: # 不用初始化embedding
            init_config.vocab_size = 1
            init_config.bos_token_id = 0
            init_config.eos_token_id = 0
        
        init_config.num_hidden_layers //= init_config.pp_size

        super().__init__(init_config)
        self.config = config
        if self.config.tie_word_embeddings:
            warnings.warn("`tie_word_embeddings` is True, but we have to set it to False")
            self.config.tie_word_embeddings = False

        # 将模型剪枝，使得其参数变为原来的1 / (tp_size * pp_size)
        # 但：暂时不要将nn.Linear改成ColumnLinear或RowLinear
        self._prune_pp()
        self._prune_tp()

    def _prune_pp(self):
        if self.config.pp_rank != 0:
            self.model.embed_tokens = nn.Identity()
        
        if self.config.pp_rank != self.config.pp_size - 1:
            self.model.norm = nn.Identity()
            self.lm_head = nn.Identity()
    
    def _prune_tp(self):

        for layer in self.model.layers:
            layer.self_attn.q_proj = linear_column_slice(
                layer.self_attn.q_proj, 
                self.config.tp_rank, 
                self.config.tp_size
            )
            layer.self_attn.k_proj = linear_column_slice(
                layer.self_attn.k_proj, 
                self.config.tp_rank, 
                self.config.tp_size
            )
            layer.self_attn.v_proj = linear_column_slice(
                layer.self_attn.v_proj, 
                self.config.tp_rank, 
                self.config.tp_size
            )
            layer.self_attn.o_proj = linear_row_slice(
                layer.self_attn.o_proj, 
                self.config.tp_rank, 
                self.config.tp_size
            )
            layer.mlp.gate_proj = linear_column_slice(
                layer.mlp.gate_proj, 
                self.config.tp_rank, 
                self.config.tp_size
            )
            layer.mlp.up_proj = linear_column_slice(
                layer.mlp.up_proj, 
                self.config.tp_rank, 
                self.config.tp_size
            )
            layer.mlp.down_proj = linear_row_slice(
                layer.mlp.down_proj, 
                self.config.tp_rank, 
                self.config.tp_size
            )
        
        if self.config.pp_rank == 0:
            self.model.embed_tokens = embedding_slice(
                self.model.embed_tokens, 
                self.config.tp_rank, 
                self.config.tp_size
            )
        
        if self.config.pp_rank == self.config.pp_size - 1:
            self.lm_head = linear_column_slice(
                self.lm_head, 
                self.config.tp_rank, 
                self.config.tp_size
            )

    def copy_from_complete_model(self, complete_model):

        complete_model_layer_offset = self.config.pp_rank * self.config.num_hidden_layers // self.config.pp_size

        for i, layer in enumerate(self.model.layers):
            complete_model_layer = complete_model.model.layers[i + complete_model_layer_offset]
            linear_complete_to_shard_column_copy(
                layer.self_attn.q_proj,
                complete_model_layer.self_attn.q_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_complete_to_shard_column_copy(
                layer.self_attn.k_proj,
                complete_model_layer.self_attn.k_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_complete_to_shard_column_copy(
                layer.self_attn.v_proj,
                complete_model_layer.self_attn.v_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_complete_to_shard_row_copy(
                layer.self_attn.o_proj,
                complete_model_layer.self_attn.o_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_complete_to_shard_column_copy(
                layer.mlp.gate_proj,
                complete_model_layer.mlp.gate_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_complete_to_shard_column_copy(
                layer.mlp.up_proj,
                complete_model_layer.mlp.up_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_complete_to_shard_row_copy(
                layer.mlp.down_proj,
                complete_model_layer.mlp.down_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            layer.input_layernorm.weight.data.copy_(
                complete_model_layer.input_layernorm.weight.data
            )
            layer.post_attention_layernorm.weight.data.copy_(
                complete_model_layer.post_attention_layernorm.weight.data
            )

        if self.config.pp_rank == 0:
            embedding_complete_to_shard_copy(
                self.model.embed_tokens,
                complete_model.model.embed_tokens,
                self.config.tp_rank, 
                self.config.tp_size
            )
        if self.config.pp_rank == self.config.pp_size - 1:
            linear_complete_to_shard_column_copy(
                self.lm_head,
                complete_model.lm_head,
                self.config.tp_rank, 
                self.config.tp_size
            )
            self.model.norm.weight.data.copy_(complete_model.model.norm.weight.data)
    
    def copy_to_complete_model(self, complete_model):
        complete_model_layer_offset = self.config.pp_rank * self.config.num_hidden_layers // self.config.pp_size

        for i, layer in enumerate(self.model.layers):
            complete_model_layer = complete_model.model.layers[i + complete_model_layer_offset]
            linear_shard_to_complete_column_copy(
                layer.self_attn.q_proj,
                complete_model_layer.self_attn.q_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_shard_to_complete_column_copy(
                layer.self_attn.k_proj,
                complete_model_layer.self_attn.k_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_shard_to_complete_column_copy(
                layer.self_attn.v_proj,
                complete_model_layer.self_attn.v_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_shard_to_complete_row_copy(
                layer.self_attn.o_proj,
                complete_model_layer.self_attn.o_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_shard_to_complete_column_copy(
                layer.mlp.gate_proj,
                complete_model_layer.mlp.gate_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_shard_to_complete_column_copy(
                layer.mlp.up_proj,
                complete_model_layer.mlp.up_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            linear_shard_to_complete_row_copy(
                layer.mlp.down_proj,
                complete_model_layer.mlp.down_proj,
                self.config.tp_rank,
                self.config.tp_size
            )
            complete_model_layer.input_layernorm.weight.data.copy_(
                layer.input_layernorm.weight.data
            )
            complete_model_layer.post_attention_layernorm.weight.data.copy_(
                layer.post_attention_layernorm.weight.data
            )

        if self.config.pp_rank == 0:
            embedding_shard_to_complete_copy(
                self.model.embed_tokens,
                complete_model.model.embed_tokens,
                self.config.tp_rank, 
                self.config.tp_size
            )
        if self.config.pp_rank == self.config.pp_size - 1:
            linear_shard_to_complete_column_copy(
                self.lm_head,
                complete_model.lm_head,
                self.config.tp_rank, 
                self.config.tp_size
            )
            complete_model.model.norm.weight.data.copy_(self.model.norm.weight.data)

    @property
    def distributed_inited(self):
        return self.process_group_manager is not None
        
    def distributed_init(self, process_group_manager):
        assert self.config.tp_rank == process_group_manager.tp_rank
        assert self.config.tp_size == process_group_manager.tp_world_size
        assert self.config.pp_rank == process_group_manager.pp_rank
        assert self.config.pp_size == process_group_manager.pp_world_size

        self.process_group_manager = process_group_manager
        
        for layer in self.model.layers:
            layer.self_attn.q_proj = ColumnParallelLinear.from_linear(layer.self_attn.q_proj, process_group_manager)
            layer.self_attn.k_proj = ColumnParallelLinear.from_linear(layer.self_attn.k_proj, process_group_manager)
            layer.self_attn.v_proj = ColumnParallelLinear.from_linear(layer.self_attn.v_proj, process_group_manager)
            layer.self_attn.o_proj = RowParallelLinear.from_linear(layer.self_attn.o_proj, process_group_manager)
            layer.mlp.gate_proj = ColumnParallelLinear.from_linear(layer.mlp.gate_proj, process_group_manager)
            layer.mlp.up_proj = ColumnParallelLinear.from_linear(layer.mlp.up_proj, process_group_manager)
            layer.mlp.down_proj = RowParallelLinear.from_linear(layer.mlp.down_proj, process_group_manager)

            # 修改每一层的配置，因为调用了q/k/v_proj后，会根据num_heads和num_key_value_head对q/k/v进行reshape
            layer.self_attn.config.num_attention_heads //= process_group_manager.tp_world_size
            layer.self_attn.num_heads //= process_group_manager.tp_world_size
            layer.self_attn.config.num_key_value_heads //= process_group_manager.tp_world_size
            layer.self_attn.num_key_value_heads //= process_group_manager.tp_world_size
            layer.self_attn.hidden_size //= process_group_manager.tp_world_size
        
        if process_group_manager.pp_is_first_stage:
            self.model.embed_tokens = EmbeddingParallel.from_embedding(self.model.embed_tokens, process_group_manager)
        
        if process_group_manager.pp_is_last_stage:
            self.lm_head = ColumnParallelLinear.from_linear(self.lm_head, process_group_manager, gather_output=True)

    def forward(self, *args, **kwargs):
        if not self.distributed_inited:
            raise ValueError("You need to call distributed_init first!")
        return super().forward(*args, **kwargs)
