from transformers import Qwen2ForCausalLM, Qwen2Config
from tqdm import tqdm
import copy

from .model import Qwen2ForCausalLMDistributedShard

def slice_model(model, tp_size, pp_size, verbose=False):
    # model: complete model
    for i in tqdm(range(pp_size * tp_size), disable=not verbose):
        pp_rank = i // tp_size
        tp_rank = i % tp_size
        config = copy.deepcopy(model.config)
        config.tp_size = tp_size
        config.pp_size = pp_size
        config.tp_rank = tp_rank
        config.pp_rank = pp_rank
        shard = Qwen2ForCausalLMDistributedShard(config)
        shard.to(model.dtype)
        shard.copy_from_complete_model(model)
        yield shard

def combine_shards(shards, verbose=False):
    config = copy.deepcopy(shards[0].config)
    del config.tp_rank
    del config.tp_size
    del config.pp_rank
    del config.pp_size

    model = Qwen2ForCausalLM(config)
    model.to(shards[0].dtype)

    for shard in tqdm(shards, disable=not verbose):
        shard.copy_to_complete_model(model)
    return model
