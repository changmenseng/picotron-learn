

def linear_column_slice(linear, id_, total):
    assert linear.out_features % total == 0
    out_features = linear.out_features // total
    linear.out_features = out_features
    linear.weight.data = linear.weight.data[
        id_ * out_features:
        (id_ + 1) * out_features
    ]
    if linear.bias is not None:
        linear.bias.data = linear.bias.data[
            id_ * out_features:
            (id_ + 1) * out_features
        ]
    return linear

def linear_row_slice(linear, id_, total):
    assert linear.in_features % total == 0
    in_features = linear.in_features // total
    linear.in_features = in_features
    linear.weight.data = linear.weight.data[:,
        id_ * in_features:
        (id_ + 1) * in_features
    ]
    return linear

def embedding_slice(embedding, id_, total):
    assert embedding.num_embeddings % total == 0
    num_embeddings = embedding.num_embeddings // total
    embedding.vocab_offset = id_ * num_embeddings
    embedding.weight.data = embedding.weight.data[
        id_ * num_embeddings:
        (id_ + 1) * num_embeddings
    ]
    embedding.num_embeddings = num_embeddings
    return embedding

def linear_complete_to_shard_column_copy(shard, complete, id_, total):
    assert shard.out_features * total == complete.out_features
    assert shard.in_features == complete.in_features
    shard.weight.data.copy_(
        complete.weight.data[
            id_ * shard.out_features:
            (id_ + 1) * shard.out_features
        ]
    )
    if shard.bias is not None:
        assert complete.bias is not None
        shard.bias.data.copy_(
            complete.bias.data[
                id_ * shard.out_features:
                (id_ + 1) * shard.out_features
            ]
        )

def linear_complete_to_shard_row_copy(shard, complete, id_, total):
    assert shard.in_features * total == complete.in_features
    assert shard.out_features == complete.out_features
    shard.weight.data.copy_(
        complete.weight.data[:,
            id_ * shard.in_features:
            (id_ + 1) * shard.in_features
        ]
    )
    if shard.bias is not None:
        assert complete.bias is not None
        shard.bias.data.copy_(complete.bias.data)

def embedding_complete_to_shard_copy(shard, complete, id_, total):
    assert shard.num_embeddings * total == complete.num_embeddings
    assert shard.embedding_dim == complete.embedding_dim
    shard.weight.data.copy_(
        complete.weight.data[
            id_ * shard.num_embeddings:
            (id_ + 1) * shard.num_embeddings,
        ]
    )

def linear_shard_to_complete_column_copy(shard, complete, id_, total):
    assert complete.out_features // total == shard.out_features
    assert complete.in_features == shard.in_features
    complete.weight.data[
        id_ * shard.out_features:
        (id_ + 1) * shard.out_features
    ].copy_(shard.weight.data)
    if complete.bias is not None:
        assert shard.bias is not None
        complete.bias.data[
            id_ * shard.out_features:
            (id_ + 1) * shard.out_features
        ].copy_(shard.bias.data)

def linear_shard_to_complete_row_copy(shard, complete, id_, total):
    assert complete.in_features // total == shard.in_features
    assert complete.out_features == shard.out_features
    complete.weight.data[:,
        id_ * shard.in_features:
        (id_ + 1) * shard.in_features
    ].copy_(shard.weight.data)
    if complete.bias is not None:
        assert shard.bias is not None
        complete.bias.data.copy_(shard.bias.data)

def embedding_shard_to_complete_copy(shard, complete, id_, total):
    assert complete.num_embeddings // total == shard.num_embeddings
    assert complete.embedding_dim == shard.embedding_dim
    complete.weight.data[
        id_ * shard.num_embeddings:
        (id_ + 1) * shard.num_embeddings
    ].copy_(shard.weight.data)
