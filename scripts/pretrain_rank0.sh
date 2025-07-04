distributed_args="
    --nproc_per_node 8 \
    --master_addr 10.82.137.13 \
    --master_port 34213 \
    --nnodes 2 \
    --node_rank 0 \
"

seed_args="
    --seed 1116 \
"

comm_args="
    --backend nccl \
    --timeout 30 \
"

model_args="
    --model_path /etc/ssd1/jiangzhongtao/base_models/Qwen2.5-32B-Instruct/picotron/tp-8_pp-2 \
    --tokenizer_path /etc/ssd1/jiangzhongtao/base_models/Qwen2.5-32B-Instruct \
    --dtype bfloat16 \
"

parallel_args="
    --tp_size 8 \
    --pp_size 2 \
"

data_args="
    --train_hf_dataset_dirs /etc/ssd1/jiangzhongtao/qwen-mla-moba/data/tigerbot/en_processed,/etc/ssd1/jiangzhongtao/qwen-mla-moba/data/tigerbot/zh_processed \
    --train_micro_batch_size 1 \
    --chunk_size 5120 \
    --num_workers 4 \
"

train_args="
    --learning_rate 1e-4 \
    --num_micro_batches_per_batch 128 \
    --accumulate_steps 1 \
    --num_steps 2856 \
    --save_step_intervals 238 \
    --save_path /etc/ssd1/jiangzhongtao/qwen-mla-moba/saved/test \
    --warmup_ratio 0.083 \
"

wandb_args="
    --wandb_entity kuaishou-search-tech \
    --wandb_project test_speed \
"

torchrun \
    $distributed_args \
    scripts/pretrain.py \
    $seed_args \
    $comm_args \
    $model_args \
    $parallel_args \
    $data_args \
    $train_args \
    $wandb_args
