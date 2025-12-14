#!/bin/bash

export HOME=/scratch/cvlab/home/yiyou
export HF_HOME=/scratch/cvlab/home/yiyou/hf
export HF_DATASETS_CACHE=/scratch/cvlab/home/yiyou/hf/datasets
export TRANSFORMERS_CACHE=/scratch/cvlab/home/yiyou/hf/transformers
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE

echo "HOME=$HOME"
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

BLOCK_SIZE=4
# RERANK_K=4  # 重排候选数量，可以设置为 4, 8, 16 等
# 可选：如需调节熵停止策略，可以在这里改参数，例如：
ENTROPY_THRESHOLD=4.0
USE_ENTROPY_STOP=false

python -u main.py \
    loader.eval_batch_size=16 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    data=openwebtext-split \
    data.insert_valid_special=False \
    model.length=1024 \
    model.attn_backend=flex \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
    sampling.use_entropy_stop=${USE_ENTROPY_STOP} \
    sampling.entropy_threshold=${ENTROPY_THRESHOLD} \
    sampling.entropy_guided_xt=true \
    wandb=null \
    mode=ppl_eval > $PWD/logs/bd3lm_owt_block_size${BLOCK_SIZE}_entropy${ENTROPY_THRESHOLD}_stop${USE_ENTROPY_STOP}_all.log

#     trainer.limit_val_batches=10 \