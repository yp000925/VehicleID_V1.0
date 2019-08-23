#!/usr/bin/env bash

EXP_ROOT=./experiments/MLSM_mobNet; shift
DATA_SET=./train_test_split/query_list_800.txt; shift
IMA_ROOT=./image; shift
CKPT=checkpoint-50000; shift

python embed.py \
--experiment_root $EXP_ROOT \
--dataset $DATA_SET \
--image_root $IMA_ROOT \
--checkpoint $CKPT \
--batch_size 128 \
--filename 800query_50000_embeddings.h5 \