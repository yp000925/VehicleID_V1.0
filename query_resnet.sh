#!/usr/bin/env bash

EXP_ROOT=./experiments/resNet; shift
DATA_SET=./train_test_split/query_list_2400.txt; shift
IMA_ROOT=./image; shift
CKPT=checkpoint-21000; shift

python embed.py \
--experiment_root $EXP_ROOT \
--dataset $DATA_SET \
--image_root $IMA_ROOT \
--checkpoint $CKPT \
--batch_size 128 \
--filename 2400query_21000_embeddings.h5 \

