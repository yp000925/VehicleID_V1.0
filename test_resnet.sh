#!/usr/bin/env bash

EXP_ROOT=./experiments/resNet; shift
DATA_SET=./train_test_split/test_list_2400.txt; shift
IMA_ROOT=./image; shift
CKPT=checkpoint-21000; shift

python embed.py \
--experiment_root $EXP_ROOT \
--dataset $DATA_SET \
--image_root $IMA_ROOT \
--checkpoint $CKPT \
--batch_size 256 \
--filename 2400gallery_21000_embeddings.h5 \

