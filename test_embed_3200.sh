#!/usr/bin/env bash

EXP_ROOT=./experiments/MLSM_resNet; shift
DATA_SET=./train_test_split/test_list_1600.txt; shift
IMA_ROOT=./image; shift
CKPT=checkpoint-50000; shift

python embed.py \
--experiment_root $EXP_ROOT \
--dataset $DATA_SET \
--image_root $IMA_ROOT \
--checkpoint $CKPT \
--batch_size 256 \
--filename 1600gallery_50000_embeddings.h5 \


