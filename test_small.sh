#!/usr/bin/env bash

EXP_ROOT=./experiments/MLSM_resNet; shift
DATA_SET=./train_test_split/test_list_3200.txt; shift
IMA_ROOT=./image; shift
CKPT=checkpoint-34; shift

python embed.py \
--experiment_root  $EXP_ROOT \
--dataset $DATA_SET \
--image_root $IMA_ROOT \
--checkpoint $CKPT \
--batch_size 128 \
--filename vehicleID_test_embeddings.h5 \
