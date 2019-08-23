#!/usr/bin/env bash

EXP_ROOT=./experiments/MLSM_resNet101; shift
DATA_SET=./train_test_split/test_list_2400.txt; shift
IMA_ROOT=./image; shift
CKPT=checkpoint-70000; shift
FILENAME=2400gallery_70000_embeddings.h5; shift

python embed.py \
--experiment_root $EXP_ROOT \
--dataset $DATA_SET \
--image_root $IMA_ROOT \
--checkpoint $CKPT \
--batch_size 256 \
--filename  $FILENAME\

