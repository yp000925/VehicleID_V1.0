#!/usr/bin/env bash

QUERY_SET=./train_test_split/query_list_800.txt; shift
GALLERY_SET=./train_test_split/test_list_800.txt; shift
QUERY_EMB=./experiments/MLSM_mobNet/800query_50000_embeddings.h5; shift
GALLERY_EMB=./experiments/MLSM_mobNet/800gallery_50000_embeddings.h5; shift


python evaluate.py \
--excluder diagonal \
--query_dataset $QUERY_SET \
--query_embeddings $QUERY_EMB \
--gallery_dataset $GALLERY_SET \
--gallery_embeddings $GALLERY_EMB \
--metric euclidean \
--filename ./experiments/MLSM_mobNet/800test_50000_evaluation.json \
--batch_size 128 \

