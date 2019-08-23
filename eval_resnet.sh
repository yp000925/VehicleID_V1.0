#!/usr/bin/env bash

QUERY_SET=./train_test_split/query_list_2400.txt; shift
GALLERY_SET=./train_test_split/test_list_2400.txt; shift
QUERY_EMB=./experiments/resNet/2400query_21000_embeddings.h5; shift
GALLERY_EMB=./experiments/resNet/2400gallery_21000_embeddings.h5; shift


python evaluate.py \
--excluder diagonal \
--query_dataset $QUERY_SET \
--query_embeddings $QUERY_EMB \
--gallery_dataset $GALLERY_SET \
--gallery_embeddings $GALLERY_EMB \
--metric euclidean \
--filename ./experiments/resNet/2400test_21000_evaluation.json \
--batch_size 256 \

