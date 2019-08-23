#!/usr/bin/env bash

QUERY_SET=./train_test_split/query_list_1600.txt; shift
GALLERY_SET=./train_test_split/test_list_1600.txt; shift
QUERY_EMB=./experiments/resnet/1600query_21000_embeddings.h5; shift
GALLERY_EMB=./experiments/resnet/1600gallery_21000_embeddings.h5; shift
FILENAME=./experiments/resnet/1600test_21000_evaluation.json; shift

python evaluate.py \
--excluder diagonal \
--query_dataset $QUERY_SET \
--query_embeddings $QUERY_EMB \
--gallery_dataset $GALLERY_SET \
--gallery_embeddings $GALLERY_EMB \
--metric euclidean \
--filename $FILENAME \
--batch_size 256 \

