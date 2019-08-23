#!/usr/bin/env bash

VAL_SET=./train_test_split/test_list_800.txt; shift
TRAIN_SET=./train_test_split/train_list.txt; shift
IMA_ROOT=./image; shift
EXP_ROOT=./experiments/MLSM_resNet101; shift
INT_CKPT=../DaRi/pre_trained_model/resnet_v1_101.ckpt; shift

python train.py \
    --validation_frequency 100 \
    --validation_image_root $IMA_ROOT \
    --validation_set $VAL_SET \
    --experiment_root $EXP_ROOT \
    --train_set $TRAIN_SET \
    --image_root $IMA_ROOT \
    --checkpoint None \
    --model_name resnet_v1_101 \
    --head_name fusion_resnet101 \
    --embedding_dim 128 \
    --initial_checkpoint $INT_CKPT \
    --batch_p 18 \
    --batch_k 4 \
    --pre_crop_height 224 --pre_crop_width 224 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric euclidean \
    --loss batch_hard \
    --learning_rate 3e-4 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --train_iterations 50000 \
    --decay_start_iteration 15000 \
    --weight_decay_factor 0.001 \
    --checkpoint_frequency 1000 \
    --flip_augment \
    --crop_augment \
