#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,7
nohup python offline_test.py \
    --root_path /data2/project/gesture/resnext-mmtm/ \
    --video_path /data/datasets/HCIGesture \
    --annotation_path annotation_hciGesture/dataV3/hciall_but_None.json \
    --result_path results/hci_clf_test \
    --resume_path results/hci-models/v3/hci_clf_train_resnext_rgb_112_allbutnone \
    --dataset hcigesture \
    --sample_duration 32 \
    --model resnext \
    --model_depth 101 \
    --resnet_shortcut B \
    --batch_size 32 \
    --n_classes 10 \
    --n_finetune_classes 10 \
    --n_threads 2 \
    --modality RGB \
    --n_val_samples 1 \
    --test_subset test \
    --sample_size 112 \
    --log_postfix _datav3_rgb > out/test_clf_hci_datav3_allbutnone_rgb.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=7
# nohup python offline_test.py \
#     --root_path /data2/project/gesture/resnext-mmtm/ \
#     --video_path /data/datasets/HCIGesture \
#     --annotation_path annotation_hciGesture/dataV3/hciall_but_None.json \
#     --result_path results/hci_clf_test \
#     --resume_path results/hci-models/v3/hci_clf_train_mmtnet_rgbd_112_allbutnone_16 \
#     --dataset hcigesture \
#     --sample_duration 16 \
#     --model mmtnet \
#     --model_depth 101 \
#     --resnet_shortcut B \
#     --batch_size 64 \
#     --n_classes 10 \
#     --n_finetune_classes 10 \
#     --n_threads 2 \
#     --modality RGB-D \
#     --n_val_samples 1 \
#     --test_subset test \
#     --sample_size 112 \
#     --log_postfix _datav3_rgbd_16 > out/test_clf_hci_datav3_allbutnone_rgbd_16.out 2>&1 &
        # --iscrop \


# export CUDA_VISIBLE_DEVICES=0,7
# nohup python offline_test.py \
#     --root_path /data2/project/gesture/resnext-mmtm/ \
#     --video_path /data/datasets/HCIGesture \
#     --annotation_path annotation_hciGesture/dataV3/hciall_but_None_keyframes.json \
#     --result_path results/hci_clf_test \
#     --resume_path results/hci-models/v3/hci_clf_train_resnext_rgb_112_allbutnone_32keyframes_crop_pretrain_2 \
#     --dataset hcigesture \
#     --sample_duration 32 \
#     --model resnext \
#     --model_depth 101 \
#     --resnet_shortcut B \
#     --batch_size 32 \
#     --n_classes 10 \
#     --n_finetune_classes 10 \
#     --n_threads 2 \
#     --modality RGB \
#     --n_val_samples 1 \
#     --test_subset test \
#     --sample_size 112 \
#     --iscrop \
#     --isKeyframes \
#     --log_postfix _datav3_rgb_32keyframes_crop_pretrain_2 > out/test_clf_hci_datav3_allbutnone_rgb_32keyframes_crop_pretrain_2.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=0,7
# nohup python offline_test.py \
#     --root_path /data2/project/gesture/resnext-mmtm/ \
#     --video_path /data/datasets/HCIGesture \
#     --annotation_path annotation_hciGesture/dataV3/hciall_but_None_keyframes.json \
#     --result_path results/hci_clf_test \
#     --resume_path results/hci-models/v3/hci_clf_train_mmtnet_rgbd_112_allbutnone_16keyframes_crop_pretrain \
#     --dataset hcigesture \
#     --sample_duration 16 \
#     --model mmtnet \
#     --model_depth 101 \
#     --resnet_shortcut B \
#     --batch_size 32 \
#     --n_classes 10 \
#     --n_finetune_classes 10 \
#     --n_threads 2 \
#     --modality RGB-D \
#     --n_val_samples 1 \
#     --test_subset test \
#     --sample_size 112 \
#     --iscrop \
#     --isKeyframes \
#     --log_postfix _datav3_rgbd_16keyframes_crop_pretrain > out/test_clf_hci_datav3_allbutnone_rgbd_16keyframes_crop_pretrain.out 2>&1 &
#     


# export CUDA_VISIBLE_DEVICES=0,1
# nohup python offline_test.py \
#     --root_path /data2/project/gesture/resnext-mmtm/ \
#     --video_path /data/datasets/HCIGesture \
#     --annotation_path annotation_hciGesture/dataV3/hciall_but_None.json \
#     --result_path results/hci_clf_test \
#     --resume_path results/hci-models/v3/hci_clf_train_mmtnet_rgbd_112_allbutnone_crop/hcigesture_mmtnet_1.0x_RGB-D_32_epoch_024.pth \
#     --dataset hcigesture \
#     --sample_duration 32 \
#     --model mmtnet \
#     --model_depth 101 \
#     --resnet_shortcut B \
#     --batch_size 64 \
#     --n_classes 10 \
#     --n_finetune_classes 10 \
#     --n_threads 1 \
#     --modality RGB-D \
#     --n_val_samples 1 \
#     --test_subset test \
#     --sample_size 112 \
#     --iscrop \
#     --log_postfix _datav3_rgbd_32_crop_epoch24 > out/test_clf_hci_datav3_allbutnone_rgbd_32_crop_epoch24.out 2>&1 &