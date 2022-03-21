#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
nohup python main.py \
   --root_path /data2/project/gesture/resnext-mmtm/ \
   --video_path /data/datasets/HCIGesture \
   --annotation_path annotation_hciGesture/dataV3/hciall_but_None.json \
   --result_path results/hci-models/v3/hci_clf_train_resnext_rgb_112_allbutnone \
   --dataset hcigesture \
   --sample_duration 32 \
   --learning_rate 0.0005 \
   --model resnext \
   --model_depth 101 \
   --resnet_shortcut B \
   --batch_size 32 \
   --n_classes 10 \
   --n_finetune_classes 10 \
   --n_threads 3 \
   --checkpoint 1 \
   --modality RGB \
   --train_crop random \
   --n_val_samples 1 \
   --test_subset test \
   --n_epochs 100 \
   --sample_size 112 \
   --save_all \
   --no_val \
   --pretrain_path report/models/jester_resnext_101_RGB_32.pth \
   --pretrain_dataset jester > out/hci_clf_train_resnext_rgb_112_allbutnone_datav3_32.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=1,2
# nohup python main.py \
#    --root_path /data2/project/gesture/resnext-mmtm/ \
#    --video_path /data/datasets/HCIGesture \
#    --annotation_path annotation_hciGesture/dataV3/hciall_but_None.json \
#    --result_path results/hci-models/v3/hci_clf_train_resnext_rgb_112_allbutnone_crop \
#    --dataset hcigesture \
#    --sample_duration 32 \
#    --learning_rate 0.0005 \
#    --model resnext \
#    --model_depth 101 \
#    --resnet_shortcut B \
#    --batch_size 32 \
#    --n_classes 10 \
#    --n_finetune_classes 10 \
#    --n_threads 3 \
#    --checkpoint 1 \
#    --modality RGB \
#    --train_crop random \
#    --n_val_samples 1 \
#    --test_subset test \
#    --n_epochs 100 \
#    --sample_size 112 \
#    --save_all \
#    --resize \
#    --resize_size 480 360 \
#    --iscrop \
#    --no_val \
#    --pretrain_path report/models/jester_resnext_101_RGB_32.pth \
#    --pretrain_dataset jester > out/hci_clf_train_resnext_rgb_112_allbutnone_crop_datav3_32.out 2>&1 &


# export CUDA_VISIBLE_DEVICES=1,3
# nohup python main.py \
#     --root_path /data2/project/gesture/resnext-mmtm/ \
#     --video_path /data/datasets/HCIGesture \
#     --annotation_path annotation_hciGesture/dataV3/hciall_but_None.json \
#     --result_path results/hci-models/v3/hci_clf_train_mmtnet_rgbd_112_allbutnone_32 \
#     --dataset hcigesture \
#     --sample_duration 32 \
#     --learning_rate 0.01 \
#     --model mmtnet \
#     --model_depth 101 \
#     --resnet_shortcut B \
#     --batch_size 16 \
#     --n_classes 10 \
#     --n_finetune_classes 10 \
#     --n_threads 1 \
#     --checkpoint 1 \
#     --modality RGB-D \
#     --train_crop random \
#     --n_val_samples 1 \
#     --test_subset test \
#     --n_epochs 100 \
#     --sample_size 112 \
#     --save_all \
#     --no_val \
#     --pretrain_path report/models/jester_resnext_101_RGB_32.pth \
#     --pretrain_dataset jester > out/hci_clf_train_resnext_rgbd_112_allbutnone_datav3_32.out 2>&1 &


# export CUDA_VISIBLE_DEVICES=4,3
# nohup python main.py \
#    --root_path /data2/project/gesture/resnext-mmtm/ \
#    --video_path /data/datasets/HCIGesture \
#    --annotation_path annotation_hciGesture/dataV3/hciall_but_None_keyframes.json \
#    --result_path results/hci-models/v3/hci_clf_train_resnext_rgb_112_allbutnone_32keyframes_crop_pretrain_2 \
#    --dataset hcigesture \
#    --sample_duration 32 \
#    --learning_rate 0.0005 \
#    --model resnext \
#    --model_depth 101 \
#    --resnet_shortcut B \
#    --batch_size 32 \
#    --n_classes 10 \
#    --n_finetune_classes 10 \
#    --n_threads 3 \
#    --checkpoint 1 \
#    --modality RGB \
#    --train_crop random \
#    --n_val_samples 1 \
#    --test_subset test \
#    --n_epochs 100 \
#    --sample_size 112 \
#    --save_all \
#    --isKeyframes \
#    --resize \
#    --resize_size 480 360 \
#    --iscrop \
#    --no_val \
#    --pretrain_path results/hci-models/v3/hci_clf_train_resnext_rgb_112_allbutnone_crop_resize/hcigesture_resnext_1.0x_RGB_32_epoch_028.pth \
#    --pretrain_dataset hcigesture > out/hci_clf_train_resnext_rgb_112_allbutnone_datav3_32keyframes_crop_pretrain_2.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=1,2
# nohup python main.py \
#    --root_path /data2/project/gesture/resnext-mmtm/ \
#    --video_path /data/datasets/HCIGesture \
#    --annotation_path annotation_hciGesture/dataV3/hciall_but_None_keyframes.json \
#    --result_path results/hci-models/v3/hci_clf_train_mmtnet_rgbd_112_allbutnone_32keyframes_crop_pretrain \
#    --dataset hcigesture \
#    --sample_duration 32 \
#    --learning_rate 0.0005 \
#    --model mmtnet \
#    --model_depth 101 \
#    --resnet_shortcut B \
#    --batch_size 32 \
#    --n_classes 10 \
#    --n_finetune_classes 10 \
#    --n_threads 3 \
#    --checkpoint 1 \
#    --modality RGB-D \
#    --train_crop random \
#    --n_val_samples 1 \
#    --test_subset test \
#    --n_epochs 100 \
#    --sample_size 112 \
#    --save_all \
#    --isKeyframes \
#    --resize \
#    --resize_size 480 360 \
#    --iscrop \
#    --no_val \
#    --pretrain_path results/hci-models/v3/hci_clf_train_mmtnet_rgbd_112_allbutnone_crop/hcigesture_mmtnet_1.0x_RGB-D_32_epoch_024.pth \
#    --pretrain_dataset hcigesture > out/hci_clf_train_mmtnet_rgbd_112_allbutnone_datav3_32keyframes_crop_pretrain.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=4,3
# nohup python main.py \
#    --root_path /data2/project/gesture/resnext-mmtm/ \
#    --video_path /data/datasets/HCIGesture \
#    --annotation_path annotation_hciGesture/dataV3/hciall_but_None_keyframes.json \
#    --result_path results/hci-models/v3/hci_clf_train_mmtnet_rgbd_112_allbutnone_16keyframes_crop_pretrain \
#    --dataset hcigesture \
#    --sample_duration 16 \
#    --learning_rate 0.0005 \
#    --model mmtnet \
#    --model_depth 101 \
#    --resnet_shortcut B \
#    --batch_size 32 \
#    --n_classes 10 \
#    --n_finetune_classes 10 \
#    --n_threads 2 \
#    --checkpoint 1 \
#    --modality RGB-D \
#    --train_crop random \
#    --n_val_samples 1 \
#    --test_subset test \
#    --n_epochs 100 \
#    --sample_size 112 \
#    --save_all \
#    --isKeyframes \
#    --resize \
#    --resize_size 480 360 \
#    --iscrop \
#    --no_val \
#    --pretrain_path results/hci-models/v3/hci_clf_train_mmtnet_rgbd_112_allbutnone_16_crop/hcigesture_mmtnet_1.0x_RGB-D_16_epoch_023.pth \
#    --pretrain_dataset hcigesture > out/hci_clf_train_mmtnet_rgbd_112_allbutnone_datav3_16keyframes_crop_pretrain.out 2>&1 &




