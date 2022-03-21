#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
nohup \
	python online_test_wo_detector_my_all.py \
    --root_path /data2/project/gesture/resnext-mmtm/ \
    --video_path /data/datasets/HCIGesture \
    --annotation_path annotation_hciGesture/dataV3/hciall.json \
	--resume_path_clf results/hci-models/v3/hci_clf_train_resnext_rgb_112_all_weighted/hcigesture_resnext_1.0x_RGB_32_epoch_022.pth  \
	--result_path results/hci-models/v3/hci_online_wo_detector_test \
	--dataset hcigesture    \
	--sample_duration_clf 32 \
	--model_clf resnext \
	--model_depth_clf 101 \
	--width_mult_clf 0.5 \
	--resnet_shortcut_clf B \
	--batch_size 1 \
	--n_classes_clf 11 \
	--n_finetune_classes_clf 11 \
	--n_threads 1 \
	--checkpoint 1 \
	--modality_clf RGB \
	--n_val_samples 1 \
	--train_crop random \
	--test_subset test  \
	--det_counter 2 \
	--clf_strategy raw \
	--clf_queue_size 16 \
	--clf_threshold_final 0.95 \
	--stride_len 2 \
	--sample_size 112 \
	--frames_dir frames \
	--active_threshold 4 \
	--log_postfix _RGB_32_raw_thre0.95_active4_stride2_all_weighted_epoch22_2 > out/test_online_hci_rgb_wo_det_raw_RGB32_thre0.95_active4_stride2_all_weighted_epoch22_2.out 2>&1 &
