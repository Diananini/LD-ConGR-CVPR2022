import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/root/data/ActivityNet', type=str, help='Root directory path of data')
    parser.add_argument('--video_path', default='video_kinetics_jpg', type=str, help='Directory path of Videos')
    parser.add_argument('--annotation_path', default='kinetics.json', type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--modality', default='RGB', type=str, help='Modality of generated model. RGB, Flow or RGBFlow')
    parser.add_argument('--pretrain_modality', default='RGB', type=str, help='Modality of the pretrain model. RGB, Flow or RGBFlow')
    parser.add_argument('--dataset', default='kinetics', type=str, help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument('--n_classes', default=400, type=int, help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_finetune_classes', default=400, type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str, help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--learning_rate', default=0.04, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_steps', default=[15, 25, 35, 45, 60, 50, 200, 250], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10') # [15, 30, 37, 50, 200, 250]
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--mean_dataset', default='activitynet', type=str, help='dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=250, type=int, help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument('--pretrain_dataset', default='', type=str, help='dataset from pretrained model')
    parser.add_argument('--ft_portion', default='complete', type=str, help='The portion of the model to apply fine tuning, either complete or last_layer')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument('--test_subset', default='val', type=str, help='Used subset in test (val | test)')
    parser.add_argument('--scale_in_test', default=1.0, type=float, help='Spatial scale in test')
    parser.add_argument('--crop_position_in_test', default='c', type=str, help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument('--no_softmax_in_test', action='store_true', help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=10, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--no_hflip', action='store_true', help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--model', default='resnet', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--version', default=1.1, type=float, help='Version of the model')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--groups', default=3, type=int, help='The number of groups at group convolutions at conv layers')
    parser.add_argument('--width_mult', default=1.0, type=float, help='The applied width multiplier to scale number of filters')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--train_validate', action='store_true', help='If true, test is performed.')
    parser.set_defaults(train_validate=False)
    parser.add_argument('--use_weightloss', action='store_true', help='If true, weighted cross entropy loss is used')
    parser.set_defaults(weighted_loss=False)
    parser.add_argument('--loss_weights', default=[1,2], type=float, nargs="+", metavar='lossweights', help='loss weights of each classes')
    parser.add_argument('--log_postfix', default='', type=str, help='postfix of test logger')
    parser.add_argument('--no_change_firstlayer', action='store_true', help='If true, first conv layer is changed.')
    parser.set_defaults(no_change_firstlayer=False)
    parser.add_argument('--force_change_firstlayer', action='store_true', help='If true, first conv layer is forced to change.')
    parser.set_defaults(force_change_firstlayer=False)
    parser.add_argument('--resize', action='store_true', help='If true, the ori image will be resized.')
    parser.set_defaults(resize=False)
    parser.add_argument('--resize_size', nargs=2, type=int, metavar=('w', 'h'), help='Resize the image to the specific size.')
    parser.add_argument('--use_weightsampler', action='store_true', help='If true, weight the classes')
    parser.set_defaults(use_weightsampler=False)
    parser.add_argument('--sampler_weights', default=[1,5,5,5], type=float, nargs="+", metavar='classweights', help='sampler weights of each classes')
    parser.add_argument('--n_samples', default=0, type=int, help='total samples in an epoch if use weightsampler')

    parser.add_argument('--extract_motion_part', action='store_true', help='store to extract motion part in magnify net.')
    parser.set_defaults(extract_motion_part=False)

    parser.add_argument('--global_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--local_depth', default=101, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')

    parser.add_argument('--fps', default=30, type=int, help='FPS of the video')

    parser.add_argument('--save_all', action='store_true', help='If true, save all checkpoints')
    parser.set_defaults(save_all=False)

    parser.add_argument('--iscrop', action='store_true', help='If true, crop gesture hand region for recognition')
    parser.set_defaults(iscrop=False)

    parser.add_argument('--isKeyframes', action='store_true', help='If true, extract key frames for recognition')
    parser.set_defaults(isKeyframes=False)

    args = parser.parse_args()

    return args


def parse_opts_online():
    # Real-time test arguments with detector and classifier architecture
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/root/data/ActivityNet', type=str, help='Root directory path of data')
    parser.add_argument('--video_path', default='video_kinetics_jpg', type=str, help='Directory path of Videos')
    parser.add_argument('--video', default='', type=str, help='Directory path of test Videos')
    parser.add_argument('--whole_path', default='video_kinetics_jpg', type=str, help='The whole path of Videos')
    parser.add_argument('--annotation_path', default='kinetics.json', type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--pretrain_dataset', default='', type=str, help='dataset from pretrained model')
    parser.add_argument('--modality', default='RGB', type=str, help='Modality of input data. RGB, Flow or RGBFlow')
    parser.add_argument('--modality_det', default='RGB', type=str, help='Modality of input data. RGB, Flow or RGBFlow')
    parser.add_argument('--modality_clf', default='RGB', type=str, help='Modality of input data. RGB, Flow or RGBFlow')
    parser.add_argument('--dataset', default='kinetics', type=str,
                        help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument('--n_classes_det', default=400, type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_finetune_classes_det', default=400, type=int,
                        help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument('--n_classes_clf', default=400, type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_finetune_classes_clf', default=400, type=int,
                        help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument('--no_change_firstlayer_det', action='store_true', help='If true, first conv layer is changed.')
    parser.set_defaults(no_change_firstlayer_det=False)
    parser.add_argument('--no_change_firstlayer_clf', action='store_true', help='If true, first conv layer is changed.')
    parser.set_defaults(no_change_firstlayer_clf=False)
    parser.add_argument('--no_change_firstlayer', action='store_true', help='If true, first conv layer is changed.')
    parser.set_defaults(no_change_firstlayer=False)
    parser.add_argument('--force_change_firstlayer_det', action='store_true', help='If true, first conv layer is forced to change.')
    parser.set_defaults(force_change_firstlayer_det=False)
    parser.add_argument('--force_change_firstlayer_clf', action='store_true', help='If true, first conv layer is forced to change.')
    parser.set_defaults(force_change_firstlayer_clf=False)
    parser.add_argument('--force_change_firstlayer', action='store_true', help='If true, first conv layer is forced to change.')
    parser.set_defaults(force_change_firstlayer=False)

    parser.add_argument('--n_classes', default=400, type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_finetune_classes', default=400, type=int,
                        help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration_det', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--sample_duration_clf', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')

    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str,
                        help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_steps', default=[10, 20, 30, 40, 100], type=float, nargs="+", metavar='LRSteps',
                        help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--mean_dataset', default='activitynet', type=str,
                        help='dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--lr_patience', default=10, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=200, type=int, help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', default=1, type=int,
                        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--resume_path_det', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--resume_path_clf', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path_det', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument('--pretrain_path_clf', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument('--pretrain_path', default='', type=str, help='Pretrained model (.pth)')

    parser.add_argument('--ft_begin_index', default=0, type=int, help='Begin block index of fine-tuning')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument('--test_subset', default='val', type=str, help='Used subset in test (val | test)')
    parser.add_argument('--scale_in_test', default=1.0, type=float, help='Spatial scale in test')
    parser.add_argument('--crop_position_in_test', default='c', type=str,
                        help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument('--no_softmax_in_test', action='store_true',
                        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=10, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--no_hflip', action='store_true', help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument('--norm_value', default=1, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')

    parser.add_argument('--model_det', default='resnet', type=str,
                        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth_det', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut_det', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k_det', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality_det', default=32, type=int, help='ResNeXt cardinality')

    parser.add_argument('--model', default='resnet', type=str,
                        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')

    parser.add_argument('--model_clf', default='resnet', type=str,
                        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth_clf', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut_clf', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k_clf', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality_clf', default=32, type=int, help='ResNeXt cardinality')

    parser.add_argument('--width_mult', default=1.0, type=float, help='The applied width multiplier to scale number of filters')
    parser.add_argument('--width_mult_det', default=1.0, type=float, help='The applied width multiplier to scale number of filters')
    parser.add_argument('--width_mult_clf', default=1.0, type=float, help='The applied width multiplier to scale number of filters')
    
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--det_strategy', default='raw', type=str, help='Detector filter (raw | median | ma | ewma)')
    parser.add_argument('--det_queue_size', default=1, type=int, help='Detector queue size')
    parser.add_argument('--det_counter', default=1, type=float, help='Number of consequtive detection')
    parser.add_argument('--clf_strategy', default='raw', type=str, help='Classifier filter (raw | median | ma | ewma)')
    parser.add_argument('--clf_queue_size', default=1, type=int, help='Classifier queue size')
    parser.add_argument('--clf_threshold_pre', default=1, type=float, help='Cumulative sum threshold to prepredict')
    parser.add_argument('--clf_threshold_final', default=1, type=float,
                        help='Cumulative sum threshold to predict at the end')
    parser.add_argument('--stride_len', default=1, type=int, help='Stride Lenght of video loader window')
    parser.add_argument('--ft_portion', default='complete', type=str, help='The portion of the model to apply fine tuning, either complete or last_layer')
    parser.add_argument('--groups', default=3, type=int, help='The number of groups at group convolutions at conv layers')
    parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
    parser.add_argument('--log_postfix', default='', type=str, help='postfix of test logger')
    parser.add_argument('--frames_dir', default='videos', type=str, help='videos dir')
    parser.add_argument('--click_frame_threshold', default=25, type=int, help='If gesture duration is longer than the threshold, update the gesture to double-click')
    
    parser.add_argument('--extract_motion_part', action='store_true', help='store to extract motion part in magnify net.')
    parser.set_defaults(extract_motion_part=False)

    parser.add_argument('--global_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--local_depth', default=101, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')

    parser.add_argument('--active_threshold', default=8, type=int, help='confirm a gesture if the total number of active frames is greater than the threshold')
    
    parser.add_argument('--out_prefix', default='out', type=str, help='prefix name for output videos ')
    args = parser.parse_args()

    return args

