from datasets.kinetics import Kinetics
from datasets.ucf101 import UCF101
from datasets.jester import Jester
from datasets.egogesture import EgoGesture
from datasets.nv import NV
from datasets.mousegesture import MouseGesture
from datasets.hcigesture import HCIGesture
from datasets.egogesture_online import EgoGestureOnline
from datasets.nv_online import NVOnline
from datasets.mousegesture_online import MouseGestureOnline
from datasets.hcigesture_online import HCIGestureOnline

def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['kinetics', 'jester', 'ucf101', 'egogesture', 'nvgesture', 'mousegesture', 'hcigesture']

    if opt.train_validate:
        subset = ['training', 'validation']
    else:
        subset = 'training'

    if opt.dataset == 'kinetics':
        training_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'jester':
        training_data = Jester(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            extract_motion_part=opt.extract_motion_part)
    elif opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'egogesture':
        training_data = EgoGesture(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'nvgesture':
        training_data = NV(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            extract_motion_part=opt.extract_motion_part)
    elif opt.dataset == 'mousegesture':
        training_data = MouseGesture(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'hcigesture':
        training_data = HCIGesture(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            iscrop=opt.iscrop,
            isKeyframes=opt.isKeyframes)
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['kinetics', 'jester', 'ucf101', 'egogesture', 'nvgesture', 'mousegesture', 'hcigesture']

    if opt.dataset == 'kinetics':
        validation_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'jester':
        validation_data = Jester(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            extract_motion_part=opt.extract_motion_part)
    elif opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'egogesture':
        validation_data = EgoGesture(
            opt.video_path,
            opt.annotation_path,
            'testing',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'nvgesture':
        validation_data = NV(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            extract_motion_part=opt.extract_motion_part)
    elif opt.dataset == 'mousegesture':
        validation_data = MouseGesture(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'hcigesture':
        validation_data = HCIGesture(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            iscrop=opt.iscrop,
            isKeyframes=opt.isKeyframes,
            n_samples_for_each_video=opt.n_val_samples)

    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'jester', 'ucf101', 'egogesture', 'nvgesture', 'mousegesture', 'hcigesture']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'jester':
        test_data = Jester(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'egogesture':
        test_data = EgoGesture(
            opt.video_path,
            opt.annotation_path,
            subset,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'nvgesture':
        test_data = NV(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            extract_motion_part=opt.extract_motion_part)
    elif opt.dataset == 'mousegesture':
        test_data = MouseGesture(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'hcigesture':
        test_data = HCIGesture(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality,
            iscrop=opt.iscrop,
            isKeyframes=opt.isKeyframes,
            n_samples_for_each_video=opt.n_val_samples)
    return test_data

def get_online_data(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in [ 'egogesture', 'nvgesture', 'mousegesture', 'hcigesture']
    whole_path = opt.whole_path
    if opt.dataset == 'egogesture':
        online_data = EgoGestureOnline(
            opt.annotation_path,  
            opt.video_path,
            opt.whole_path,  
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality="RGB-D",
            stride_len = opt.stride_len,
            sample_duration=opt.sample_duration)
    if opt.dataset == 'nvgesture':
        online_data = NVOnline(
            opt.annotation_path,  
            opt.video_path,
            opt.whole_path,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality="RGB-D",
            stride_len = opt.stride_len,
            sample_duration=opt.sample_duration)
    if opt.dataset == 'mousegesture':
        online_data = MouseGestureOnline(
            opt.annotation_path,
            opt.video_path,
            opt.whole_path,  
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality='RGB-D',
            stride_len = opt.stride_len,
            sample_duration=opt.sample_duration)
    if opt.dataset == 'hcigesture':
        online_data = HCIGestureOnline(
            opt.annotation_path,
            opt.video_path,
            opt.whole_path,  
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality='RGB-D',
            stride_len = opt.stride_len,
            sample_duration=opt.sample_duration)

    return online_data
