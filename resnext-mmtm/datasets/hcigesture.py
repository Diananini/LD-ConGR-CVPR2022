import torch
import torch.utils.data as data
from PIL import Image
from spatial_transforms import *
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random
import sys

from utils import load_value_file, get_location
import pdb


# import sys
# sys.path.insert(0, '/data6/diana/gesture/Real-time-GesRec/utils')
# import darknet_utils
crop_regions_json_path = 'annotation_hciGesture/expand_crop_regions.json'
with open(crop_regions_json_path) as f:
    crop_regions = json.load(f)

def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality == 'RGB':
                return img.convert('RGB')
            elif modality == 'Depth':
                return img.convert('L') # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def video_loader(video_dir_path, frame_indices, modality, sample_duration, iscrop, start_frame_index, image_loader):
    video = []
    if iscrop:
            video_name = '_'.join(os.path.normpath(video_dir_path).split('frames/')[1].split(os.sep))
            # if video_name not in crop_regions:
            #     print('!!!!!! Error: crop_region of %s missed'%video_name)
            #     exit()
            # elif str(start_frame_index) not in crop_regions[video_name]:
            #     print('!!!!!! Error: crop_region of %s %d %d missed'%(video_name, start_frame_index))
            #     exit()
            # else:
            crop_region = crop_regions[video_name][str(start_frame_index)]

    if modality == 'RGB':
        # pre_rec = ()
        # human_annos = get_human_annos(os.path.normpath(video_dir_path).replace('frames', 'human_annos')+'.json')
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:05d}.png'.format(i))
            if os.path.exists(image_path):
                # ------ Manually annotate the human region ------
                # rec = get_location(os.path.basename(video_dir_path).split('_')[2])
                # video.append(image_loader(image_path, modality).crop(rec))

                # ------ yolo v4 detect ------
                # image_path = os.path.normpath(image_path) # A/./B and A/foo/../B all become A/B.
                # ori_image = image_loader(image_path, modality)
                # try:
                #     rec = human_annos[os.path.basename(image_path)]
                # except:
                #     print(image_path, 'human annos wrong')
                # video.append(ori_image.crop(rec).resize((320, 240), Image.BILINEAR))
                # pre_rec = rec

                # ------ original image ------
                # video.append(image_loader(image_path, modality))

                # ------ crop gesture region --------
                ori_img = image_loader(image_path, modality)
                if iscrop:
                    video.append(ori_img.crop(crop_region))#.resize((320, 240), Image.BILINEAR)
                else:
                    video.append(ori_img)
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'Depth':

        for i in frame_indices:
            image_path = os.path.join(video_dir_path.replace('color','depth'), '{:05d}.png'.format(i) )
            if os.path.exists(image_path):
                ori_img = image_loader(image_path, modality)
                if iscrop:
                    video.append(ori_img.crop(crop_region))
                else:
                    video.append(ori_img)
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'RGB-D':
        for i in frame_indices: # index 35 is used to change img to flow
            image_path = os.path.join(video_dir_path, '{:05d}.png'.format(i))

            image_path_depth = os.path.join(video_dir_path.replace('color','depth'), '{:05d}.png'.format(i) )
            
            image = image_loader(image_path, 'RGB')
            image_depth = image_loader(image_path_depth, 'Depth')

            if os.path.exists(image_path):
                if iscrop:
                    video.append(image.crop(crop_region))
                    video.append(image_depth.crop(crop_region))
                else:
                    video.append(image)
                    video.append(image_depth)
            else:
                print(image_path, "------- Does not exist")
                return video
    
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_human_annos(anno_path):
    with open(anno_path) as anno:
        return json.load(anno)

def get_hand_annos(base_anno_path, frame_index):
    anno_path = base_anno_path+'_{:05d}.txt'.format(int((frame_index+4)/2))
    if not os.path.exists(anno_path):
        anno_path = base_anno_path+'_{:05d}.txt'.format(int((frame_index+4)/2)+1)
    if not os.path.exists(anno_path):
        print('invalid hand anno path', anno_path)
        return []
    with open(anno_path) as f:
        for line in f:
            return [float(t) for t in line.split()[1:]]

def get_crop_region(bbox, width, height):
    x0, y0, w0, h0 = bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height
    region_w, region_h = 5*w0, 4*h0
    # region_w, region_h = 3*w0, 2*w0
    coors = [x0-region_w/2, y0-region_h/2, x0+region_w/2, y0+region_h/2]
    for i in [0,1]:
        if coors[i] < 0: 
            coors[i] -= coors[i]
            coors[i+2] -= coors[i]
    
    if coors[2] > width:
        coors[0] -= coors[2]-width
        coors[2] = width
    if coors[3] > height:
        coors[1] -= coors[3]-height
        coors[3] = height

    coors[0] = max(coors[0], 0)
    coors[1] = max(coors[1], 0)

    return [int(round(t)) for t in coors]

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            #video_names.append('{}/{}'.format(label, key))
            video_names.append(key.split('^')[0])
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration, isKeyframes):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    print("[INFO]: HCIGesture Dataset - " + subset + " is loading...")
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        
        if not os.path.exists(video_path):
            print('video path not exist!!!', video_path)
            continue

        begin_t = int(annotations[i]['start_frame'])
        end_t = int(annotations[i]['end_frame'])
        n_frames = end_t - begin_t + 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            #'video_id': video_names[i].split('/')[1]
            'video_id': i
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            if isKeyframes:
                key_frames = [int(t) for t in annotations[i]['key_frames']]
                if len(key_frames) < sample_duration:
                    all_frames = list(range(begin_t, end_t + 1))
                    if len(all_frames)<sample_duration:
                        key_frames = all_frames
                    else:
                        non_key_frames = set(all_frames)-set(key_frames)
                        add_frames = random.sample(non_key_frames, sample_duration-len(key_frames))
                        key_frames = sorted(key_frames+add_frames)
                sample['frame_indices'] = key_frames

            else:
                sample['frame_indices'] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class HCIGesture(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 iscrop=False,
                 isKeyframes=False,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration, isKeyframes)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader()
        self.iscrop = iscrop
        self.n_samples_for_each_video = n_samples_for_each_video
        # self.human_annos = get_human_annos(os.path.join(os.path.dirname(annotation_path), 'human_annos.json'))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        start_frame_index, end_frame_index = self.data[index]['segment'] #frame_indices[0]

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)  # crop the length of frame indices to sample_duration

        clip = self.loader(path, frame_indices, self.modality, self.sample_duration, self.iscrop, start_frame_index)

        oversample_clip =[]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        
        # print('length of clip', len(clip))

        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
             
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.n_samples_for_each_video != 1:
            info = ' '.join([path, str(start_frame_index), str(end_frame_index),str(self.data[index]['label'])])
            return clip, target, info
        return clip, target, path

    def __len__(self):
        return len(self.data)


