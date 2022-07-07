import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter # pip install tensorboard

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import *
from train import train_epoch
# from train_mmtm_2 import train_epoch
from validation import val_epoch


if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        if opt.resume_path:
            resume_paths = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)#'{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.width_mult) + 'x',
                               opt.modality, str(opt.sample_duration)])
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)

    log_mode = 'w'
    if opt.resume_path:
        log_mode = 'a'
    # Egogesture, with "no-gesture" training, weighted loss
    # class_weights = torch.cat((0.012*torch.ones([1, 83]), 0.00015*torch.ones([1, 1])), 1)
    if opt.use_weightloss:
        assert len(opt.loss_weights) == opt.n_classes, 'len(class_weights) not match classes '
        loss_weights = torch.tensor(opt.loss_weights)
        criterion = nn.CrossEntropyLoss(weight=loss_weights, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss()

    # # nvgesture, with "no-gesture" training, weighted loss
    # class_weights = torch.cat((0.04*torch.ones([1, 25]), 0.0008*torch.ones([1, 1])), 1)
    # criterion = nn.CrossEntropyLoss(weight=class_weights, size_average=False)

    # criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        if opt.resize:
            spatial_transform = Compose([
                Scale(opt.resize_size),
                crop_method,
                SpatialElasticDisplacement(),
                ToTensor(opt.norm_value), norm_method])
        else:
            spatial_transform = Compose([
                #RandomHorizontalFlip(),
                #RandomRotate(),
                #RandomResize(),
                crop_method,
                #MultiplyValues(),
                #Dropout(),
                #SaltImage(),
                #Gaussian_blur(),
                SpatialElasticDisplacement(),
                ToTensor(opt.norm_value), norm_method
            ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        
        if opt.use_weightsampler:
            from torch.utils.data.sampler import  WeightedRandomSampler
            weights = opt.sampler_weights
            samples_weights = [weights[sample['label']] for sample in training_data.data]
            sampler = WeightedRandomSampler(samples_weights, opt.n_samples if opt.n_samples else len(samples_weights))  # replacement=True  (default)
            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                # shuffle=True,
                num_workers=opt.n_threads,
                sampler=sampler,
                pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True)
        # train_logger = Logger(
        #     os.path.join(opt.result_path, opt.store_name + '_train.log'),
        #     ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
        # train_batch_logger = Logger(
        #     os.path.join(opt.result_path, 'train_batch.log'),
        #     ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
        train_logger = Logger(
            os.path.join(opt.result_path, 'train_{}.log'.format(opt.store_name)),
            ['epoch', 'loss', 'acc', 'precision','recall','lr', 'confusion_matrix'], log_mode)
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch_{}.log'.format(opt.store_name)),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'precision', 'recall', 'lr'], log_mode)
        best_info_logger = Logger(
            os.path.join(opt.result_path, 'best_info_{}.log'.format(opt.store_name)),
            ['epoch', 'acc', 'precision','recall', 'confusion_matrix'], log_mode)

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening

        if opt.model == 'mmtnet':
            optimizer = [optim.SGD(model.module.get_rgb_params(), lr=opt.learning_rate, momentum=opt.momentum, dampening=dampening, weight_decay=opt.weight_decay, nesterov=opt.nesterov),
                        optim.SGD(model.module.get_depth_params(), lr=opt.learning_rate, momentum=opt.momentum, dampening=dampening, weight_decay=opt.weight_decay, nesterov=opt.nesterov)]    
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()), #parameters,  # 
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)

        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 'min', patience=opt.lr_patience)
        train_writer = SummaryWriter(opt.result_path)

    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        #temporal_transform = LoopPadding(opt.sample_duration)
        temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        # val_logger = Logger(
        #     os.path.join(opt.result_path, opt.store_name + '_val.log'), ['epoch', 'loss', 'prec1', 'prec5'])
        val_logger = Logger(
            os.path.join(opt.result_path, 'val_{}.log'.format(opt.store_name)), 
            ['epoch', 'loss', 'acc','precision', 'recall', 'confusion_matrix'], log_mode)

    best_info = {'epoch':0, 'acc':0, 'precision':0, 'recall':0, 'confusion_matrix': np.zeros((opt.n_classes, opt.n_classes), dtype=np.int).tolist()}
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        best_info = checkpoint['best_info']
        opt.begin_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['state_dict'])
        if type(optimizer) == list:
            for i in range(len(optimizer)):
                optimizer[i].load_state_dict(checkpoint['optimizer'][i])
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
    # for i in range(opt.begin_epoch, opt.begin_epoch + 10):
        if not opt.no_train:
            adjust_learning_rate(optimizer, i, opt)
            with torch.autograd.set_detect_anomaly(True):
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger, train_writer)
            if type(optimizer) == list:
                optim_state_dict = [temp.state_dict() for temp in optimizer]
            else:
                optim_state_dict = optimizer.state_dict()
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optim_state_dict,
                'best_info': best_info
                }
            save_checkpoint(state, False, opt)
            
        if not opt.no_val:
            validation_loss, acc, precision, recall, confusion_matrix = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
            is_best = acc > best_info['acc']
            if is_best:
                best_info['epoch'] = i
                best_info['acc'] = acc
                best_info['precision'] = precision
                best_info['recall'] = recall
                best_info['confusion_matrix'] = confusion_matrix

                bs_name = '%s_%d_%s'%(opt.dataset, opt.n_classes, opt.modality)
                class_names = get_classnames(opt.dataset)
                np.set_printoptions(precision=2)

                plot_confusion_matrix(np.array(confusion_matrix), classes=class_names,
                                      title='Confusion matrix without normalization',
                                      save_name=os.path.join(opt.result_path, bs_name))

                plot_confusion_matrix(np.array(confusion_matrix), classes=class_names, normalize=True,
                                      title='Normalized confusion matrix',
                                      save_name=os.path.join(opt.result_path, '%s_norm'%bs_name))

            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optim_state_dict,#optimizer.state_dict(),
                'best_info': best_info
                }
            save_checkpoint(state, is_best, opt)
            
    best_info_logger.log(best_info)
    train_writer.close()





