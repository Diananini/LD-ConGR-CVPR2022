import time
import os
import glob
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set, get_online_data
# import test
from utils import *

def plot_cm(cm, classes, normalize = True):
    import seaborn as sns
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    ax= plt.subplot()
    sns.heatmap(cm, annot=False, ax = ax); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
   
def plot_accuracy(accs_dic):
    epochs = list(accs_dic.keys())
    accs = np.array([accs_dic[epoch] for epoch in sorted(epochs)])
    max_idx = np.argmax(accs)
    plt.cla()
    plt.scatter(epochs, accs, alpha=0.6)
    show_max = '('+str(max_idx)+','+str(accs[max_idx])+')'
    plt.annotate(show_max, xytext=(max_idx,accs[max_idx]), xy=(max_idx,accs[max_idx]))
    plt.savefig(os.path.join(opt.result_path, 'acc'+opt.log_postfix+'.png'), dpi=300)

def test_one(model, test_loader):
    recorder = []
    print('run')
    model.eval()

    batch_time = AverageMeter()
    accuracies_l = [AverageMeter()]
    precisions_l = [AverageMeter()]
    recalls_l = [AverageMeter()]
    confusions_l = [ConfusionMeter(opt.n_classes)]

    y_true = []
    y_pred = []
    diatance_map = {'l1': 1, 'l2': 2, 'l3':3, 'r1':1, 'r2':2, 'r3':3} # 1: [1m, 2m), 2: [2m, 3m), 3: [3m, 4m]
    distance = [] 
    end_time = time.time()
    for i, (inputs, targets, paths) in enumerate(test_loader):
        torch.cuda.empty_cache()
        if not opt.no_cuda:
            # targets = targets.cuda(async=True)
            targets = targets.cuda()
        # inputs = Variable(torch.squeeze(inputs), volatile=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)

            outputs_l = model(inputs)
        
        if type(outputs_l) == tuple:
            outputs_l = list(outputs_l)
            outputs_fusion = sum(outputs_l)
            outputs_l.append(outputs_fusion)
        else:
            outputs_l = [outputs_l]
        
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs_l[-1])
        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())
        for p in paths:
            distance.append(diatance_map[p.split('_')[2]])

        while len(accuracies_l)<len(outputs_l):
            accuracies_l.append(AverageMeter())
            precisions_l.append(AverageMeter())
            recalls_l.append(AverageMeter())
            confusions_l.append(ConfusionMeter(opt.n_classes))

        for c, outputs in enumerate(outputs_l):
            acc = calculate_accuracy(outputs.data, targets.data)[0]
            precision = calculate_precision(outputs, targets)
            recall = calculate_recall(outputs,targets)
            confusion_m = calculate_confusion_matrix(outputs, targets, labels=range(opt.n_classes))

            accuracies_l[c].update(acc, inputs.size(0))
            precisions_l[c].update(precision, inputs.size(0))
            recalls_l[c].update(recall,inputs.size(0))
            confusions_l[c].update(confusion_m)

            c += 1

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # print(y_true)
        # print(y_pred)
        # break


    acc_l = [accuracies.avg for accuracies in accuracies_l]
    pre_l = [precisions.avg for precisions in precisions_l]
    recall_l = [recalls.avg for recalls in recalls_l]
    cm_l = [confusions.conf.tolist() for confusions in confusions_l]
    

    return acc_l, pre_l, recall_l, cm_l, y_true, y_pred, distance

def test_one_split(model, test_loader):
    recorder = {}
    print('run')
    model.eval()

    batch_time = AverageMeter()
    accuracies_l = [AverageMeter()]
    precisions_l = [AverageMeter()]
    recalls_l = [AverageMeter()]
    confusions_l = [ConfusionMeter(opt.n_classes)]

    y_true = []
    y_pred = []
    end_time = time.time()
    for i, (inputs, targets, infos) in enumerate(test_loader):
        torch.cuda.empty_cache()
        if not opt.no_cuda:
            # targets = targets.cuda(async=True)
            targets = targets.cuda()
        # inputs = Variable(torch.squeeze(inputs), volatile=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)

            outputs_l = model(inputs)
        
        if type(outputs_l) == tuple:
            outputs_l = list(outputs_l)
            outputs_l = sum(outputs_l)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs_l,dim=1)
        outputs = outputs.data.cpu().numpy().copy()
        for t, info in enumerate(infos):
            if info not in recorder:
                recorder[info] = []
            recorder[info].append(outputs[t].tolist())

    return recorder#acc_l, pre_l, recall_l, cm_l

opt = parse_opts()
if opt.root_path != '':
    opt.video_path = os.path.join(opt.root_path, opt.video_path)
    opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
    opt.result_path = os.path.join(opt.root_path, opt.result_path)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
    if opt.pretrain_path:
        opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
opt.scales = [opt.initial_scale]
for i in range(1, opt.n_scales):
    opt.scales.append(opt.scales[-1] * opt.scale_step)
opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
opt.mean = get_mean(opt.norm_value)
opt.std = get_std(opt.norm_value)

print(opt)

with open(os.path.join(opt.result_path, 'opts'+opt.log_postfix+'.json'), 'w') as opt_file:
    json.dump(vars(opt), opt_file)

torch.manual_seed(opt.manual_seed)

model, parameters = generate_model(opt)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if
                           p.requires_grad)
print("Total number of trainable parameters: ", pytorch_total_params)

if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)


spatial_transform = Compose([
    #Scale(opt.sample_size),
    Scale(opt.sample_size),
    CenterCrop(opt.sample_size),
    ToTensor(opt.norm_value), norm_method
    ])
temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
#temporal_transform = TemporalBeginCrop(opt.sample_duration)
#temporal_transform = TemporalEndCrop(opt.sample_duration)
target_transform = ClassLabel()
if opt.test_subset == 'train':
    test_data = get_training_set(
        opt, spatial_transform, temporal_transform, target_transform)
else:
    test_data = get_test_set(
        opt, spatial_transform, temporal_transform, target_transform)

test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
test_logger = Logger(os.path.join(opt.result_path, 'test'+opt.log_postfix+'.log'),
 ['epoch', 'top1', 'precision', 'recall', 'confusion_matrix'], open_mode='a')#, open_mode='a' #'top5', 


if os.path.isfile(opt.resume_path):
    model_paths = [opt.resume_path]
else:
    model_paths = glob.glob(os.path.join(opt.resume_path, '*epoch*.pth'))

accs_dic = {}
for model_path in sorted(model_paths):
    print('loading checkpoint {}'.format(model_path))
    checkpoint = torch.load(model_path)
    # print('opt.arch',opt.arch)
    # print("checkpoint['arch']", checkpoint['arch'])
    # assert opt.arch == checkpoint['arch']
    epoch = checkpoint['epoch']
    if epoch in accs_dic:
        continue
    model.load_state_dict(checkpoint['state_dict'])

    if opt.n_val_samples != 1:
        recorder = test_one_split(model, test_loader)
        with open(os.path.join(opt.result_path, 'recorder'+opt.log_postfix+'.json'), 'w') as f:
            json.dump(recorder, f)
    else:
        # acc_l, pre_l, recall_l, cm_l = test_one(model, test_loader) #top5, 
        acc_l, pre_l, recall_l, cm_l, y_true, y_pred, distance = test_one(model, test_loader)
        # test_logger.log({
        #         'epoch': epoch,
        #         'top1': top1.avg,
        #         # 'top5': top5.avg,
        #         'precision':precisions.avg,
        #         'recall':recalls.avg,
        #         'confusion_matrix': confusions.conf.tolist()
        #     })
        test_logger.log({
            'epoch': epoch,
            'top1': acc_l,
            # 'top5': top5.avg,
            'precision':pre_l,
            'recall':recall_l,
            'confusion_matrix': cm_l
        })

        accs_dic[epoch] = round(float(acc_l[-1]), 3)
        # print('y_true=', y_true)
        # print('y_pred=', y_pred)
        # print('distance=', distance)
    

print(accs_dic)
print(sorted(accs_dic.items(), key=lambda item:item[1], reverse=True))
# plot_accuracy(accs_dic)
    # bs_name = '%s_%d_%s'%(opt.dataset, opt.n_classes, opt.modality)
    # class_names = get_classnames(opt.dataset)
    # np.set_printoptions(precision=2)

    # plot_confusion_matrix(confusions.conf, classes=class_names,
    #                       title='Confusion matrix without normalization',
    #                       save_name=os.path.join(opt.result_path, bs_name))

    # plot_confusion_matrix(confusions.conf, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix',
    #                       save_name=os.path.join(opt.result_path, '%s_norm'%bs_name))

    # print('-----Evaluation is finished------')
    # print('Overall Prec@1 {:.05f}% Prec@5 {:.05f}%'.format(top1.avg, top5.avg))
