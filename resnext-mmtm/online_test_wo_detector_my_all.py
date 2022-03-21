import os
import sys
import glob
import json
import pandas as pd
import csv
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
# from temporal_transforms import *
from target_transforms import ClassLabel
from dataset import get_online_data
from utils import  AverageMeter, LevenshteinDistance, Queue

import pdb
import numpy as np
import datetime


def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))


opt = parse_opts_online()


def load_models(opt):
    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.width_mult = opt.width_mult_clf
    opt.modality = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf
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
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_clf'+opt.log_postfix+'.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
#        assert opt.arch == checkpoint['arch']

        classifier.load_state_dict(checkpoint['state_dict'])

    print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return classifier

label_csv_path = os.path.join(opt.annotation_path.rsplit(os.sep, 1)[0], 'classIndAll.txt')
labels_data = pd.read_csv(label_csv_path, delimiter=' ', header=None)

classifier = load_models(opt)


if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)

spatial_transform = Compose([
    Scale(opt.sample_size),
    CenterCrop(opt.sample_size),
    ToTensor(opt.norm_value), norm_method
])

target_transform = ClassLabel()

## Get list of videos to test
if opt.dataset == 'egogesture':
    subject_list = ['Subject{:02d}'.format(i) for i in [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]]
    test_paths = []
    for subject in subject_list:
        for x in glob.glob(os.path.join(opt.video_path, subject, '*/*/rgb*')):
            test_paths.append(x)
elif opt.dataset == 'nvgesture':
    df = pd.read_csv(os.path.join(opt.video_path, 'nvgesture_test_correct_cvpr2016_v2.lst'), delimiter=' ', header=None)
    test_paths = []
    for x in df[0].values:
        test_paths.append(os.path.join(opt.video_path, x.replace('path:', ''), 'sk_color_all'))
elif opt.dataset == 'mousegesture':
    test_paths = []
    with open(os.path.join(opt.video_path, 'test_labels.txt')) as test_file:
        for line in test_file:
            video_name = line.split(',')[0]
            test_paths.append(os.path.join(opt.video_path, './'+opt.frames_dir, video_name+'_color_all'))
elif opt.dataset == 'hcigesture':
    test_paths = []
    with open(os.path.join(opt.video_path, 'test_labels.txt')) as test_file:
        for line in test_file:
            video_name = line.split(',')[0]
            test_paths.append(os.path.join(opt.video_path, './'+opt.frames_dir, video_name+'_color_all'))

print('Start Evaluation')
classifier.eval()


levenshtein_accuracies = AverageMeter()
videoidx = 0
results = {}
results_file = open(os.path.join(opt.result_path, 'results'+opt.log_postfix+'.txt'), "w")
ori_prediction_file = open(os.path.join(opt.result_path, 'ori_prediction'+opt.log_postfix+'.txt'), "w")
for path in test_paths[90:]:
    video_name = path.split(opt.frames_dir+'/')[1].replace('_color_all', '')
    ori_prediction_file.write(video_name+'\n')
    if opt.dataset == 'egogesture':
        opt.whole_path = os.path.join(*path.rsplit(os.sep, 4)[1:])
    elif opt.dataset == 'nvgesture':
        opt.whole_path = os.path.join(*path.rsplit(os.sep, 5)[1:])
    elif opt.dataset == 'mousegesture':
        opt.whole_path = os.path.join(*path.rsplit(os.sep, 3)[1:])
    elif opt.dataset == 'hcigesture':
        opt.whole_path = os.path.join(*path.rsplit(os.sep, 4)[1:]) # delete opt.video_path

    videoidx += 1
    active_index = 0
    passive_count = 0
    last_activate_frame = 0
    prev_gesture = opt.n_classes_clf-1  # None

    cum_sum = np.zeros(opt.n_classes_clf, )
    clf_selected_queue = np.zeros(opt.n_classes_clf, )
    myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)

    print('[{}/{}]============'.format(videoidx, len(test_paths)))
    print(video_name)
    opt.sample_duration = opt.sample_duration_clf
    # temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
    # temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
    test_data = get_online_data(
        opt, spatial_transform, None, target_transform)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)

    results[video_name] = {} # start: (end, label)
    # prev_best1 = opt.n_classes_clf
    dataset_len = len(test_loader.dataset)
    for i, (inputs, targets) in enumerate(test_loader):
        window_tail_frame = i * opt.stride_len + opt.sample_duration_clf
        if not opt.no_cuda:
            targets = targets.cuda()
        # ground_truth_array = np.zeros(opt.n_classes_clf + 1, )
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
            if opt.modality_clf == 'RGB':
                inputs_clf = inputs[:, :-1, :, :, :]
            elif opt.modality_clf == 'Depth':
                inputs_clf = inputs[:, -1, :, :, :].unsqueeze(1)
            elif opt.modality_clf == 'RGB-D':
                inputs_clf = inputs[:, :, :, :, :]
            # inputs_clf = torch.Tensor(inputs_clf.numpy()[:, :, ::2, :, :])
            outputs_clf = classifier(inputs_clf)
            if type(outputs_clf) == tuple:  # mmtnet
                outputs_clf = sum(list(outputs_clf))
            outputs_clf = F.softmax(outputs_clf, dim=1)
            outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )

            
            myqueue_clf.enqueue(outputs_clf.tolist())
            # passive_count = 0

            if opt.clf_strategy == 'raw':
                clf_selected_queue = outputs_clf
            elif opt.clf_strategy == 'median':
                clf_selected_queue = myqueue_clf.median
            elif opt.clf_strategy == 'ma':
                clf_selected_queue = myqueue_clf.ma
            elif opt.clf_strategy == 'ewma':
                clf_selected_queue = myqueue_clf.ewma

            prediction_clf = np.argmax(clf_selected_queue)
            predict_gesture = prediction_clf#labels_data[1][prediction_clf]
            ori_prediction_file.write('%d %d %f\n' % (window_tail_frame, predict_gesture, clf_selected_queue[prediction_clf]))
            # print('%d %d %f\n' % (window_tail_frame, predict_gesture, clf_selected_queue[prediction_clf]))
            
            if clf_selected_queue[prediction_clf] > opt.clf_threshold_final:
                if predict_gesture != prev_gesture: # and not (prev_gesture=='double-click' and predict_gesture=='click')
                    if passive_count >= opt.det_counter:  # activate a new gesture
                        active_index = 1
                        start_frame_index = window_tail_frame-opt.sample_duration_clf # window_head_frame
                        passive_count = 0
                        isAdded = False
                        prev_gesture = predict_gesture
                        last_activate_frame = window_tail_frame#num_frame
                    else:
                        passive_count += 1
                        # if active_index > 10 and prev_gesture != 'None':
                        #     cv2.putText(frame, prev_gesture, (200, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
                else: 
                    active_index += opt.stride_len
                    last_activate_frame = window_tail_frame#num_frame
                    if active_index > opt.active_threshold and (labels_data[1][prev_gesture] not in ['none', 'None']):# and not isAdded:
                        # print('predict', labels_data[1][prev_gesture], start_frame_index, last_activate_frame)
                        #results[video_name].append(((i * opt.stride_len) + opt.sample_duration_clf, prev_gesture))
                        #isAdded = True
                        results[video_name][start_frame_index] = (last_activate_frame, prev_gesture)
            else:
                # if the unconvinced frames are more than 3, deactivate the current gesture 
                if window_tail_frame-last_activate_frame > 2*opt.stride_len:
                    active_index = 0
                    prev_gesture = opt.n_classes_clf-1
                    # passive_count = 0
                    # last_activate_frame = 0

    result_line = video_name
    for s in results[video_name]:
        e, label  = results[video_name][s]
        result_line += ','+' '.join([str(s),str(e),str(label)])
    results_file.write(result_line+'\n')
    sys.stdout.flush()

results_file.close()
ori_prediction_file.close()
#     if opt.dataset == 'egogesture':
#         target_csv_path = os.path.join(opt.video_path,
#                                        'labels-final-revised1',
#                                        opt.whole_path.rsplit(os.sep, 2)[0],
#                                        'Group' + opt.whole_path[-1] + '.csv').replace('Subject', 'subject')
#         true_classes = []
#         with open(target_csv_path) as csvfile:
#             readCSV = csv.reader(csvfile, delimiter=',')
#             for row in readCSV:
#                 true_classes.append(int(row[0]) - 1)
#     elif opt.dataset == 'nvgesture':
#         true_classes = []
#         with open('./annotation_nvGesture/vallistall.txt') as csvfile:
#             readCSV = csv.reader(csvfile, delimiter=' ')
#             for row in readCSV:
#                 if row[0] == opt.whole_path:
#                     if row[1] != '26':
#                         true_classes.append(int(row[1]) - 1)
#     elif opt.dataset == 'mousegesture':
#         true_classes = []
#         with open(os.path.join(opt.annotation_path.rsplit(os.sep, 1)[0], 'vallistall.txt')) as csvfile:
#             readCSV = csv.reader(csvfile, delimiter=' ')
#             for row in readCSV:
#                 if row[0] == opt.whole_path:
#                     if row[1] != str(opt.n_classes+1):
#                         true_classes.append(int(row[1]) - 1)
#     elif opt.dataset == 'hcigesture':
#         true_classes = []
#         with open(os.path.join(opt.annotation_path.rsplit(os.sep, 1)[0], 'vallistall.txt')) as csvfile:
#             readCSV = csv.reader(csvfile, delimiter=' ')
#             for row in readCSV:
#                 if row[0] == opt.whole_path:
#                     if row[1] != len(labels_data[1]):#str(opt.n_classes+1):
#                         true_classes.append(int(row[1]) - 1)
#     if len(results[video_name]) != 0:
#         predicted = np.array(results[video_name])[:, 1]
#     else:
#         predicted = []
#     true_classes = np.array(true_classes)
#     levenshtein_distance = LevenshteinDistance(true_classes, predicted)
#     levenshtein_accuracy = 1 - (levenshtein_distance / len(true_classes))
#     if levenshtein_distance < 0:  # Distance cannot be less than 0
#         levenshtein_accuracies.update(0, len(true_classes))
#     else:
#         levenshtein_accuracies.update(levenshtein_accuracy, len(true_classes))

#     print('predicted classes: \t', predicted)
#     print('True classes :\t\t', true_classes)
#     print('Levenshtein Accuracy = {} ({})'.format(levenshtein_accuracies.val, levenshtein_accuracies.avg))

# print('Average Levenshtein Accuracy= {}'.format(levenshtein_accuracies.avg))

# print('-----Evaluation is finished------')
# with open(os.path.join(opt.result_path, 'online-test-wo-detector-my-'+opt.log_postfix+'.log'), "a") as myfile:
#     myfile.write("{}, {}, {}, {}, {}, {}".format(datetime.datetime.now(),
#                                     opt.resume_path_clf,
#                                     opt.model_clf,
#                                     opt.width_mult_clf,
#                                     opt.modality_clf,
#                                     levenshtein_accuracies.avg))
