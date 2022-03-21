import csv
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import shutil
import numpy as np

import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import cv2


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ConfusionMeter(object):
    """Maintains a confusion matrix """

    def __init__(self, k):
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.cur_conf = np.ndarray((k, k), dtype=np.int32)
        self.k = k
        self.reset()

    def reset(self):
        self.cur_conf.fill(0)
        self.conf.fill(0)

    def update(self, cur_conf):
        self.cur_conf = cur_conf
        self.conf += cur_conf
        

class Logger(object):

    def __init__(self, path, header, open_mode='w'):
        self.log_file = open(path, open_mode)
        self.logger = csv.writer(self.log_file, delimiter='\t')
        if open_mode == 'w':
            self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()



class Queue:
    # Constructor creates a list
    def __init__(self, max_size, n_classes):
        self.queue = list(np.zeros((max_size, n_classes), dtype=float).tolist())
        self.max_size = max_size
        self.median = None
        self.ma = None
        self.ewma = None

    # Adding elements to queue
    def enqueue(self, data):
        self.queue.insert(0, data)
        self.median = self._median()
        self.ma = self._ma()
        self.ewma = self._ewma()
        return True

    # Removing the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        return ("Queue Empty!")

    # Getting the size of the queue
    def size(self):
        return len(self.queue)

    # printing the elements of the queue
    def printQueue(self):
        return self.queue

    # Average
    def _ma(self):
        return np.array(self.queue[:self.max_size]).mean(axis=0)

    # Median
    def _median(self):
        return np.median(np.array(self.queue[:self.max_size]), axis=0)

    # Exponential average
    def _ewma(self):
        weights = np.exp(np.linspace(-1., 0., self.max_size))
        weights /= weights.sum()
        average = weights.reshape(1, self.max_size).dot(np.array(self.queue[:self.max_size]))
        return average.reshape(average.shape[1], )


class HandCache:
    def __init__(self, max_size):
        self.hand_dict = {}
        self.max_size = max_size
        self.max_hand_id = -1

    def update(self, bboxes, frame, cur_frame_index): # bboxes: [[l,t,r,b], ...]
        for bbox in bboxes:
            for hand_id in self.hand_dict:
                crop_bbox = self.hand_dict[hand_id]['crop_bbox']
                if bbox[0]>=crop_bbox[0] and bbox[1]>=crop_bbox[1] and bbox[2]<=crop_bbox[2] and bbox[3]<=crop_bbox[3]: # bbox in crop_bbox
                    self.add(hand_id, frame[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]], cur_frame_index)
                    return
            self.add_new(bbox, frame, cur_frame_index)
        self.delete_inactivate_hand(cur_frame_index)

    def add(self, hand_id, crop_region, cur_frame_index):
        if len(self.hand_dict[hand_id]['crop_regions']) == self.max_size:
            del self.hand_dict[hand_id]['crop_regions'][0]

        self.hand_dict[hand_id]['crop_regions'].append(crop_region)
        self.hand_dict[hand_id]['latest_frame_index'] = cur_frame_index
        
    def add_new(self, bbox, frame, cur_frame_index):
        def get_crop_region(bbox, width, height):  # bbox format???
            x0, y0, w0, h0 = bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height
            w0 = max(w0, 50)  # at the start of downward gesture, the height of the hand is very small
            h0 = max(h0, 50)
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
        self.max_hand_id += 1
        crop_bbox = get_crop_region(bbox, frame.shape[1], frame.shape[0])
        self.hand_dict[self.max_hand_id]['crop_regions'] = [frame[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]]
        self.hand_dict[self.max_hand_id]['crop_bbox'] = crop_bbox
        self.hand_dict[self.max_hand_id]['latest_frame_index'] = cur_frame_index


    def delete_inactivate_hand(self, cur_frame_index):
        for hand_id in self.hand_dict:
            if cur_frame_index-self.hand_dict[hand_id]['latest_frame_index'] > 5:
                del self.hand_dict[hand_id]

    def get_all_clips(self, sample_duration): # no need to consider the case that the length less than sample_duration, temporal_transform will process this case
        clips = []
        for hand_id in sorted(self.hand_dict.keys()):
            clips.append(self.hand_dict[hand_id]['crop_regions'])



def LevenshteinDistance(a, b):
    # This is a straightforward implementation of a well-known algorithm, and thus
    # probably shouldn't be covered by copyright to begin with. But in case it is,
    # the author (Magnus Lie Hetland) has, to the extent possible under law,
    # dedicated all copyright and related and neighboring rights to this software
    # to the public domain worldwide, by distributing it under the CC0 license,
    # version 1.0. This software is distributed without any warranty. For more
    # information, see <http://creativecommons.org/publicdomain/zero/1.0>
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)
    if current[n]<0:
        return 0
    else:
        return current[n]


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(float(correct_k.mul_(100.0 / batch_size)))
    return res


def calculate_precision(outputs, targets, average='macro'):

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return  precision_score(targets.view(-1).cpu(), pred.view(-1).cpu(), average=average)


def calculate_recall(outputs, targets, average='macro'):

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return  recall_score(targets.view(-1).cpu(), pred.view(-1).cpu(), average=average)

def calculate_confusion_matrix(outputs, targets, labels):
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return confusion_matrix(targets.view(-1).cpu(), pred.view(-1).cpu(), labels=labels)

def save_checkpoint(state, is_best, opt):
    if opt.save_all:
        save_name = '%s/%s_epoch_%03d.pth' % (opt.result_path, opt.store_name, state['epoch'])
    else:
        save_name = '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name)
    
    torch.save(state, save_name)
    if is_best:
        shutil.copyfile(save_name,'%s/%s_best.pth' % (opt.result_path, opt.store_name))

def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    optimizer_l = optimizer if type(optimizer)==list else [optimizer]
    for optimizer in optimizer_l:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_new
            #param_group['lr'] = opt.learning_rate

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          save_name='draw/cm',
                          cmap=plt.cm.Blues):
    """
    This,function,prints,and,plots,the,confusion,matrix.
    Normalization,can,be,applied,by,setting `normalize=True`.
    """
    plt.figure(figsize=(13,11)).clf()#

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix without normalization')

    print(cm)

    h = plt.imshow(cm, interpolation='nearest', aspect='auto', cmap=cmap)
    plt.title(title)
    plt.colorbar(h)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='vertical', fontsize=10)#, fontsize=8, rotation=45
    plt.yticks(tick_marks, classes, fontsize=10)
    # plt.margins(0.2)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)

    plt.savefig(save_name, bbox_inches='tight')

def get_classnames(dataset):
    if dataset == 'nvgesture':
        return ['move_hand_left', 'move_hand_right', 'move_hand_up', 'move_hand_down', 
              'move_two_fingers_left', 'move_two_fingers_right', 'move_two_fingers_up', 'move_two_fingers_down',
              'click_index_finger', 'call_someone', 'open_hand', 'shaking_hand', 'show_index_finger', 'show_two_fingers',
              'show_three_fingers', 'push_hand_up', 'push_hand_down', 'push_hand_out', 'push_hand_in', 'rotate_fingers_CW',
              'rotate_fingers_CCW', 'push_two_fingers_away', 'close_hand_two_times', 'thumb_up', 'okay']
    if dataset == 'egogesture':
        class_file = pd.read_table('annotation_EgoGesture/classIndAllbutNone.txt', sep=' ', header=None)
        return [class_name for class_name in class_file[1]]
    if dataset == 'hcigesture':
        class_file = pd.read_table('annotation_hciGesture/classIndAllbutNone.txt', sep=' ', header=None)
        return [class_name for class_name in class_file[1]]

    raise RuntimeError('Unrecognized dataset in utils get_classnames()')


def get_location(point):
    # x1, y1, x2, y2
    points = {
        'l1': (700, 250, 1200, 650),
        'l2': (700, 300, 1100, 600),
        'l3': (700, 300, 1000, 500),
        'r1': (200, 300, 500, 600),
        'r2': (250, 300, 550, 550),
        'r3': (350, 300, 560, 500)
    }
    return points[point]