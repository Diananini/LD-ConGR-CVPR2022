import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import *


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, writer):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    # accuracies = AverageMeter()
    # precisions = AverageMeter()
    # recalls = AverageMeter()
    # confusions = ConfusionMeter(opt.n_classes)

    if type(optimizer) != list:
        optimizer_l = [optimizer]
    else:
        optimizer_l = optimizer
    len_optmizer = len(optimizer_l)
    losses_l = [AverageMeter() for _ in range(len_optmizer)]
    accuracies_l = [AverageMeter() for _ in range(len_optmizer+1)]
    precisions_l = [AverageMeter() for _ in range(len_optmizer+1)]
    recalls_l = [AverageMeter() for _ in range(len_optmizer+1)]
    confusions_l = [ConfusionMeter(opt.n_classes) for _ in range(len_optmizer+1)]
        
    # top1 = AverageMeter()
    # top5 = AverageMeter()
    
    printer = 100 if opt.dataset == 'jester' else 10

    end_time = time.time()
    for i, (inputs, targets, _) in enumerate(data_loader):
        iters = (epoch - 1) * len(data_loader) + (i + 1)
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()
            # inputs = inputs.cuda()
        inputs = Variable(inputs)
        targets = Variable(targets)

        batch_size = inputs.size(0)

        c = 0
        loss_temp_l = []
        for losses, accuracies, precisions, recalls, confusions, optimizer in \
                zip(losses_l, accuracies_l, precisions_l, recalls_l, confusions_l, optimizer_l):
            
            outputs_l = model(inputs)
            if type(outputs_l) == tuple:
                outputs = outputs_l[c]
                if c == 0:
                    outputs_fusion = sum(outputs_l)
                    acc = calculate_accuracy(outputs_fusion.data, targets.data)[0]
                    precision = calculate_precision(outputs_fusion, targets)
                    recall = calculate_recall(outputs_fusion,targets)
                    confusion_m = calculate_confusion_matrix(outputs, targets, labels=range(opt.n_classes))
                    accuracies_l[-1].update(acc, batch_size)
                    precisions_l[-1].update(precision, batch_size)
                    recalls_l[-1].update(recall, batch_size)
                    confusions_l[-1].update(confusion_m)
            else:
                outputs = outputs_l

            # print('targets.shape', targets.shape)
            # print('outputs.shape', outputs.shape)
            loss = criterion(outputs, targets)

            writer.add_scalar('loss%d_train'%c, loss.data, iters)

            acc = calculate_accuracy(outputs.data, targets.data)[0]
            precision = calculate_precision(outputs, targets)
            recall = calculate_recall(outputs,targets)
            confusion_m = calculate_confusion_matrix(outputs, targets, labels=range(opt.n_classes))

            losses.update(loss.data, batch_size)
            # prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
            # top1.update(prec1, inputs.size(0))
            # top5.update(prec5, inputs.size(0))
            accuracies.update(acc, batch_size)
            precisions.update(precision, batch_size)
            recalls.update(recall, batch_size)
            confusions.update(confusion_m)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            c += 1


        batch_time.update(time.time() - end_time)
        end_time = time.time()

        loss_l = [losses.val.item() for losses in losses_l]
        acc_l = [accuracies.val for accuracies in accuracies_l]
        pre_l = [precisions.val for precisions in precisions_l]
        recall_l = [recalls.val for recalls in recalls_l]
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': iters,
            'loss': loss_l+[sum(loss_l)/len(loss_l)],
            # 'prec1': top1.val.item(),
            # 'prec5': top5.val.item(),
            'acc': acc_l,
            'precision': pre_l,
            'recall': recall_l,
            'lr': optimizer.param_groups[0]['lr']
        })
            
    loss_l = [losses.avg.item() for losses in losses_l]
    acc_l = [accuracies.avg for accuracies in accuracies_l]
    pre_l = [precisions.avg for precisions in precisions_l]
    recall_l = [recalls.avg for recalls in recalls_l]
    cm_l = [confusions.conf.tolist() for confusions in confusions_l]
        
    epoch_logger.log({
        'epoch': epoch,
        'loss': loss_l+[sum(loss_l)/len(loss_l)],
        'acc': acc_l,
        'precision': pre_l,
        'recall': recall_l,
        # 'prec1': top1.avg.item(),
        # 'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr'],
        'confusion_matrix': cm_l
    })

    #if epoch % opt.checkpoint == 0:
    #    save_file_path = os.path.join(opt.result_path,
    #                                  'save_{}.pth'.format(epoch))
    #    states = {
    #        'epoch': epoch + 1,
    #        'arch': opt.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #    }
    #    torch.save(states, save_file_path)
