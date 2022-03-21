import torch
from torch.autograd import Variable
import time
import sys

from utils import *


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_l = [AverageMeter()]
    accuracies_l = [AverageMeter()]
    precisions_l = [AverageMeter()]
    recalls_l = [AverageMeter()]
    confusions_l = [ConfusionMeter(opt.n_classes)]

    printer = 100 if opt.dataset == 'jester' else 10

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()
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
        
        while len(losses_l)<len(outputs_l):
            losses_l.append(AverageMeter())
            accuracies_l.append(AverageMeter())
            precisions_l.append(AverageMeter())
            recalls_l.append(AverageMeter())
            confusions_l.append(ConfusionMeter(opt.n_classes))

        for c, outputs in enumerate(outputs_l):
            loss = criterion(outputs, targets)
            # prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
            # top1.update(prec1, inputs.size(0))
            # top5.update(prec5, inputs.size(0))
            acc = calculate_accuracy(outputs.data, targets.data)[0]
            precision = calculate_precision(outputs, targets)
            recall = calculate_recall(outputs,targets)
            confusion_m = calculate_confusion_matrix(outputs, targets, labels=range(opt.n_classes))

            losses_l[c].update(loss.data, inputs.size(0))
            accuracies_l[c].update(acc, inputs.size(0))
            precisions_l[c].update(precision, inputs.size(0))
            recalls_l[c].update(recall,inputs.size(0))
            confusions_l[c].update(confusion_m)

            c += 1


        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # if i % printer ==0:
        #   print('Epoch: [{0}][{1}/{2}]\t'
        #       'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
        #       'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #       # 'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
        #       # 'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'
        #       'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
        #       'Precision {precision.val:.3f}({precision.avg:.3f})\t'
        #       'Recall {recall.val:.3f}({recall.avg:.3f})'.format(
        #           epoch,
        #           i + 1,
        #           len(data_loader),
        #           batch_time=batch_time,
        #           data_time=data_time,
        #           loss=losses,
        #           # top1=top1,
        #           # top5=top5
        #           acc=accuracies,
        #           precision=precisions,
        #           recall=recalls,
        #           ))
          # print('batch confusion matrix:')
          # print(confusions.cur_conf)
    # print('total confusion matrix:')
    # print(confusions.conf)
    # sys.stdout.flush()

    # logger.log({'epoch': epoch,
    #             'loss': losses.avg.item(),
    #             # 'prec1': top1.avg.item(),
    #             # 'prec5': top5.avg.item(),
    #             'acc': accuracies.avg.item(),
    #             'precision':precisions.avg.item(),
    #             'recall':recalls.avg.item(),
    #             'confusion_matrix':confusions.conf.tolist()})

    loss_l = [losses.avg.item() for losses in losses_l]
    acc_l = [accuracies.avg for accuracies in accuracies_l]
    pre_l = [precisions.avg for precisions in precisions_l]
    recall_l = [recalls.avg for recalls in recalls_l]
    cm_l = [confusions.conf.tolist() for confusions in confusions_l]
    # total_cm = cm_l[0]
    # for cm in cm_l[1:]:
    #     total_cm += cm
    # cm_l.append(total_cm)

    # for c in range(len(cm_l)):
    #     cm_l[c] = cm_l[c].tolist()


    logger.log({
        'epoch': epoch,
        'loss': loss_l,
        'acc': acc_l,
        'precision': pre_l,
        'recall': recall_l,
        # 'prec1': top1.avg.item(),
        # 'prec5': top5.avg.item(),
        'confusion_matrix': cm_l
    })

    return loss_l[-1], acc_l[-1], pre_l[-1], recall_l[-1], cm_l[-1]
