import argparse
import os
import random
import shutil
import time
import datetime
import warnings
from os.path import join

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model.utils import load_pretrained_weights
from model.slstt import SLSTT
from datasets import ME_dataset

parser = argparse.ArgumentParser(description='MER Training')
parser.add_argument('-d', '--dataset', metavar='DATABASE', default='casme2',
                    help='name of dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='B_16_imagenet1k',
                    help='model architecture (default: B_16_imagenet1k)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--input-size', default=384, type=int,
                    help='input size')
parser.add_argument('--aggregation', default='LSTM', type=str, 
                    help='use Mean or LSTM Aggregation (default:LSTM)')
parser.add_argument('-s', '--sub-val', default=1, type=int, metavar='N',
                    help='the subject used to validate in LOSO (default: 1)')
parser.add_argument('--combined',  dest='combined', action='store_true',
                    help='if the label of dataset is the combined 3 classes. ')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='finetune model from pretrained model on other dataset (e.g. CK+)')
parser.add_argument('--modal-path', default='', type=str,
                    help='path to checkpoint modal to evaluate or finetune (default: none)')
parser.add_argument('--dir', default='examples/SDE', type=str, metavar='saving or evaluation model path',
                    help='dir to find trained slstt models (default: examples/SDE)')

best_acc1 = 0
best_f1 = 0

def trainer(loso=False, args=None):
    if not loso:
        args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    global best_acc1
    global best_f1

    if args.combined or "smic" in args.dataset:
        num_classes = 3
    elif "casme2" in args.dataset:
        num_classes = 5
    elif "samm" in args.dataset:
        num_classes = 5
    # elif args.dataset == 'ck+':
    #     num_classes = 8
        
    model = SLSTT(num_classes=num_classes, input_size=args.input_size, aggregation=args.aggregation)
    load_pretrained_weights(model.vit, load_fc=False, weights_path=args.arch+'.pth')

    print("=> using model '{}' (pretrained)".format(args.arch))



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1)

    #  load model from the pretrained model
    if args.finetune:
        if os.path.isfile(args.resume):
            print("=> loading pretrained model '{}'".format(args.pretrained_modal_path))
            checkpoint = torch.load(args.resume,map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            state_dict.pop('module.fc.weight')
            state_dict.pop('module.fc.bias')
            if not args.lstm:
                model.load_state_dict(state_dict, strict=False)
            else:
                model.vit.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()}, strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no pretrained model found at '{}'".format(args.resume))
        model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.Resize([args.input_size,args.input_size]),
                transforms.ToTensor(),
                normalize])
    if "com" not in args.dataset:
        train_dataset = ME_dataset(
            transform=trans,
            combined=args.combined,
            train=True,
            val_subject=args.sub_val,
            dataset=args.dataset)

        
        val_dataset = ME_dataset(
                transform=trans,
                combined=args.combined,
                train=False,
                val_subject=args.sub_val,
                dataset=args.dataset)
    else:
        val_casme2 = 0
        val_samm = 0
        val_smic = 0
        if "casme2" in args.dataset:
            val_casme2 = args.sub_val
        elif "smic" in args.dataset:
            val_smic = args.sub_val
        elif "samm" in args.dataset:
            val_samm = args.sub_val
        train_casme2 = ME_dataset(
                transform=trans,
                combined=args.combined,
                train=True,
                val_subject=val_casme2,
                dataset="casme2")
        train_samm = ME_dataset(
            transform=trans,
            combined=args.combined,
            train=True,
            val_subject=val_samm,
            dataset="samm")
        train_smic = ME_dataset(
            transform=trans,
            combined=args.combined,
            train=True,
            val_subject=val_smic,
            dataset="smic")
        train_dataset = torch.utils.data.ConcatDataset([train_casme2,train_samm,train_smic]) 
        val_casme2 = ME_dataset(
                transform=trans,
                combined=args.combined,
                train=False,
                val_subject=val_casme2,
                dataset="casme2")
        val_samm = ME_dataset(
            transform=trans,
            combined=args.combined,
            train=False,
            val_subject=val_samm,
            dataset="samm")
        val_smic = ME_dataset(
            transform=trans,
            combined=args.combined,
            train=False,
            val_subject=val_smic,
            dataset="smic")
        val_dataset = torch.utils.data.ConcatDataset([val_casme2,val_samm,val_smic])    


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        res, top2, f1 = validate(val_loader, model, criterion, args)
        with open('results.csv', 'a') as f:
            print(datetime.datetime.now(), ', sub'+str(args.sub_val).zfill(2)+', ', str(len(val_loader))+', ', res.item()+', ', top2.item()+', ', f1.item(), file=f)
        return

    for epoch in range(1,args.epochs+1):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, args)

        # evaluate on validation set
        acc1, _ , f1= validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        if args.combined:
            is_best = f1 >= best_f1
            best_f1 = max(f1, best_f1)
            if is_best:
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
        else:
            is_best = acc1 >= best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                is_best = f1 > best_f1
                best_f1 = max(f1, best_f1)

            save_checkpoint(args,{
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_f1': best_f1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.sub_val)

        if best_acc1 == 100:
            break

def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top2, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader, 1):
        # measure data loading time
        data_time.update(time.time() - end)
        outputs = model(inputs)
        # outputs = torch.stack(outputs)
        targets = torch.t(targets)[0].to(outputs.device)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(outputs, targets, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top2.update(acc2[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
    scheduler.step()
    return top1.avg


def validate(val_loader, model, criterion, args, loso=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top2,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        outputs_total = []
        targets_total = []

        for i, (inputs, targets) in enumerate(val_loader, 1):
            outputs = model(inputs)
            targets = torch.t(targets)[0].to(outputs.device)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(outputs, targets, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top2.update(acc2[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % len(val_loader) == 0:
            progress.print(i)

            outputs_total += outputs.cpu().numpy().tolist()   
            targets_total += targets.cpu().numpy().tolist()
        
        pred_total = [output.index(max(output)) for output in outputs_total]
        f1 = f1_score(targets_total, pred_total, average='macro')
        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f} F1@1 {f1:.3f}'
              .format(top1=top1, top2=top2, f1=f1))

    if loso:
        return outputs_total, targets_total
    return top1.avg, top2.avg, f1

def save_checkpoint(args, state, is_best, sub_val):
    if "casme2" in args.dataset:
        filename=join(args.dir,'checkpoint_casme2_sub'+str(sub_val).zfill(2)+'.pth.tar')
        torch.save(state, filename)
        if is_best:
            #torch.save(state, join(args.dir,'model_best_casme2_sub'+str(sub_val).zfill(2)+'.pth.tar'))
            shutil.copyfile(filename, join(args.dir,'model_best_casme2_sub'+str(sub_val).zfill(2)+'.pth.tar'))
    elif "samm" in args.dataset:
        filename=join(args.dir,'checkpoint_samm_'+str(sub_val).zfill(3)+'.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, join(args.dir,'model_best_samm_'+str(sub_val).zfill(3)+'.pth.tar'))
    elif "smic" in args.dataset:
        filename=join(args.dir,'checkpoint_smic_s'+str(sub_val)+'.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, join(args.dir,'model_best_smic_s'+str(sub_val)+'.pth.tar'))
    elif args.dataset == 'com-casme2':
        filename=join(args.dir,'checkpoint_com_casme2_sub'+str(sub_val).zfill(2)+'.pth.tar')
        torch.save(state, filename)
        if is_best:
            #torch.save(state, join(args.dir,'model_best_casme2_sub'+str(sub_val).zfill(2)+'.pth.tar'))
            shutil.copyfile(filename, join(args.dir,'model_best_com_casme2_sub'+str(sub_val).zfill(2)+'.pth.tar'))
    elif args.dataset == 'com-samm':
        filename=join(args.dir,'checkpoint_com_samm_'+str(sub_val).zfill(3)+'.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, join(args.dir,'model_best_com_samm_'+str(sub_val).zfill(3)+'.pth.tar'))
    elif args.dataset == 'com-smic':
        filename=join(args.dir,'checkpoint_com_smic_s'+str(sub_val)+'.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, join(args.dir,'model_best_com_smic_s'+str(sub_val)+'.pth.tar'))



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    trainer()

