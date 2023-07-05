import os
import argparse
from os.path import join

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, recall_score, classification_report

from model.utils import load_pretrained_weights
from model.slstt import SLSTT
from datasets import ME_dataset
from trainer import validate, trainer


parser = argparse.ArgumentParser(description='SLSTT')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dir', default='examples/SDE', type=str, metavar='saving or evaluation model path',
                    help='dir to find trained slstt models (default: examples/SDE)')
parser.add_argument('-m', '--mean', dest='slstt-mean', action='store_true',
                    help='use slstt-mean model')
parser.add_argument('--start-sub', default=1, type=int, metavar='N',
                    help='start from this subject (default: 1)')
parser.add_argument('-c','--combined', dest='combined', action='store_true',
                    help='if the label of dataset is the combined 3 classes. ')
parser.add_argument('-d', '--dataset', metavar='DATABASE', default='casme2',
                    help='name of dataset (default: casme2)')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='finetune model from pretrained model on other dataset (e.g. CK+)')

parser.add_argument('-a', '--arch', metavar='ARCH', default='B_16_imagenet1k',
                    help='model architecture (default: B_16_imagenet1k)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--input_size', default=384, type=int,
                    help='input image size (default: 384)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--aggregation', default='LSTM', type=str, 
                    help='use Mean or LSTM Aggregation (default:LSTM)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

def main():
    args = parser.parse_args()
    # torch.backends.cudnn.benchmark = True
    
    if "casme2" in args.dataset:
        sub_list = range(1,27)
    elif "samm" in args.dataset:
        sub_list = [6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,31,32,33,34,35,36,37]
    elif "smic" in args.dataset:
        sub_list = [1,2,3,4,5,6,8,9,11,12,13,14,15,18,19,20]

    if args.evaluate:
        if args.combined or "smic" in args.dataset:
            num_classes = 3
        elif "casme2" in args.dataset:
            num_classes = 5
        elif "samm" in args.dataset:
            num_classes = 5
        model = SLSTT(num_classes=num_classes, input_size=args.input_size, aggregation=args.aggregation)
        load_pretrained_weights(model.vit, load_fc=False, weights_path=args.arch+'.pth')
        if torch.cuda.is_available():
            device = list(range(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model, device_ids=device, output_device=device[-1]).cuda()
            cudnn.benchmark = True
            criterion = nn.CrossEntropyLoss().cuda()
            print("Device: ",device)
            if args.dataset == 'com':
                cde(args,model,criterion)
            else:
                loso(model, criterion, args, sub_list)
        else:
            print("CUDA is not available!")
    else:
        for i in sub_list[sub_list.index(args.start_sub):]:
            print("train start with leave subject {} out".format(i))
            args.sub_val = i
            trainer(True,args)

def loso(model, criterion, args, sub_list):
    outputs_loso, targets_loso = [], []
    for i in sub_list[sub_list.index(args.start_sub):]:
        if "smic" in args.dataset:
            path = join(args.dir, 'model_best_smic_s'+str(i)+'.pth.tar')
        elif "casme2" in args.dataset:
            path = join(args.dir, 'model_best_casme2_sub'+str(i).zfill(2)+'.pth.tar')
        elif "samm" in args.dataset:
            path = join(args.dir, 'model_best_samm_'+str(i).zfill(3)+'.pth.tar')
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(path))
        else:
            print("=> no checkpoint found at '{}'".format(path))
            continue
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([transforms.Resize([args.input_size,args.input_size]),
                transforms.ToTensor(),
                normalize])
        val_dataset = ME_dataset(
            transform=trans,
            combined=args.combined,
            train=False,
            val_subject=i,
            dataset=args.dataset)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        outputs, targets = validate(val_loader, model, criterion, args, loso=True)
        outputs_loso += outputs
        targets_loso += targets
    cor, acc, f1, uar = res(outputs_loso, targets_loso, topk=(1, 2))
    print('LOSO: Accuracy: {:d}/{} ({:.3f}%), F1-score(UF1): {:.3f}, UAR: {:.3f}\n Top2 Accuracy: {:d}/{} ({:.3f}%), F1-score(UF1): {:.3f}, UAR: {:.3f}\n'.format(
        int(cor[0]), len(targets_loso), acc[0], f1[0], uar[0], int(cor[1]), len(targets_loso), acc[1], f1[1], uar[1]))
    
    #     plot_confusion_matrix(outputs_S, targets_S,
    #                                 labels=[i for i in range(len(classes))],
    #                                 display_labels=classes,
    #                                 cmap=plt.cm.Blues,
    #                                 normalize='true')
    #     plt.savefig(pth+'_confusionmatrix_spatial_'+str(epoch)+'.png')

def cde(args,model,criterion):
    outputs_loso, targets_loso = [], []
    sub_casme2 = range(1,27)
    sub_samm = [6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,31,32,33,34,35,36,37]
    sub_smic = [1,2,3,4,5,6,8,9,11,12,13,14,15,18,19,20]
    for sub in ["sub_casme2","sub_samm","sub_smic"]:
        val_casme2 = 0
        val_samm = 0
        val_smic = 0
        for i in eval(sub):
            if sub == 'sub_casme2':
                path = join(args.dir, 'model_best_com_casme2_sub'+str(i).zfill(2)+'.pth.tar')
                val_casme2 = i
            elif sub == 'sub_samm':
                path = join(args.dir, 'model_best_com_samm_'+str(i).zfill(3)+'.pth.tar')
                val_samm = i
            elif sub == 'sub_smic':
                path = join(args.dir, 'model_best_com_smic_s'+str(i)+'.pth.tar')
                val_smic = i
            if os.path.isfile(path):
                checkpoint = torch.load(path)
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}'".format(path))
            else:
                print("=> no checkpoint found at '{}'".format(path))
                continue
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            trans = transforms.Compose([transforms.Resize([args.input_size,args.input_size]),
                    transforms.ToTensor(),
                    normalize])
            casme2 = ME_dataset(
                transform=trans,
                combined=args.combined,
                train=False,
                val_subject=val_casme2,
                dataset="casme2")
            samm = ME_dataset(
                transform=trans,
                combined=args.combined,
                train=False,
                val_subject=val_samm,
                dataset="samm")
            smic = ME_dataset(
                transform=trans,
                combined=args.combined,
                train=False,
                val_subject=val_smic,
                dataset="smic")
            val_dataset = torch.utils.data.ConcatDataset([casme2,samm,smic])    
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            outputs, targets = validate(val_loader, model, criterion, args, loso=True)
            outputs_loso += outputs
            targets_loso += targets
    cor, acc, f1, uar = res(outputs_loso, targets_loso, topk=(1, 2))
    print('LOSO: Accuracy: {:d}/{} ({:.3f}%), F1-score(UF1): {:.3f}, UAR: {:.3f}\n Top2 Accuracy: {:d}/{} ({:.3f}%), F1-score(UF1): {:.3f}, UAR: {:.3f}\n'.format(
        int(cor[0]), len(targets_loso), acc[0], f1[0], uar[0], int(cor[1]), len(targets_loso), acc[1], f1[1], uar[1]))
    

def res(output, target, topk=(1,)):
    """Computes the results over the k top predictions for the specified values of k"""
    with torch.no_grad():
        output = torch.Tensor(output)
        target = torch.LongTensor(target)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        cor, acc, f1, uar = [], [], [], []
        for k in topk:
            cor_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            cor.append(cor_k.item())
            acc.append(cor_k.item()/batch_size*100)
            correct_k = correct[:k].sum(0)
            pred_k = torch.where(correct_k==0, pred[0], target)
            print(classification_report(target.numpy().tolist(), pred_k.numpy().tolist()))
            f1.append(f1_score(target.numpy().tolist(), pred_k.numpy().tolist(), average='macro'))
            uar.append(recall_score(target.numpy().tolist(), pred_k.numpy().tolist(), average='macro'))
        return cor, acc, f1, uar

if __name__ == '__main__':
    main()
