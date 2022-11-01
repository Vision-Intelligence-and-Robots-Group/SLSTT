import os
import argparse

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import f1_score, recall_score

from pytorch_pretrained_vit import ViT, load_pretrained_weights
from datasets import ME_dataset
from main import validate


parser = argparse.ArgumentParser(description='LOSO MER')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-sub', default=1, type=int, metavar='N',
                    help='start from this subject (default: 1)')
parser.add_argument('--combined', dest='combined', action='store_true',
                    help='if the label of dataset is the combined 3 classes. ')
parser.add_argument('-d', '--dataset', metavar='DATABASE', default='casme2',
                    help='name of dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='B_16_imagenet1k',
                    help='model architecture (default: B_16_imagenet1k)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='use pre-trained model to finetune')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--num-images', default=11, type=int,
                    help='the number of images in each sample. ')

def main():
    args = parser.parse_args()
    args.evaluate = True
    if args.dataset == 'casme2' or args.dataset == 'casme2-EVM':
        sub_list = range(1,27)
    elif args.dataset == 'samm' or args.dataset == 'samm-EVM':
        sub_list = [6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,31,32,33,34,35,36,37]
    elif args.dataset == 'smic' or args.dataset == 'smic-EVM':
        sub_list = [1,2,3,4,5,6,8,9,11,12,13,14,15,18,19,20]

    if args.evaluate:
        if args.combined or args.dataset == 'smic' or args.dataset == 'smic-EVM':
            num_classes = 3
        elif args.dataset == 'casme2' or args.dataset == 'casme2-EVM':
            num_classes = 5
        elif args.dataset == 'samm' or args.dataset == 'samm-EVM':
            num_classes = 5


        model = ViT(args.arch, pretrained=args.pretrained, num_classes=num_classes, image_size=args.image_size)        
        load_pretrained_weights(model, load_fc=False, weights_path='./jax_to_pytorch/weights/'+args.arch+'.pth')

        device = list(range(torch.cuda.device_count()))
        #print(device)
        #model = torch.nn.DataParallel(model, device_ids=device, output_device=device[-1]).cuda()
        model = model.cuda()
        cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss().cuda()
        loso(model, criterion, args, sub_list)
    else:
        for i in sub_list[sub_list.index(args.start_sub):]:
            print("train start without subject "+str(i))
            os.system("python3 PyTorch_ViT/main.py --vit -d casme2 --epochs 40 --lr 0.001 --dir pth-001-4 --gpu 0 --finetune --resume ~/MER/ViT_MER/PyTorch_ViT/jax_to_pytorch/weights/model_best_ck+0.00005.pth.tar --num-images "+str(args.num_images)+" --batch-size "+str(args.batch_size)+" -s "+str(i))
            # os.system("/home/hongxiaopeng/zlf/environment/python3/bin/python3 PyTorch_ViT/main.py --vit -d smic --epochs 40 --lr 0.001 --dir pth-001-4/casme2 --finetune --resume /home/hongxiaopeng/zlf/MER/ViT_MER/pth/model_best_sub15.pth.tar --num-images "+str(args.num_images)+" --batch-size "+str(args.batch_size)+" -s "+str(i))

def loso(model, criterion, args, sub_list):
    outputs_loso, targets_loso = [], []
    for i in sub_list[sub_list.index(args.start_sub):]:
        if args.dataset == 'casme2' or args.dataset == 'casme2-EVM':
            path = './model_best_casme2_sub'+str(i).zfill(2)+'.pth.tar'
        elif args.dataset == 'samm' or args.dataset == 'samm-EVM':
            path = 'pth-0005-4/model_best_samm_'+str(i).zfill(3)+'.pth.tar'
        elif args.dataset == 'smic' or args.dataset == 'smic-EVM':
            path = 'pth-001-4/casme2/model_best_smic_s'+str(i)+'.pth.tar'
        #path = 'pth/model_best_sub'+str(i).zfill(2)+'.pth.tar'
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path))
            continue
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([transforms.Resize([args.image_size,args.image_size]),
                transforms.ToTensor(),
                normalize])
        if args.num_images == 11:
            val_dataset = ME_dataset(
	        transform=trans,
	        combined=args.combined,
	        train=False,
	        val_subject=i,
                datadir='data-11',
	        dataset=args.dataset,
	        Spatial=False,
	        img_num=args.num_images)
        elif args.num_images == 100:
            val_dataset = ME_dataset(
	        transform=trans,
	        combined=args.combined,
	        train=False,
	        val_subject=i,
                datadir='data',
	        dataset=args.dataset,
	        Spatial=False,
	        img_num=args.num_images)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        outputs, targets = validate(val_loader, model, criterion, args, loso=True)
        outputs_loso += outputs
        targets_loso += targets
    cor, acc, f1, uar = res(outputs_loso, targets_loso, topk=(1, 2))
    # f1 = f1_score(targets_loso, outputs_loso, average='macro')
    print('LOSO: Accuracy: {:d}/{} ({:.3f}%), F1-score(UF1): {:.3f}, UAR: {:.3f}\n Top2 Accuracy: {:d}/{} ({:.3f}%), F1-score(UF1): {:.3f}, UAR: {:.3f}\n'.format(
        int(cor[0]), len(targets_loso), acc[0], f1[0], uar[0], int(cor[1]), len(targets_loso), acc[1], f1[1], uar[1]))
    
    #     plot_confusion_matrix(outputs_S, targets_S,
    #                                 labels=[i for i in range(len(classes))],
    #                                 display_labels=classes,
    #                                 cmap=plt.cm.Blues,
    #                                 normalize='true')
    #     plt.savefig(pth+'_confusionmatrix_spatial_'+str(epoch)+'.png')

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
            f1.append(f1_score(target.numpy().tolist(), pred_k.numpy().tolist(), average='macro'))
            uar.append(recall_score(target.numpy().tolist(), pred_k.numpy().tolist(), average='macro'))
        return cor, acc, f1, uar

if __name__ == '__main__':
    main()
