import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from datasets import ME_dataset

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.Resize([384,384]),
        transforms.ToTensor(),
        normalize])
train_sampler = None
dataset = ME_dataset(
    transform=trans,
    train=False,
    val_subject=1,
    dataset='smic',
    combined=True,
    Spatial=False,
    img_num = 100)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=(train_sampler is None),
    num_workers=4, pin_memory=True, sampler=train_sampler)

# with torch.no_grad():
#     for i, (images, targets) in enumerate(loader, 1):
#         print(images,targets)

