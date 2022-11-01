import os
import cv2
import torch
# import argparse
from torch.nn import functional as F
from RIFE.model.RIFE_HD import Model
import warnings
warnings.filterwarnings("ignore")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# parser.add_argument('--img', dest='img', nargs=2, required=True)
# parser.add_argument('--exp', default=4, type=int)
# args = parser.parse_args()

def interpolate(img0, img1):
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    model = Model()
    model.load_model('PyTorch_ViT/RIFE/train_log', -1)
    model.eval()
    model.device()

    with torch.no_grad():
        img_0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img_1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        n, c, h, w = img_0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img_0 = F.pad(img_0, padding)
        img_1 = F.pad(img_1, padding)

        img_mid = model.inference(img_0, img_1)
        img_mid = (img_mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    return img_mid

