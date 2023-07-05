import torch
import torchvision.transforms as transforms

import os
from os.path import join
import glob
import cv2
import numpy as np
import pandas as pd
from math import floor

from PIL import Image
import torch.utils.data as data
from RIFE.interpolate_img import interpolate

class ME_dataset(data.Dataset):
    def __init__(self, transform=None, train=True, val_subject=1, dataset='smic', datadir='inputs', combined = False):
        self.transform = transform
        self.val_subject = val_subject
        self.train = train
        self.dataset = dataset
        self.data_folder = join(datadir,dataset,'LOF')
        self.labels = ('negative','positive','surprise')
        if combined:
            data_gt = pd.read_csv(join(datadir,'combined_3class_gt.csv'), header=None, names=['Dataset','Subject','Filename','Class'])
        elif dataset == 'casme2':
            self.labels = ('disgust', 'happiness', 'repression', 'surprise', 'others')
        elif dataset == 'samm':
            self.labels = ('Other', 'Anger', 'Contempt', 'Happiness', 'Surprise')
        # elif dataset == 'ck+':
        #     self.labels = ('neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise')
        #     self.ftype = '.png'
        
        data_info = pd.read_csv(join(datadir,dataset,'coding.csv'))
        # label indexing, {'negative': array(0}, ...}
        self.label_index = {label : i for i, label in enumerate(self.labels)}
        self.subject_num = data_info.Subject.nunique()
        if combined:
            self.subject_num = data_gt.Subject.nunique()
        image_list = glob.glob(join(self.data_folder,'*','*','*'+'.jpg'), recursive=True)
        if self.dataset == 'smic':
            image_list = glob.glob(join(self.data_folder,'*','*','*','*'+'.bmp'), recursive=True)
        
        self.train_image = image_list.copy()
        self.val_image = []
        self.val_label = {}
        self.val_images = {}
        self.train_images = {}
        self.train_label = {}
        val_imgs = {}
        train_imgs = {}

        if self.dataset == 'casme2':
            for image in image_list:
                sub = data_info[data_info['Subject'].isin([int(image.split('/')[-3][3:])])]
                ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                if combined:
                    sub = data_gt[data_gt['Subject'].isin([image.split('/')[-3]])]
                    ep = sub[sub['Filename'].isin([image.split('/')[-2]])]

                if sub.empty or ep.empty:
                    self.train_image.remove(image)
                elif str(ep.Subject.values[0]) in [str(self.val_subject), 'sub'+str(self.val_subject).zfill(2)]:
                    if combined:
                        self.val_label[image] = ep.Class.values[0]
                        if ep.Subject.values[0]+ep.Filename.values[0] not in val_imgs.keys():
                            val_imgs[ep.Subject.values[0]+ep.Filename.values[0]] = []
                        val_imgs[ep.Subject.values[0]+ep.Filename.values[0]].append(image)
                    elif ep.Emotion.values[0] in self.labels:
                        self.val_label[image] = self.label_index[ep.Emotion.values[0]]
                        if 'sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0] not in val_imgs.keys():
                            val_imgs['sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0]] = []
                        val_imgs['sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0]].append(image)
                    self.train_image.remove(image)
                else:
                    if combined:
                        self.train_label[image] = ep.Class.values[0]
                        if ep.Subject.values[0]+ep.Filename.values[0] not in train_imgs.keys():
                            train_imgs[ep.Subject.values[0]+ep.Filename.values[0]] = []
                        train_imgs[ep.Subject.values[0]+ep.Filename.values[0]].append(image)
                    elif ep.Emotion.values[0] in self.labels:
                        self.train_label[image] = self.label_index[ep.Emotion.values[0]]
                        if 'sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0] not in train_imgs.keys():
                            train_imgs['sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0]] = []
                        train_imgs['sub'+str(ep.Subject.values[0]).zfill(2)+ep.Filename.values[0]].append(image)
                    else:
                        self.train_image.remove(image)
        elif self.dataset == 'samm':
            for image in image_list:
                    sub = data_info[data_info['Subject'].isin([int(image.split('/')[-3])])]
                    ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                    if combined:
                        sub = data_gt[data_gt['Subject'].isin([str(int(image.split('/')[-3]))])]
                        ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                    
                    if sub.empty or ep.empty:
                        self.train_image.remove(image)

                    elif str(ep.Subject.values[0]) in [str(self.val_subject), str(self.val_subject).zfill(3)]:
                        if combined:
                            self.val_label[image] = ep.Class.values[0]
                            if str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0] not in val_imgs.keys():
                                val_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]] = []
                            val_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]].append(image)
                        elif ep.Emotion.values[0] in self.labels:
                            self.val_label[image] = self.label_index[ep.Emotion.values[0]]
                            if str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0] not in val_imgs.keys():
                                val_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]] = []
                            val_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]].append(image)
                        self.train_image.remove(image)
                    else:
                        if combined:
                            self.train_label[image] = ep.Class.values[0]
                            if str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0] not in train_imgs.keys():
                                train_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]] = []
                            train_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]].append(image)
                        elif ep.Emotion.values[0] in self.labels:
                            self.train_label[image] = self.label_index[ep.Emotion.values[0]]
                            if str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0] not in train_imgs.keys():
                                train_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]] = []
                            train_imgs[str(ep.Subject.values[0]).zfill(3)+ep.Filename.values[0]].append(image)
                        else:
                            self.train_image.remove(image)
        elif self.dataset == 'smic':
            for image in image_list:
                sub = data_info[data_info['Subject'].isin([image.split('/')[-4]])]
                ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                if combined:
                    sub = data_gt[data_gt['Subject'].isin([image.split('/')[-4]])]
                    ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                
                if sub.empty or ep.empty:
                    self.train_image.remove(image)

                elif str(ep.Subject.values[0]) in [str(self.val_subject), 's'+str(self.val_subject)]:
                    if combined:
                        self.val_label[image] = ep.Class.values[0]
                        if ep.Subject.values[0]+ep.Filename.values[0] not in val_imgs.keys():
                            val_imgs[ep.Subject.values[0]+ep.Filename.values[0]] = []
                        val_imgs[ep.Subject.values[0]+ep.Filename.values[0]].append(image)
                    elif ep.Emotion.values[0] in self.labels:
                        self.val_label[image] = self.label_index[ep.Emotion.values[0]]
                        if str(ep.Subject.values[0])+ep.Filename.values[0] not in val_imgs.keys():
                            val_imgs[str(ep.Subject.values[0])+ep.Filename.values[0]] = []
                        val_imgs[str(ep.Subject.values[0])+ep.Filename.values[0]].append(image)
                    self.train_image.remove(image)
                else:
                    if combined:
                        self.train_label[image] = ep.Class.values[0]
                        if ep.Subject.values[0]+ep.Filename.values[0] not in train_imgs.keys():
                            train_imgs[ep.Subject.values[0]+ep.Filename.values[0]] = []
                        train_imgs[ep.Subject.values[0]+ep.Filename.values[0]].append(image)
                    elif ep.Emotion.values[0] in self.labels:
                        self.train_label[image] = self.label_index[ep.Emotion.values[0]]
                        if str(ep.Subject.values[0])+ep.Filename.values[0] not in train_imgs.keys():
                            train_imgs[str(ep.Subject.values[0])+ep.Filename.values[0]] = []
                        train_imgs[str(ep.Subject.values[0])+ep.Filename.values[0]].append(image)
                    else:
                        self.train_image.remove(image)
        # elif self.dataset == 'ck+':
        #     for image in image_list:
        #         sub = data_info[data_info['Subject'].isin([image.split('/')[-3]])]
        #         ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
        #         self.train_label[image] = self.label_index[ep.Emotion.values[0]]
        #         if ep.Subject.values[0]+str(ep.Filename.values[0]) not in train_imgs.keys():
        #             train_imgs[ep.Subject.values[0]+str(ep.Filename.values[0])] = []
        #         train_imgs[ep.Subject.values[0]+str(ep.Filename.values[0])].append(image)

        for i, key in enumerate(val_imgs.keys(), 0):
            self.val_images[i] = val_imgs[key]
        for i, key in enumerate(train_imgs.keys(), 0):
            self.train_images[i] = train_imgs[key]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, subject) where target is index of the target class, subject is the subject number
        """

        img, target = [], []

        if self.train:
            for image in self.train_images[index]:
                target.append(self.train_label[image])
                # doing this so that it is consistent with all other datasets
                # to return a PIL Image
                im = Image.open(image)
                if self.transform is not None:
                    im = self.transform(im)
                img.append(im)

        else:
            for image in self.val_images[index]:
                target.append(self.val_label[image])
                # doing this so that it is consistent with all other datasets
                # to return a PIL Image
                im = Image.open(image)
                if self.transform is not None:
                    im = self.transform(im)
                img.append(im)
        
        img = torch.stack(img)
        target = torch.LongTensor(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_images)
        else:
            return len(self.val_images)
