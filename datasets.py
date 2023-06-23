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
    def __init__(self, transform=None, Spatial=False, train=True, val_subject=1, dataset='ck+', datadir='data', combined = False, AU = False, img_num = 11):
        self.transform = transform
        self.val_subject = val_subject
        self.train = train
        self.dataset = dataset
        self.images_folder = join('../data_raw',dataset)
        self.data_folder = join('..',datadir,dataset)
        self.ftype = '.jpg'
        self.img_num = img_num
        if combined:
            self.labels = ('negative','positive','surprise')
            data_gt = pd.read_csv('../data_raw/combined_3class_gt.csv', header=None, names=['Dataset','Subject','Filename','Class'])
        elif dataset == 'casme2' or dataset == 'casme2-EVM':
            self.labels = ('disgust', 'happiness', 'repression', 'surprise', 'others')
        elif dataset == 'ck+':
            self.labels = ('neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise')
            self.ftype = '.png'
        elif dataset == 'samm' or self.dataset == 'samm-EVM':
            self.labels = ('Other', 'Anger', 'Contempt', 'Happiness', 'Surprise')
        
        if dataset == 'smic' or dataset == 'smic-EVM' or self.dataset == 'demo':
            self.labels = ('negative','positive','surprise')
            self.ftype = '.bmp'
        
        if AU:
            data_au = pd.read_csv(join(self.images_folder,'code_AU.csv'))
        else:
            data_info = pd.read_csv(join(self.images_folder,'coding.csv'))
        # label indexing, {'negative': array(0}, ...}
        self.label_index = {label : i for i, label in enumerate(self.labels)}
        self.subject_num = data_info.Subject.nunique()

        if combined:
            self.subject_num = data_gt.Subject.nunique()

        if not os.path.exists(self.data_folder):
            os.makedirs(join(self.data_folder,'spatial'))
            os.makedirs(join(self.data_folder,'temporal'))
            for index, sample in data_info.iterrows():
                if not sample.ApexFrame == '/':
                    self.selectframes(sample, join(self.data_folder,'spatial'), join(self.data_folder,'temporal'))

        # {image: label}
        if Spatial:
            f_name = 'spatial'
        else:
            f_name = 'temporal'
        image_list = glob.glob(join(self.data_folder, f_name,'*','*','*'+self.ftype), recursive=True)
        if self.dataset == 'smic' or self.dataset == 'smic-EVM':
            image_list = glob.glob(join(self.data_folder, f_name,'*','*','*','*'+self.ftype), recursive=True)
        
        self.train_image = image_list.copy()
        self.val_image = []
        self.val_label = {}
        self.val_images = {}
        self.train_images = {}
        self.train_label = {}
        val_imgs = {}
        train_imgs = {}

        if self.dataset == 'casme2' or self.dataset == 'casme2-EVM':
            for image in image_list:
                if AU:
                    #need to change to video-based
                    ep = data_au[data_au['name'].isin([image.split('/')[-3]+'/'+image.split('/')[-2]])]
                    sub = ep.name.values[0].split('/')[0]
                    if sub == 'sub'+str(self.val_subject).zfill(2):
                        self.val_label[image] = ep.values[0][1:].astype(np.float32)
                        self.val_image.append(image)
                        self.train_image.remove(image)
                    else:
                        self.train_label[image] = ep.values[0][1:].astype(np.float32)

                else:
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
        elif self.dataset == 'samm' or self.dataset == 'samm-EVM':
            for image in image_list:
                if AU:
                    #need to change to video-based
                    ep = data_au[data_au['name'].isin([image.split('/')[-3]+'/'+image.split('/')[-2]])]
                    sub = ep.name.values[0].split('/')[0]
                    if sub == str(self.val_subject).zfill(3):
                        self.val_label[image] = ep.values[0][1:].astype(np.float32)
                        self.val_image.append(image)
                        self.train_image.remove(image)
                    else:
                        self.train_label[image] = ep.values[0][1:].astype(np.float32)

                else:
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
        elif self.dataset == 'smic' or self.dataset == 'smic-EVM'  or self.dataset == 'demo':
            for image in image_list:
                if AU:
                    #need to change to video-based
                    ep = data_au[data_au['name'].isin([image.split('/')[-3]+'/'+image.split('/')[-2]])]
                    sub = ep.name.values[0].split('/')[0]
                    if sub == 'sub'+str(self.val_subject).zfill(2):
                        self.val_label[image] = ep.values[0][1:].astype(np.float32)
                        self.val_image.append(image)
                        self.train_image.remove(image)
                    else:
                        self.train_label[image] = ep.values[0][1:].astype(np.float32)

                else:
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
        elif self.dataset == 'ck+':
            for image in image_list:
                sub = data_info[data_info['Subject'].isin([image.split('/')[-3]])]
                ep = sub[sub['Filename'].isin([image.split('/')[-2]])]
                self.train_label[image] = self.label_index[ep.Emotion.values[0]]
                if ep.Subject.values[0]+str(ep.Filename.values[0]) not in train_imgs.keys():
                    train_imgs[ep.Subject.values[0]+str(ep.Filename.values[0])] = []
                train_imgs[ep.Subject.values[0]+str(ep.Filename.values[0])].append(image)

        for i, key in enumerate(val_imgs.keys(), 0):
            self.val_images[i] = val_imgs[key]
        for i, key in enumerate(train_imgs.keys(), 0):
            self.train_images[i] = train_imgs[key]

    def selectframes(self, sample, spatial_path, temporal_path):
        if int(sample.ApexFrame)>int(sample.OnsetFrame):
            inter_num = 0
            if self.dataset == 'casme2' or self.dataset == 'casme2-EVM':
                startFrame = max(int(sample.ApexFrame)-int(self.img_num/2),int(sample.OnsetFrame)+1)
                endFrame = min(startFrame+self.img_num-1,int(sample.OffsetFrame))
                origin_fname_list = [join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(i))+self.ftype for i in range(startFrame,endFrame+1)]
                if startFrame+self.img_num>int(sample.OffsetFrame):
                    inter_fname_list = {}
                    inter_num = startFrame+self.img_num-1-int(sample.OffsetFrame)
                    inter_frame_num = int(inter_num/(len(origin_fname_list)-1))
                    left_inter_num = inter_num-inter_frame_num*int(len(origin_fname_list)-1)
                    start_left = int(len(origin_fname_list)/2)-int(left_inter_num/2)
                    end_left = start_left+left_inter_num
                    for i in range(0,len(origin_fname_list)-1):
                        prev_fname = origin_fname_list[i]
                        inter_fname_list[prev_fname] = [prev_fname[:prev_fname.index(self.ftype)]+"_"+str(i)+self.ftype for i in range(1,inter_frame_num+1)]
                        if i in range(start_left,end_left):
                            inter_fname_list[prev_fname].append(prev_fname[:prev_fname.index(self.ftype)]+"_"+str(inter_frame_num+1)+self.ftype)
                # fname_list = [join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(i))+self.ftype for i in range(startFrame,startFrame+self.img_num)]
                onset_raw = cv2.imread(join(self.images_folder,'data','sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(sample.OnsetFrame)+self.ftype))
                landmarks = np.load(join(self.images_folder,'landmarks','sub'+str(sample.Subject).zfill(2),sample.Filename+'.npy'))
                os.makedirs(join(spatial_path,'sub'+str(sample.Subject).zfill(2),sample.Filename))
                os.makedirs(join(temporal_path,'sub'+str(sample.Subject).zfill(2),sample.Filename))
            elif self.dataset == 'ck+':
                startFrame = max(int(sample.ApexFrame)-int(self.img_num/2),int(sample.OnsetFrame)+1)
                endFrame = min(startFrame+self.img_num-1,int(sample.OffsetFrame))
                origin_fname_list = [join(sample.Subject,str(sample.Filename).zfill(3),'img'+str(i))+self.ftype for i in range(startFrame,endFrame+1)]
                if startFrame+self.img_num>int(sample.OffsetFrame):
                    inter_fname_list = {}
                    inter_num = startFrame+self.img_num-1-int(sample.OffsetFrame)
                    inter_frame_num = int(inter_num/(len(origin_fname_list)-1))
                    left_inter_num = inter_num-inter_frame_num*int(len(origin_fname_list)-1)
                    start_left = int(len(origin_fname_list)/2)-int(left_inter_num/2)
                    end_left = start_left+left_inter_num
                    for i in range(0,len(origin_fname_list)-1):
                        prev_fname = origin_fname_list[i]
                        inter_fname_list[prev_fname] = [prev_fname[:prev_fname.index(self.ftype)]+"_"+str(i)+self.ftype for i in range(1,inter_frame_num+1)]
                        if i in range(start_left,end_left):
                            inter_fname_list[prev_fname].append(prev_fname[:prev_fname.index(self.ftype)]+"_"+str(inter_frame_num+1)+self.ftype)
                # fname_list = [join(sample.Subject,str(sample.Filename).zfill(3),'img'+str(i))+self.ftype for i in range(startFrame,startFrame+self.img_num)]
                onset_raw = cv2.imread(join(self.images_folder,'data',sample.Subject,str(sample.Filename).zfill(3),'img'+str(sample.OnsetFrame)+self.ftype))
                landmarks = np.load(join(self.images_folder,'landmarks',sample.Subject,str(sample.Filename).zfill(3)+'.npy'))
                os.makedirs(join(spatial_path,sample.Subject,str(sample.Filename).zfill(3)))
                os.makedirs(join(temporal_path,sample.Subject,str(sample.Filename).zfill(3)))
            elif self.dataset == 'samm' or self.dataset == 'samm-EVM':
                startFrame = max(int(sample.ApexFrame)-int(self.img_num/2),int(sample.OnsetFrame)+1)
                endFrame = min(startFrame+self.img_num-1,int(sample.OffsetFrame))
                origin_fname_list = [join(str(sample.Subject).zfill(3),sample.Filename,'img'+str(i))+self.ftype for i in range(startFrame,endFrame+1)]
                if startFrame+self.img_num>int(sample.OffsetFrame):
                    inter_fname_list = {}
                    inter_num = startFrame+self.img_num-1-int(sample.OffsetFrame)
                    inter_frame_num = int(inter_num/(len(origin_fname_list)-1))
                    left_inter_num = inter_num-inter_frame_num*int(len(origin_fname_list)-1)
                    start_left = int(len(origin_fname_list)/2)-int(left_inter_num/2)
                    end_left = start_left+left_inter_num
                    for i in range(0,len(origin_fname_list)-1):
                        prev_fname = origin_fname_list[i]
                        inter_fname_list[prev_fname] = [prev_fname[:prev_fname.index(self.ftype)]+"_"+str(i)+self.ftype for i in range(1,inter_frame_num+1)]
                        if i in range(start_left,end_left):
                            inter_fname_list[prev_fname].append(prev_fname[:prev_fname.index(self.ftype)]+"_"+str(inter_frame_num+1)+self.ftype)
                # fname_list = [join(str(sample.Subject).zfill(3),sample.Filename,'img'+str(i))+self.ftype for i in range(startFrame,startFrame+self.img_num)]
                onset_raw = cv2.imread(join(self.images_folder,'data',str(sample.Subject).zfill(3),sample.Filename,'img'+str(sample.OnsetFrame)+self.ftype))
                landmarks = np.load(join(self.images_folder,'landmarks',str(sample.Subject).zfill(3),sample.Filename+'.npy'))
                os.makedirs(join(spatial_path,str(sample.Subject).zfill(3),sample.Filename))
                os.makedirs(join(temporal_path,str(sample.Subject).zfill(3),sample.Filename))
            elif self.dataset == 'smic' or self.dataset == 'smic-EVM' or self.dataset == 'demo':
                startFrame = max(int(sample.ApexFrame)-int(self.img_num/2),int(sample.OnsetFrame)+1)
                endFrame = min(startFrame+self.img_num-1,int(sample.OffsetFrame))
                origin_fname_list = [join(str(sample.Subject),sample.Emotion,sample.Filename,'image'+str(i).zfill(6))+self.ftype for i in range(startFrame,endFrame+1)]
                if startFrame+self.img_num>int(sample.OffsetFrame):
                    inter_fname_list = {}
                    inter_num = startFrame+self.img_num-1-int(sample.OffsetFrame)
                    inter_frame_num = int(inter_num/(len(origin_fname_list)-1))
                    left_inter_num = inter_num-inter_frame_num*int(len(origin_fname_list)-1)
                    start_left = int(len(origin_fname_list)/2)-int(left_inter_num/2)
                    end_left = start_left+left_inter_num
                    for i in range(0,len(origin_fname_list)-1):
                        prev_fname = origin_fname_list[i]
                        inter_fname_list[prev_fname] = [prev_fname[:prev_fname.index(self.ftype)]+"_"+str(i)+self.ftype for i in range(1,inter_frame_num+1)]
                        if i in range(start_left,end_left):
                            inter_fname_list[prev_fname].append(prev_fname[:prev_fname.index(self.ftype)]+"_"+str(inter_frame_num+1)+self.ftype)
                # fname_list = [join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(i))+self.ftype for i in range(startFrame,startFrame+self.img_num)]
                onset_raw = cv2.imread(join(self.images_folder,'data',str(sample.Subject),sample.Emotion,sample.Filename,'image'+str(sample.OnsetFrame).zfill(6)+self.ftype))
                landmarks = np.load(join(self.images_folder,'landmarks',str(sample.Subject),sample.Filename+'.npy'))
                os.makedirs(join(spatial_path,sample.Subject,sample.Emotion,sample.Filename))
                os.makedirs(join(temporal_path,sample.Subject,sample.Emotion,sample.Filename))

            centre_y,centre_x = landmarks[30]
            height = landmarks[8][1]-landmarks[19][1]+landmarks[8][1]-landmarks[57][1]
            # onset = onset_raw[:,floor(centre_x-onset_raw.shape[0]/2):floor(centre_x+onset_raw.shape[0]/2),:]
            onset = onset_raw[floor(centre_x-height/2):floor(centre_x+height/2),floor(centre_y-height/2):floor(centre_y+height/2),:]
            onset_gray = cv2.cvtColor(onset, cv2.COLOR_RGB2GRAY)

            hsv = np.zeros_like(onset)
            hsv[..., 1] = 255

            for i in range(0,len(origin_fname_list)):
                fname = origin_fname_list[i]
                img_raw = cv2.imread(join(self.images_folder,'data',fname))
                # img = img_raw[:,floor(centre_x-onset_raw.shape[0]/2):floor(centre_x+onset_raw.shape[0]/2),:]
                img = img_raw[floor(centre_x-height/2):floor(centre_x+height/2),floor(centre_y-height/2):floor(centre_y+height/2),:]
                cv2.imwrite(join(spatial_path,fname), img)

                # optical flow:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(onset_gray, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(join(temporal_path,fname), rgb)

                if inter_num>0 and i+1<len(origin_fname_list) and len(inter_fname_list[fname])>0:
                    img_prev = img
                    img_next = cv2.imread(join(self.images_folder,'data',origin_fname_list[i+1]))[floor(centre_x-height/2):floor(centre_x+height/2),floor(centre_y-height/2):floor(centre_y+height/2),:]
                    inter_img_list = self.interframes(img_prev, img_next,len(inter_fname_list[fname]))
                    inter_img_list = inter_img_list[int(len(inter_img_list)/2)-int(len(inter_fname_list[fname])/2):int(len(inter_img_list)/2)-int(len(inter_fname_list[fname])/2)+len(inter_fname_list[fname])]
                    for inter_fname, inter_img in zip(inter_fname_list[fname],inter_img_list):
                        cv2.imwrite(join(spatial_path,inter_fname), inter_img)

                        img_gray = cv2.cvtColor(inter_img, cv2.COLOR_RGB2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(onset_gray, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        hsv[..., 0] = ang*180/np.pi/2
                        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        cv2.imwrite(join(temporal_path,inter_fname), rgb)


            # while i < num_inter:
            #     # interpolate list as ApexFrame + [0,-1,+1,-2,+2]
            #     if self.dataset == 'casme2' or self.dataset == 'casme2-EVM':
            #         prev_fname = join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(int(sample.ApexFrame)+int((i+1)/2)*pow(-1,i))+self.ftype)
            #         next_fname = join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(int(sample.ApexFrame)+int((i+1)/2)*pow(-1,i)+1)+self.ftype)
            #         inter_fname = join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(int(sample.ApexFrame)+int((i+1)/2)*pow(-1,i)+1)+'-'+str(int(sample.ApexFrame)+int((i+1)/2)*pow(-1,i))+self.ftype)
            #     elif self.dataset == 'ck+':
            #         prev_fname = join(sample.Subject,str(sample.Filename).zfill(3),'img'+str(int(sample.ApexFrame)-i-1)+self.ftype)
            #         next_fname = join(sample.Subject,str(sample.Filename).zfill(3),'img'+str(int(sample.ApexFrame)-i)+self.ftype)
            #         inter_fname = join(sample.Subject,str(sample.Filename).zfill(3),'img'+str(int(sample.ApexFrame)-i)+'-'+str(int(sample.ApexFrame)-i-1)+self.ftype)
            #     elif self.dataset == 'samm' or self.dataset == 'samm-EVM' or self.dataset == 'demo':
            #         prev_fname = join(str(sample.Subject).zfill(3),sample.Filename,'img'+str(int(sample.ApexFrame)+int((i+1)/2)*pow(-1,i))+self.ftype)
            #         next_fname = join(str(sample.Subject).zfill(3),sample.Filename,'img'+str(int(sample.ApexFrame)+int((i+1)/2)*pow(-1,i)+1)+self.ftype)
            #         inter_fname = join(str(sample.Subject).zfill(3),sample.Filename,'img'+str(int(sample.ApexFrame)+int((i+1)/2)*pow(-1,i)+1)+'-'+str(int(sample.ApexFrame)+int((i+1)/2)*pow(-1,i))+self.ftype)
                
            #     img0 = cv2.imread(join(self.images_folder, 'data', prev_fname))
            #     img1 = cv2.imread(join(self.images_folder, 'data', next_fname))
            #     if img0 is None or img1 is None or prev_fname.split('/')[-1] == 'img'+str(sample.OnsetFrame)+self.ftype:
            #         if self.dataset == 'casme2' or self.dataset == 'casme2-EVM' or self.dataset == 'samm' or self.dataset == 'samm-EVM' or self.dataset == 'demo':
            #             num_inter += 1
            #             i += 1
            #             continue
            #         elif self.dataset == 'ck+':
            #             prev_fname = join(sample.Subject,str(sample.Filename).zfill(3),'img'+str(int(sample.ApexFrame)-(num_inter-i)+1)+'-'+str(int(sample.ApexFrame)-(num_inter-i))+self.ftype)
            #             next_fname = join(sample.Subject,str(sample.Filename).zfill(3),'img'+str(int(sample.ApexFrame)-(num_inter-i)+1)+self.ftype)
            #             inter_fname = join(sample.Subject,str(sample.Filename).zfill(3),'img'+str(int(sample.ApexFrame)-(num_inter-i)+1)+'-'+str(int(sample.ApexFrame)-(num_inter-i))+'v1'+self.ftype)
            #             img0 = cv2.imread(join(spatial_path, prev_fname))
            #             img1 = cv2.imread(join(spatial_path, next_fname))
            #         img = interpolate(img0,img1)
            #     else:
            #         img = interpolate(img0,img1)
            #         # img = img[:,floor(centre_x-onset_raw.shape[0]/2):floor(centre_x+onset_raw.shape[0]/2),:]
            #         img = img[floor(centre_x-height/2):floor(centre_x+height/2),floor(centre_y-height/2):floor(centre_y+height/2),:]
                
                # cv2.imwrite(join(spatial_path,inter_fname), img)

                # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # flow = cv2.calcOpticalFlowFarneback(onset_gray, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                # hsv[..., 0] = ang*180/np.pi/2
                # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # cv2.imwrite(join(temporal_path,inter_fname), rgb)
                # i = i+1


            # using data without EVM for temporal stream
            # if 'EVM' in self.dataset:
            #     temporal_dataset = self.dataset[:-4]
            #     onset_raw = cv2.imread(join('../data_raw',temporal_dataset,'../data','sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(sample.OnsetFrame)+self.ftype))
            #     landmarks = np.load(join('../data_raw',temporal_dataset,'landmarks','sub'+str(sample.Subject).zfill(2),sample.Filename+'.npy'))
            #     centre_x = landmarks[33][0]

            #     onset = onset_raw[:,centre_x-floor(onset_raw.shape[0]/2):centre_x+floor(onset_raw.shape[0]/2),:]
            #     onset_gray = cv2.cvtColor(onset, cv2.COLOR_RGB2GRAY)

            #     hsv = np.zeros_like(onset)
            #     hsv[..., 1] = 255

            #     for fname in fname_list:
            #         img_raw = cv2.imread(join('../data_raw',temporal_dataset,'../data',fname))
            #         if img_raw is not None:
            #             img = img_raw[:,centre_x-floor(onset_raw.shape[0]/2):centre_x+floor(onset_raw.shape[0]/2),:]

            #             img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #             flow = cv2.calcOpticalFlowFarneback(onset_gray, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #             mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #             hsv[..., 0] = ang*180/np.pi/2
            #             hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            #             rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #             cv2.imwrite(join(temporal_path,fname), rgb)

            if self.dataset == 'casme2' or self.dataset == 'casme2-EVM':
                print('spatial & temporal images are extracted in '+join('sub'+str(sample.Subject).zfill(2),sample.Filename))
            elif self.dataset == 'ck+':
                print('spatial & temporal images are extracted in '+join(sample.Subject,str(sample.Filename).zfill(3)))
            elif self.dataset == 'samm' or self.dataset == 'samm-EVM':
                print('spatial & temporal images are extracted in '+join(str(sample.Subject).zfill(3),sample.Filename))
            elif self.dataset == 'smic' or self.dataset == 'smic-EVM' or self.dataset == 'demo':
                print('spatial & temporal images are extracted in '+join(str(sample.Subject),sample.Emotion,sample.Filename))

    def interframes(self, img0, img1, inter_num):
        img_list = [img0, img1]
        while len(img_list) < inter_num+2:
            tmp = []
            for i in range(len(img_list) - 1):
                mid = interpolate(img_list[i], img_list[i + 1])
                tmp.append(img_list[i])
                tmp.append(mid)
            tmp.append(img1)
            img_list = tmp
        return img_list[1:-1]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, subject) where target is index of the target class, subject is the subject number
        """

        img, target = [], []

        if self.train:
            # img = self.train_image[index]
            # target = self.train_label[img]

            # # doing this so that it is consistent with all other datasets
            # # to return a PIL Image
            # img = Image.open(img)
            # if self.transform is not None:
            #     img = self.transform(img)
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
