import os
from os.path import join
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import dlib
from math import floor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from RIFE.interpolate_img import interpolate

DIR_DATA = "preprocess/databases"
DIR_INPUT = "inputs"

def data_init(dataset, img_num):
    print('detecting landmarks for: '+dataset)
    landmark(dataset)
    print('Done!')
    print('generate inputs for: '+dataset)
    data_folder = join(DIR_DATA,dataset)
    input_folder = join(DIR_INPUT,dataset)
    data_info = pd.read_csv(join(data_folder,'coding.csv'))
    if not os.path.exists(join(input_folder,'LOF')):
        os.makedirs(join(input_folder,'LOF'))
        for index, sample in data_info.iterrows():
            if not sample.ApexFrame == '/':
                selectframes(dataset, sample, img_num)
    print('Done!')

def landmark(dataset):
    data_folder = join(DIR_DATA,dataset)
    data_info = pd.read_csv(join(data_folder,'coding.csv'))
    if not os.path.exists(join(data_folder,'landmarks')):
        os.makedirs(join(data_folder,'landmarks'))
    else:
        return

    ftype = '.jpg'
    if dataset == 'smic':
        ftype = '.bmp'
    for _,sample in data_info.iterrows():
        sub = sample.Subject
        ep = sample.Filename
        if dataset == 'smic':
            apex = 'image'+str(sample.ApexFrame).zfill(6)+ftype
            img = cv2.imread(join(data_folder,'data',sub,sample.Emotion,ep,apex))
        elif dataset == 'casme2':
            sub = 'sub'+str(sample.Subject).zfill(2)
            apex = 'img'+str(sample.ApexFrame)+ftype
            img = cv2.imread(join(data_folder,'data',sub,ep,apex))
        elif dataset == 'samm':
            sub = str(sample.Subject).zfill(3)
            apex = 'img'+str(sample.ApexFrame)+ftype
            img = cv2.imread(join(data_folder,'data',sub,ep,apex))

        if not os.path.exists(join(data_folder,'landmarks',sub)):
            os.makedirs(join(data_folder,'landmarks',sub))
        landmark = get_landmarks(img)
        np.save(join(data_folder,'landmarks',sub,ep+".npy"), landmark)

def get_landmarks(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("preprocess/shape_predictor_68_face_landmarks.dat")
    
    rects = detector(img_gray,0)
    
    for i in range(len(rects)):
        landmark = np.matrix([[p.x,p.y] for p in predictor(img,rects[i]).parts()])
    return landmark

def selectframes(dataset, sample, img_num):
    data_folder = join(DIR_DATA,dataset)
    input_folder = join(DIR_INPUT,dataset,'LOF')
    if not os.path.exists(join(input_folder,'sub'+str(sample.Subject).zfill(2),sample.Filename)):
        ftype = '.jpg'
        if int(sample.ApexFrame)>int(sample.OnsetFrame):
            inter_num = 0
            if dataset == 'casme2':
                startFrame = max(int(sample.ApexFrame)-int(img_num/2),int(sample.OnsetFrame)+1)
                endFrame = min(startFrame+img_num-1,int(sample.OffsetFrame))
                origin_fname_list = [join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(i))+ftype for i in range(startFrame,endFrame+1)]
                if startFrame+img_num>int(sample.OffsetFrame):
                    inter_fname_list = {}
                    inter_num = startFrame+img_num-1-int(sample.OffsetFrame)
                    inter_frame_num = int(inter_num/(len(origin_fname_list)-1))
                    left_inter_num = inter_num-inter_frame_num*int(len(origin_fname_list)-1)
                    start_left = int(len(origin_fname_list)/2)-int(left_inter_num/2)
                    end_left = start_left+left_inter_num
                    for i in range(0,len(origin_fname_list)-1):
                        prev_fname = origin_fname_list[i]
                        inter_fname_list[prev_fname] = [prev_fname[:prev_fname.index(ftype)]+"_"+str(i)+ftype for i in range(1,inter_frame_num+1)]
                        if i in range(start_left,end_left):
                            inter_fname_list[prev_fname].append(prev_fname[:prev_fname.index(ftype)]+"_"+str(inter_frame_num+1)+ftype)
                onset_raw = cv2.imread(join(data_folder,'data','sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(sample.OnsetFrame)+ftype))
                landmarks = np.load(join(data_folder,'landmarks','sub'+str(sample.Subject).zfill(2),sample.Filename+'.npy'))
                os.makedirs(join(input_folder,'sub'+str(sample.Subject).zfill(2),sample.Filename))
            elif dataset == 'samm':
                startFrame = max(int(sample.ApexFrame)-int(img_num/2),int(sample.OnsetFrame)+1)
                endFrame = min(startFrame+img_num-1,int(sample.OffsetFrame))
                origin_fname_list = [join(str(sample.Subject).zfill(3),sample.Filename,'img'+str(i))+ftype for i in range(startFrame,endFrame+1)]
                if startFrame+img_num>int(sample.OffsetFrame):
                    inter_fname_list = {}
                    inter_num = startFrame+img_num-1-int(sample.OffsetFrame)
                    inter_frame_num = int(inter_num/(len(origin_fname_list)-1))
                    left_inter_num = inter_num-inter_frame_num*int(len(origin_fname_list)-1)
                    start_left = int(len(origin_fname_list)/2)-int(left_inter_num/2)
                    end_left = start_left+left_inter_num
                    for i in range(0,len(origin_fname_list)-1):
                        prev_fname = origin_fname_list[i]
                        inter_fname_list[prev_fname] = [prev_fname[:prev_fname.index(ftype)]+"_"+str(i)+ftype for i in range(1,inter_frame_num+1)]
                        if i in range(start_left,end_left):
                            inter_fname_list[prev_fname].append(prev_fname[:prev_fname.index(ftype)]+"_"+str(inter_frame_num+1)+ftype)
                onset_raw = cv2.imread(join(data_folder,'data',str(sample.Subject).zfill(3),sample.Filename,'img'+str(sample.OnsetFrame)+ftype))
                landmarks = np.load(join(data_folder,'landmarks',str(sample.Subject).zfill(3),sample.Filename+'.npy'))
                os.makedirs(join(input_folder,str(sample.Subject).zfill(3),sample.Filename))
            elif dataset == 'smic':
                ftype = '.bmp'
                startFrame = max(int(sample.ApexFrame)-int(img_num/2),int(sample.OnsetFrame)+1)
                endFrame = min(startFrame+img_num-1,int(sample.OffsetFrame))
                origin_fname_list = [join(str(sample.Subject),sample.Emotion,sample.Filename,'image'+str(i).zfill(6))+ftype for i in range(startFrame,endFrame+1)]
                if startFrame+img_num>int(sample.OffsetFrame):
                    inter_fname_list = {}
                    inter_num = startFrame+img_num-1-int(sample.OffsetFrame)
                    inter_frame_num = int(inter_num/(len(origin_fname_list)-1))
                    left_inter_num = inter_num-inter_frame_num*int(len(origin_fname_list)-1)
                    start_left = int(len(origin_fname_list)/2)-int(left_inter_num/2)
                    end_left = start_left+left_inter_num
                    for i in range(0,len(origin_fname_list)-1):
                        prev_fname = origin_fname_list[i]
                        inter_fname_list[prev_fname] = [prev_fname[:prev_fname.index(ftype)]+"_"+str(i)+ftype for i in range(1,inter_frame_num+1)]
                        if i in range(start_left,end_left):
                            inter_fname_list[prev_fname].append(prev_fname[:prev_fname.index(ftype)]+"_"+str(inter_frame_num+1)+ftype)
                onset_raw = cv2.imread(join(data_folder,'data',str(sample.Subject),sample.Emotion,sample.Filename,'image'+str(sample.OnsetFrame).zfill(6)+ftype))
                landmarks = np.load(join(data_folder,'landmarks',str(sample.Subject),sample.Filename+'.npy'))
                os.makedirs(join(input_folder,sample.Subject,sample.Emotion,sample.Filename))

            centre_y,centre_x = landmarks[30]
            height = landmarks[8][1]-landmarks[19][1]+landmarks[8][1]-landmarks[57][1]
            onset = onset_raw[floor(centre_x-height/2):floor(centre_x+height/2),floor(centre_y-height/2):floor(centre_y+height/2),:]
            onset_gray = cv2.cvtColor(onset, cv2.COLOR_RGB2GRAY)

            hsv = np.zeros_like(onset)
            hsv[..., 1] = 255

            for i in range(0,len(origin_fname_list)):
                fname = origin_fname_list[i]
                img_raw = cv2.imread(join(data_folder,'data',fname))
                img = img_raw[floor(centre_x-height/2):floor(centre_x+height/2),floor(centre_y-height/2):floor(centre_y+height/2),:]
                # long term optical flow:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(onset_gray, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(join(input_folder,fname), rgb)

                if inter_num>0 and i+1<len(origin_fname_list) and len(inter_fname_list[fname])>0:
                    img_prev = img
                    img_next = cv2.imread(join(data_folder,'data',origin_fname_list[i+1]))[floor(centre_x-height/2):floor(centre_x+height/2),floor(centre_y-height/2):floor(centre_y+height/2),:]
                    inter_img_list = interframes(img_prev, img_next,len(inter_fname_list[fname]))
                    inter_img_list = inter_img_list[int(len(inter_img_list)/2)-int(len(inter_fname_list[fname])/2):int(len(inter_img_list)/2)-int(len(inter_fname_list[fname])/2)+len(inter_fname_list[fname])]
                    for inter_fname, inter_img in zip(inter_fname_list[fname],inter_img_list):
                        cv2.imwrite(join(input_folder,inter_fname), inter_img)

                        img_gray = cv2.cvtColor(inter_img, cv2.COLOR_RGB2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(onset_gray, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        hsv[..., 0] = ang*180/np.pi/2
                        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        cv2.imwrite(join(input_folder,inter_fname), rgb)


def interframes(img0, img1, inter_num):
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


if __name__ == '__main__':
    data_init('casme2',11)
    data_init('smic',11)
    data_init('samm',11)