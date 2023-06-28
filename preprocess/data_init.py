import os
from os.path import join
import glob
import cv2
import numpy as np
import pandas as pd
from math import floor

DIR_DATA = "databases"
DIR_INPUT = "../inputs"

def data_init(dataset, img_num):
    data_folder = join(DIR_DATA,dataset)
    input_folder = join(DIR_INPUT,dataset)
    data_info = pd.read_csv(join(data_folder,'coding.csv'))
    if not os.path.exists(input_folder):
        os.makedirs(join(input_folder,'spatial'))
        os.makedirs(join(input_folder,'temporal'))
        for index, sample in data_info.iterrows():
            if not sample.ApexFrame == '/':
                selectframes(dataset, sample, img_num)

def landmark(dataset):
    dataPath = join(DIR_DATA,dataset)
    data_info = pd.read_csv(join(dataPath,'coding.csv'))

    for _,sample in data_info.iterrows():
        sub = sample.Subject
        ep = sample.Filename
        ftype = '.jpg'
        if not os.path.exists(join(dataPath,'landmarks',sub)):
            os.makedirs(join(dataPath,'landmarks',sub))
    #     onset = 'img'+str(sample.OnsetFrame)+'.jpg'
    #     img = cv2.imread(join(dataPath,'data',sub,ep,onset))
    #     landmark = get_landmarks(img)
    #     np.save(join(dataPath,'landmarks',sub,ep,str(sample.OnsetFrame)+".npy"), landmark)

        if dataset == 'smic':
            ftype = '.bmp'

                        origin_fname_list = [join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(i))+ftype for i in range(startFrame,endFrame+1)]
        elif dataset == 'casme2':
                        origin_fname_list = [join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(i))+ftype for i in range(startFrame,endFrame+1)]
        elif dataset == 'samm'

        apex = 'image'+str(sample.ApexFrame).zfill(6)+'.bmp'
        img = cv2.imread(join(dataPath,'data',sub,sample.Emotion,ep,apex))
        landmark = get_landmarks(img)
        np.save(join(dataPath,'landmarks',sub,ep+".npy"), landmark)

def get_landmarks(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    rects = detector(img_gray,0)
    img_land = np.zeros(img.shape)
    
    for i in range(len(rects)):
        landmark = np.matrix([[p.x,p.y] for p in predictor(img,rects[i]).parts()])
#     return landmark
#         print(landmark)
#         np.save(path+".npy", landmark)
        for idx,point in enumerate(landmark):
            pos = (point[0,0],point[0,1]) 
#             print(idx,pos)
            cv2.circle(img_land,pos,3,(255,0,255))
            cv2.putText(img_land,str(idx),pos,cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),1,cv2.LINE_AA)
#         cv2.imwrite("points_"+path,img)
    return img_land

def selectframes(dataset, sample, img_num):
    data_folder = join(DIR_DATA,dataset)
    input_folder = join(DIR_INPUT,dataset)
    spatial_path, temporal_path = join(input_folder,'spatial'), join(input_folder,'temporal')
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
            # fname_list = [join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(i))+ftype for i in range(startFrame,startFrame+img_num)]
            onset_raw = cv2.imread(join(data_folder,'data','sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(sample.OnsetFrame)+ftype))
            landmarks = np.load(join(data_folder,'landmarks','sub'+str(sample.Subject).zfill(2),sample.Filename+'.npy'))
            os.makedirs(join(spatial_path,'sub'+str(sample.Subject).zfill(2),sample.Filename))
            os.makedirs(join(temporal_path,'sub'+str(sample.Subject).zfill(2),sample.Filename))
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
            # fname_list = [join(str(sample.Subject).zfill(3),sample.Filename,'img'+str(i))+ftype for i in range(startFrame,startFrame+img_num)]
            onset_raw = cv2.imread(join(data_folder,'data',str(sample.Subject).zfill(3),sample.Filename,'img'+str(sample.OnsetFrame)+ftype))
            landmarks = np.load(join(data_folder,'landmarks',str(sample.Subject).zfill(3),sample.Filename+'.npy'))
            os.makedirs(join(spatial_path,str(sample.Subject).zfill(3),sample.Filename))
            os.makedirs(join(temporal_path,str(sample.Subject).zfill(3),sample.Filename))
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
            # fname_list = [join('sub'+str(sample.Subject).zfill(2),sample.Filename,'img'+str(i))+ftype for i in range(startFrame,startFrame+img_num)]
            onset_raw = cv2.imread(join(data_folder,'data',str(sample.Subject),sample.Emotion,sample.Filename,'image'+str(sample.OnsetFrame).zfill(6)+ftype))
            landmarks = np.load(join(data_folder,'landmarks',str(sample.Subject),sample.Filename+'.npy'))
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
            img_raw = cv2.imread(join(data_folder,'data',fname))
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
                img_next = cv2.imread(join(data_folder,'data',origin_fname_list[i+1]))[floor(centre_x-height/2):floor(centre_x+height/2),floor(centre_y-height/2):floor(centre_y+height/2),:]
                inter_img_list = interframes(img_prev, img_next,len(inter_fname_list[fname]))
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


if __name__ == '__main__':
    landmark()
    data_init('casme2',11)
    data_init('smic',11)
    data_init('samm',11)