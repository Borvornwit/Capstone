import json
import cv2
import os
import time
import sys
from itertools import combinations
import logging
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, Image
import base64
from pyDOE2 import *
import numpy.matlib
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
import pandas as pd
from tqdm.notebook import tqdm

from os import listdir, mkdir
from os.path import isfile, join

sys.path.insert(0,'/mnt/storage/shared/datasets/face_dataset/faceDetetorAndAlignment')
from faceDetectorAndAlignment import faceDetectorAndAlignment
from faceLandmark import faceLandmark
_faceDetectorAndAlignment = faceDetectorAndAlignment('/mnt/storage/shared/datasets/face_dataset/faceDetetorAndAlignment/models/faceDetectorV2.onnx', processScale=0.20) # increase processScale will increase detection accuracy but reduce fps
_faceLandmarks = faceLandmark('/mnt/storage/shared/datasets/face_dataset/faceDetetorAndAlignment/models/faceLandmark64Light.onnx')

def get_rect_point(point):
  x1,y1 = point[0]
  x2,y2 = point[1]
  return np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
def fps(path):
	cap = cv2.VideoCapture(path)
	fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration = frame_count/fps

	print('fps = ' + str(fps))
	print('number of frames = ' + str(frame_count))
	print('duration (S) = ' + str(duration))
	minutes = int(duration/60)
	seconds = duration%60
	print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

	cap.release()
	return
def count_frames(path, override=False):
	video = cv2.VideoCapture(path)
	total = 0
	if override:
		total = count_frames_manual(video)
	else:
		try:
			if is_cv3():
				total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
			else:
				total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		except:
			total = count_frames_manual(video)
	video.release()
	return total
def count_frames_manual(video):
	total = 0
	while True:
		(grabbed, frame) = video.read()
	 
		if not grabbed:
			break
		total += 1
	return total
def moving_average(x, w):
		x = np.pad(x,(1,),'mean')
		return np.convolve(x, np.ones(w), 'valid') / w

def getRoiSignal(frame,mask):
  m,n,c = frame.shape
  #print(frame.shape)
  #print(mask.shape)
  signal = np.zeros((1,1,c))
  #print(mask)
  signal[0,0] = cv2.mean(frame,mask=mask)[:3]
  '''
  for i in range(c):
    tmp = img[:,:,i]
    signal[0,0,i] = cv2.mean(frame, mask = mask)[i]
  '''
  return signal

def getCombinedSignalMap(SignalMap, ROInum):
  #print(ROInum)
  ROInum_size = ROInum.shape[0]
  All_idx = ff2n(int(ROInum_size))
  SignalMapOut = np.zeros((np.shape(All_idx)[0]-1,1,np.shape(SignalMap)[1]))
  for i in range(1,np.shape(All_idx)[0]):
      tmp_idx = np.where(All_idx[i,:] == 1)[0]
      # tmp_signal = [SignalMap[k,:] if k in tmp_idx else np.zeros(SignalMap[0,:].shape) for k in range(len(All_idx[:,0]))]
      # tmp_ROI = [ROInum[k] if k in tmp_idx else np.zeros(ROInum[0].shape) for k in range(len(All_idx[:,0]))]
      tmp_signal = [SignalMap[k,:] for k in tmp_idx]
      tmp_ROI = [ROInum[k] for k in tmp_idx]
      tmp_ROI = tmp_ROI/sum(tmp_ROI)
      # print("tmp_idx",tmp_idx)
      # print(tmp_ROI)
      # print(tmp_ROI.shape)
      tmp_ROI = np.repeat(tmp_ROI,3, axis=1)
      
      SignalMapOut[i-1,:,:] = np.sum(np.multiply(tmp_signal,tmp_ROI),axis=0)
      #print(SignalMapOut.shape)
      #print("--------")
      #print(SignalMapOut[:,0,:])
  return SignalMapOut[:,0,:]

def generateSignalMap(MSTmap_whole,frame,idx,landmarks):
  roi_list = list()
  mask_list = list()
  for landmark in landmarks:
    w,h,c = frame.shape
    mask = np.zeros((w,h), dtype=np.uint8)
    cv2.fillPoly(mask, [landmark], (255))

    #res = cv2.bitwise_and(frame,frame,mask = mask)

    #rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    #cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    mask_list.append(mask)
  #cv2_imshow(mask)
  #print(cv2.countNonZero(mask))
  signal_tmp = np.zeros((len(mask_list),3))
  roi_size = np.zeros((len(mask_list),1),dtype=np.int32)
  for i in range(len(mask_list)):
    signal_tmp[i] = getRoiSignal(frame,mask_list[i])
    #print("mask_list[",i,"]",cv2.countNonZero(mask_list[i]))
    roi_size[i] = cv2.countNonZero(mask_list[i])
  MSTmap_whole[:,idx,:] = getCombinedSignalMap(signal_tmp,roi_size)
  #if(idx != 0): print(MSTmap_whole[:,idx,:] == MSTmap_whole[:,idx-1,:])
  return MSTmap_whole

def forehead_pos(x1,y1,x2,y2):
  x_diff = x2-x1
  y_diff = y2-y1
  return [(int(x1+(0.25*x_diff)),int(y1+0.1*y_diff)),(int(x2-(0.25*x_diff)),int(y1+0.2*y_diff))]
def leftCheekUpper_pos(x1,y1,x2,y2):
  x_diff = x2-x1
  y_diff = y2-y1
  return [(int(x1+(0.05*x_diff)),int(y1+(0.5*y_diff))),(int(x1+(0.25*x_diff)),int(y1+0.65*y_diff))]
def leftCheekBelow_pos(x1,y1,x2,y2):
  x_diff = x2-x1
  y_diff = y2-y1
  return [(int(x1+(0.1*x_diff)),int(y1+(0.65*y_diff))),(int(x1+(0.25*x_diff)),int(y1+0.8*y_diff))]
def rightCheekUpper_pos(x1,y1,x2,y2):
  x_diff = x2-x1
  y_diff = y2-y1
  return [(int(x2-(0.05*x_diff)),int(y1+(0.5*y_diff))),(int(x2-(0.25*x_diff)),int(y1+0.65*y_diff))]
def rightCheekBelow_pos(x1,y1,x2,y2):
  x_diff = x2-x1
  y_diff = y2-y1
  return [(int(x2-(0.1*x_diff)),int(y1+(0.65*y_diff))),(int(x2-(0.25*x_diff)),int(y1+0.8*y_diff))]
def chin_pos(x1,y1,x2,y2):
  x_diff = x2-x1
  y_diff = y2-y1
  return [(int(x1+(0.3*x_diff)),int(y2-(0.1*y_diff))),(int(x2-(0.3*x_diff)),int(y2))]
#def get_face_contours(x1,y1,x2,y2):
#  func_list = [forehead_pos, leftCheekUpper_pos, leftCheekBelow_pos, rightCheekUpper_pos, rightCheekBelow_pos, chin_pos]
#  return [get_rect_point(func(x1,y1,x2,y2)) for func in func_list]
def get_face_contours(faceLandmarks):
  contours = []
  contours.append(faceLandmarks[0,np.array([0,1,2,31,41,0])].astype(np.int32))
  contours.append(faceLandmarks[0,np.array([2,3,4,5,48,31,2])].astype(np.int32))
  contours.append(faceLandmarks[0,np.array([16,15,14,35,46,16])].astype(np.int32))
  contours.append(faceLandmarks[0,np.array([14,13,12,11,54,35,14])].astype(np.int32))
  contours.append(faceLandmarks[0,np.concatenate((np.arange(5,12),np.arange(54,60),np.array([48,5])))].astype(np.int32))
  forehead = faceLandmarks[0,np.concatenate((np.arange(17,22),np.arange(22,27)))]
  left_eye = np.mean(faceLandmarks[0,36:42],axis=0)
  right_eye = np.mean(faceLandmarks[0,42:48],axis=0)
  eye_distance = np.linalg.norm(left_eye - right_eye)
  tmp = (np.mean(faceLandmarks[0,17:22],axis=0) + np.mean(faceLandmarks[0,22:27],axis=0))/2 - (left_eye + right_eye)/2
  tmp = eye_distance/np.linalg.norm(tmp)*0.5*tmp
  contours.append(np.concatenate((forehead,(forehead[-1]+tmp).reshape(1,-1),(forehead[0]+tmp).reshape(1,-1),forehead[0].reshape(1,-1))).astype(np.int32))
  return contours

def topRight_pos(x1,y1,x2,y2,w,h):
  y_diff = y2-y1
  return [(0,y1),(int(0.5*x1),y1),(int(0.5*x1),y1+int(0.5*y_diff)),(0,y1+int(0.5*y_diff))]

def topLeft_pos(x1,y1,x2,y2,w,h):
  y_diff = y2-y1
  return [(x2+int(0.5*(w-x2)),y1),(w,y1),(w,y1+int(0.5*y_diff)),(x2+int(0.5*(w-x2)),y1+int(0.5*y_diff))]

def bottomRight_pos(x1,y1,x2,y2,w,h):
  y_diff = y2-y1
  return [(0,y2-int(0.5*y_diff)),(int(0.5*x1),y2-int(0.5*y_diff)),(int(0.5*x1),y2),(0,y2)]

def bottomLeft_pos(x1,y1,x2,y2,w,h):
  y_diff = y2-y1
  return  [(x2+int(0.5*(w-x2)),y2-int(0.5*y_diff)),(w,y2-int(0.5*y_diff)),(w,y2),(x2+int(0.5*(w-x2)),y2)]

def get_bg_contours(x1,y1,x2,y2,w,h):
  func_list = [topRight_pos,topLeft_pos,bottomRight_pos,bottomLeft_pos]
  return [get_rect_point(func(x1,y1,x2,y2,w,h)) for func in func_list]

def mstmap_generation_RightOne(path,from_start = False,duration=10,isnormalized = True):
  vid = cv2.VideoCapture(path)
  fps = vid.get(cv2.CAP_PROP_FPS)
  #print(fps)
  total_frame = count_frames(path)
  frame_count = 0
  if from_start: 
    start_frame = 0
    end_frame = int(start_frame+duration*fps)-1
  else:
    start_frame = int(total_frame/2)
    end_frame = int(start_frame+duration*fps)-1 #start_frame+300
  #print(start_frame,end_frame)
  region_num_face = 6
  region_num_bg = 4
  color_channel = 3
  mstmap_whole_face = np.zeros((2**region_num_face-1,end_frame-start_frame+1,color_channel))
  mstmap_whole_bg = np.zeros((2**region_num_bg-1,end_frame-start_frame+1,color_channel))
  while(vid.isOpened()):
    ret, frame = vid.read()
    if ret == False: break
    if ret:
        frame = cv2.resize(frame, (0,0), fx = 1.5, fy = 1.5)
        h , w,_ = frame.shape
        faceBoxes, faceLandmarksFivePts, alignedFaces = _faceDetectorAndAlignment.detect(frame)
        if len(faceBoxes) > 0:
            faceLandmarks = _faceLandmarks.extractLandmark(frame, faceBoxes)

            for faceBox in faceBoxes:
                x1,y1,x2,y2,_ = faceBox.astype(np.int32)
                #cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                #face_contours = get_face_contours(x1,y1,x2,y2)
                face_contours = get_face_contours(faceLandmarks)
                bg_contours = get_bg_contours(x1,y1,x2,y2,w,h)
            # for currentFaceLandmark in faceLandmarks:
            #     for pts in currentFaceLandmark:
            #         x, y = pts.astype(np.int32)
            #         cv2.circle(frame, (x, y), 2, (0,255,0), -1)
    frame_count += 1
    if frame_count < start_frame : continue
    mstmap_whole_face = generateSignalMap(mstmap_whole_face,frame,frame_count-start_frame,face_contours)
    mstmap_whole_bg = generateSignalMap(mstmap_whole_bg,frame,frame_count-start_frame,bg_contours)
    if frame_count == end_frame: break
  if isnormalized:
    final_map_face = mstmap_whole_face
    for idx in range(np.size(final_map_face,0)):
      for c in range(color_channel):
        temp = mstmap_whole_face[idx,:,c]
        temp =  moving_average(temp,3)
        temp[temp<0.8*np.mean(temp)] = temp[temp>=0.8*np.mean(temp)][0]
        final_map_face[idx,:,c] = (temp - np.amin(temp)) / (np.amax(temp) - np.amin(temp)) * 255
    #cv2_imshow(final_map_face)

    final_map_bg = mstmap_whole_bg
    for idx in range(np.size(final_map_bg,0)):
      for c in range(color_channel):
        temp = mstmap_whole_bg[idx,:,c]
        temp =  moving_average(temp,3)
        temp[temp<0.8*np.mean(temp)] = temp[temp>=0.8*np.mean(temp)][0]
        final_map_bg[idx,:,c] = (temp - np.amin(temp)) / (np.amax(temp) - np.amin(temp)) * 255
    #cv2_imshow(final_map_bg)
    return final_map_face,final_map_bg
  else:
    return mstmap_whole_face, mstmap_whole_bg

path = '/mnt/storage/shared/datasets/face_dataset/3DMAD-60fps'
attack_path = [join(path, f) for f in listdir(path) if f.split('_')[1]=='03']
real_path = [join(path, f) for f in listdir(path) if f.split('_')[1]!='03']
path = '/mnt/storage/shared/datasets/face_dataset/3DMAD MSTmap'
mkdir(path)
mkdir(join(path, 'attack'))
mkdir(join(path, 'real'))
mkdir(join(path, 'bg'))
path_bg = path + '/bg'
mkdir(join(path_bg, 'attack'))
mkdir(join(path_bg, 'real'))

for i in range(len(attack_path)):
  print('attack',i)
  filename_face = path+'/attack/'+attack_path[i].split('/')[-1][:-3]+'bmp'
  filename_bg = path_bg+'/attack/'+attack_path[i].split('/')[-1][:-3]+'bmp'
  mstmap_rightone_face, mstmap_rightone_bg = mstmap_generation_RightOne(attack_path[i],from_start = True,duration = 5)
  cv2.imwrite(filename_face,mstmap_rightone_face)
  cv2.imwrite(filename_bg,mstmap_rightone_bg)

for i in range(len(real_path)):
  print('real',i)
  filename_face = path+'/real/'+real_path[i].split('/')[-1][:-3]+'bmp'
  filename_bg = path_bg+'/real/'+real_path[i].split('/')[-1][:-3]+'bmp'
  mstmap_rightone_face, mstmap_rightone_bg = mstmap_generation_RightOne(real_path[i],from_start = True,duration = 5)
  cv2.imwrite(filename_face,mstmap_rightone_face)
  cv2.imwrite(filename_bg,mstmap_rightone_bg)