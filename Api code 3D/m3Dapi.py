import json
import re
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

from TranRppg2 import TransRppg,ViTRppg,TransRppg2
from mstMap_gen import MSTmap_generator, get_face_contours, get_bg_contours, generateSignalMap, norm_mst

class FAS3D:
    def __init__(self):
        self.label = ['attack','real']
        ## Load model
        # self.model = []
        self.model = TransRppg2(emb_size=96,mlp_mul=2)
        load_path = '3D model\Tranrppg10sec new model_drop_5.pt'

        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()

        self.color_channel = 3
        self.time_frame = 300
        self.region_num_face = 6
        self.region_num_bg = 4
        self.frame_count = 0

        # self.reset_mst()
        self.mstmap_whole_face = dict()
        self.mstmap_whole_bg = dict()
        return

    def classifyVid(self,video,face_landmark,bg_landmark): #Receive Video as input
        ## recieve video data and face landmark
        ## create MSTmap from video and face landmark
        final_mstmap_face, final_mstmap_bg = MSTmap_generator(video,face_landmark,bg_landmark)
        ## classify Target
        result = self.model(final_mstmap_face,final_mstmap_bg)
        predict = np.argmax(result)
        if predict == 0:
            return 'attack'
        else:
            return 'real'
        # result = self.model(final_mstmap_face, final_mstmap_bg)
        # return final_mstmap_face, final_mstmap_bg
    
    def classifySeq(self,frame,face_landmark,bg_landmark,frame_count,cam_id):
        if cam_id not in self.mstmap_whole_face:
            self.reset_mst(cam_id)
        if frame_count==0:
            self.frame_count = 0
            self.reset_mst(cam_id)
        face_contours = get_face_contours(face_landmark)
        bg_contours = get_bg_contours(*bg_landmark)
        if frame_count>299:
            self.mstmap_whole_face[cam_id] = np.roll(self.mstmap_whole_face[cam_id], (0, -1), axis=(0, 1))
            self.mstmap_whole_bg[cam_id] = np.roll(self.mstmap_whole_bg[cam_id], (0, -1), axis=(0, 1))
        self.mstmap_whole_face[cam_id] = generateSignalMap(self.mstmap_whole_face[cam_id],frame,min(frame_count,299),face_contours)
        self.mstmap_whole_bg[cam_id] = generateSignalMap(self.mstmap_whole_bg[cam_id],frame,min(frame_count,299),bg_contours)
        if frame_count >= 299:
            ### Normalise MSTmap
            final_mstmap_face = norm_mst(self.mstmap_whole_face[cam_id],self.color_channel)
            final_mstmap_bg = norm_mst(self.mstmap_whole_bg[cam_id],self.color_channel)
            ### classify Target
            result = self.model(final_mstmap_face,final_mstmap_bg)
            predict = np.argmax(result)
            if predict == 0:
                return 'attack'
            else:
                return 'real'
            # return final_mstmap_face, final_mstmap_bg
        return None

    def reset_mst(self,cam_id):
        self.mstmap_whole_face[cam_id] = np.zeros((2**self.region_num_face-1,self.time_frame,self.color_channel))
        self.mstmap_whole_bg[cam_id] = np.zeros((2**self.region_num_bg-1,self.time_frame,self.color_channel))
        return