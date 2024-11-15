from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import numbers
import random
from torchvision import transforms 
import cv2 
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
import csv

import copy

import matplotlib.pyplot as plt 


def pil_to_cv2(pil_image): # New
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)     
    return cv2_image

def cv2_to_gray(cv2_image):
    gray_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    return gray_image

def flow_to_pil(flow_input):
    return flow_input

def get_optical_flow_image_from_features(flow_x, flow_y, mask, count, im1, im2, halve_features, show_image=False):
    # converting to dense optical flow based on https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mask = np.float32(mask)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    if show_image:
        print(count)
        temp_im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
        temp_im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
        if halve_features:
            try:
                os.mkdir("./drift_half")
            except:
                pass
            cv2.imwrite(f"./drift_half/{count}.png", rgb)
        else:
            three_images = np.concatenate( (np.concatenate((temp_im1, temp_im2), axis=1), rgb), axis=1)
            try:
                os.mkdir("./drift")
            except:
                pass
            cv2.imwrite(f"./drift/{count}.png", three_images)

    return rgb


def compute_optical_flow(video_frames, halve_features = False, save_image = False): # New
    flows = []
    im1_pil = video_frames[0]
    im1 = pil_to_cv2(im1_pil)

    if halve_features:
        mask = np.zeros((int(im1.shape[0]/2), int(im1.shape[1]/2), 3))
    else:
        mask = np.zeros_like(im1)

    mask[..., 1] = 255
    count = 0

    flow = None
    im1 = cv2_to_gray(im1)
    for i, im2_pil in enumerate(video_frames[1:]):
        im2 = cv2_to_gray(pil_to_cv2(im2_pil))
        flow = cv2.calcOpticalFlowFarneback(im1, im2, flow, 
							pyr_scale = 0.5, levels = 1, iterations = 1, 
							winsize = 11, poly_n = 5, poly_sigma = 1.1,  
							flags = 0 if flow is None else cv2.OPTFLOW_USE_INITIAL_FLOW )
        #print(f"shape of a single flow is {flow.shape}") # (360, 640, 2) = (ht, width, 2) where 2 is for flow in x and y directions
        
        if halve_features:
            flow_x = cv2.resize(flow[..., 0], None, fx=0.5, fy=0.5)
            flow_y = cv2.resize(flow[..., 1], None, fx=0.5, fy=0.5)
        else:
            flow_x, flow_y = flow[..., 0], flow[..., 1]

        flow_image = get_optical_flow_image_from_features(flow_x, flow_y, mask, count, im1, im2, halve_features, show_image=save_image)
        count += 1

        # move im1 forward
        im1 = im2

        # print(flow_image.shape) # (ht, wdth, 3)
        # flow_image = flow_image.transpose((2, 0, 1))
        # print(flow_image.shape)

        flows.append(flow_image)
    
    if save_image:
        exit(0)
    return flows


class DriftDataset(data.Dataset):
    """Drift dataset
    Args:
        root_dir (string): Directory with training/test data.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self,
                root_dir,
                clip_len=16,
                train=True,
                cal=False,
                transforms_=None,
                img_hgt=360,
                img_width=640,
                in_dist_test=False,
                use_image = True,
                use_of = True,
                transformation_list = ["speed","shuffle","reverse","periodic","identity"]):
        
        assert use_of or use_image, "Atleast one of the use_image or use_of should be true"
        self.root_dir = root_dir
        self.clip_total_frames = clip_len
        self.train = train
        self.cal = cal
        self.in_dist_test = in_dist_test
        self.transforms_ = transforms_
        self.img_hgt = img_hgt
        self.img_width = img_width
        self.videos = []
        self.use_image = use_image
        self.use_of = use_of
        self.tranformation_list = transformation_list
        self.num_classes = len(self.tranformation_list)


        if self.train:

            #print("Training")
            no_episodes = 24 #new
            # no_episodes = 14 #old
            
            for episode in range(1,no_episodes+1): 
                #print("episode: ", episode)
                file = open("{}/{}/n_frames".format(root_dir, episode), 'r')
                no_images = int(file.read())
                video_frames = [] 
                for img in range(1,no_images+1):
                    #print("Image: ", img)
                    with Image.open('{}/{}/image_{}.jpg'.format(root_dir, episode, (str(img).zfill(5)))) as im: 
                        video_frames.append(im.resize((self.img_width, self.img_hgt)))  
                # video_frames = compute_optical_flow(video_frames, halve_features = False, save_image = False) # New
                self.videos.append(video_frames)

        elif self.cal:
            #print("Calibration")
            no_episodes = 14 #new
            # no_episodes = 9 #old


            for episode in range(1,no_episodes+1): 
                #print("Episode: ", episode)
                file = open("{}/{}/n_frames".format(root_dir, episode), 'r')
                no_images = int(file.read())
                video_frames = [] 
                for img in range(1,no_images+1):
                    #print("Image: ", img)
                    with Image.open('{}/{}/image_{}.jpg'.format(root_dir, episode, (str(img).zfill(5)))) as im: 
                        video_frames.append(im.resize((self.img_width, self.img_hgt)))  
                # video_frames = compute_optical_flow(video_frames, halve_features = False, save_image = False) # New
                self.videos.append(video_frames)

        else:

            if self.in_dist_test==True: # Testing for in-distribution 
                no_episodes = 34 #new
                # no_episodes = 9 #old
                ext = ''
            else:
                no_episodes = 100
                ext='_1'

            for episode in range(1,no_episodes+1):
                file = open("{}/{}{}/n_frames".format(root_dir, episode, ext), 'r') # episode
                no_images = int(file.read())
                video_frames = [] 
                for img in range(1,no_images+1):
                    with Image.open('{}/{}{}/image_{}.jpg'.format(root_dir, episode, ext, (str(img).zfill(5)))) as im:  # episode
                        video_frames.append(im.resize((self.img_width, self.img_hgt)))
                # video_frames = compute_optical_flow(video_frames, halve_features = False, save_image = False) # New
                self.videos.append(video_frames)      
    
    def __len__(self):
        return len(self.videos)

    def transform_clip(self, clip):
        # trans_clip = []
        
        # for frame in clip:
        #     frame = self.transforms_(frame) # from  img [H x W x C]  to tensor [C x H x W]
        #     trans_clip.append(frame)
        # trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
        # return trans_clip

        trans_clip = list(map(lambda x:self.transforms_(x), clip))
        trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])

        return trans_clip

    def __getitem__(self, idx): # this is only for CAL/TRAIN
        """
        Returns:
            original video frames, transformed video frames and the applied transformation
        """
        # print(idx)
        videodata = self.videos[idx]

        length = len(videodata)

        clip_start = random.randint(0, length - self.clip_total_frames)

        orig_clip =  videodata[clip_start:clip_start+self.clip_total_frames]
        trans_clip = videodata[clip_start:clip_start+self.clip_total_frames]

        # random transformation selection with 0: 2x speed 1: shuffle, 2: reverse, 3: periodic (forward, backward), 4: Identity
        transform_id = random.randint(0, self.num_classes-1)

        if self.tranformation_list[transform_id] == "speed": #  2x speed 
            
            clip_start = random.randint(0, length-(2*self.clip_total_frames)) # multiplying 2 for 2x speed

            orig_clip =  videodata[clip_start:clip_start+self.clip_total_frames]

            trans_clip = videodata[clip_start:clip_start+2*self.clip_total_frames:2]

        elif self.tranformation_list[transform_id] == "shuffle": # shuffle
            random.shuffle(trans_clip)

        elif self.tranformation_list[transform_id] == "reverse": # reverse
            trans_clip.reverse()

        elif self.tranformation_list[transform_id] == "periodic": # periodic (forward, backward)
           trans_clip[self.clip_total_frames//2:self.clip_total_frames] = reversed(trans_clip[self.clip_total_frames//2:self.clip_total_frames]) 
        
        elif self.tranformation_list[transform_id] == "identity": #do nothing
            pass

        else:
            raise Exception("Invalid transformation")
        
        
        if self.use_of:
            # applying optical flow on original and transformed clips and apply tensorization
            # print("Use of")
            trans_clip_flow = compute_optical_flow(trans_clip, halve_features = False, save_image = False)
            orig_clip_flow = compute_optical_flow(orig_clip, halve_features = False, save_image = False)
            trans_clip_flow = self.transform_clip(trans_clip_flow)
            orig_clip_flow = self.transform_clip(orig_clip_flow)

        
        if self.use_image:
            # print("Use image")
            # converting to tensors and new dim = C X no. of frames X H X W, if use_image is true
            trans_clip = self.transform_clip(trans_clip)
            orig_clip = self.transform_clip(orig_clip)

        if self.use_of and self.use_image:
            # print("Use both")
            # Removing first frame from the video data to make it consistent with the flow data as flow data has 1 frame lesser than the video data
            orig_clip = orig_clip[:,1:,]
            trans_clip = trans_clip[:,1:,]
            # now concatinating along channel dim
            orig_clip = torch.cat((orig_clip, orig_clip_flow), dim=0)
            trans_clip = torch.cat((trans_clip, trans_clip_flow), dim=0)

        # print("After concat, concatenated clip dim: ", orig_clip.shape)

        if not self.use_image:
            return orig_clip_flow, trans_clip_flow, transform_id
        else:
            return orig_clip, trans_clip, transform_id

    def __get_test_item__(self, idx): # generator for getting sequential shuffled tuples/windows on test data

        videodata = self.videos[idx]
    
        length = len(videodata)

        # last_win_starting_point = max(length-(2*self.clip_total_frames),1) # to get at least one datapoint from the trace if the trace is too short (total len = 2*clip_total_frames)

        for i in range(0, length-(2*self.clip_total_frames)+1): 
            clip_start = i

            transform_id = random.randint(0, self.num_classes-1)
            
            orig_clip =  videodata[clip_start:clip_start+self.clip_total_frames]
            trans_clip = videodata[clip_start:clip_start+self.clip_total_frames]

            if self.tranformation_list[transform_id] == "speed": 
                
                trans_clip = videodata[clip_start:clip_start+2*self.clip_total_frames:2]

            elif self.tranformation_list[transform_id] == "shuffle":

                random.shuffle(trans_clip)

            elif self.tranformation_list[transform_id] == "reverse": 

                trans_clip.reverse()

            elif self.tranformation_list[transform_id] == "periodic":  # periodic (forward, backward)
                trans_clip[self.clip_total_frames//2:self.clip_total_frames] = reversed(trans_clip[self.clip_total_frames//2:self.clip_total_frames])

            elif self.tranformation_list[transform_id] == "identity": 
                pass 
            
            else:
                raise Exception("Invalid transformation id: ", transform_id)

            if self.use_of:
                # applying optical flow on original and transformed clips and apply tensorization
                trans_clip_flow = compute_optical_flow(trans_clip, halve_features = False, save_image = False)
                orig_clip_flow = compute_optical_flow(orig_clip, halve_features = False, save_image = False)
                trans_clip_flow = self.transform_clip(trans_clip_flow)
                orig_clip_flow = self.transform_clip(orig_clip_flow)

            
            if self.use_image:
                # converting to tensors and new dim = C X no. of frames X H X W, if use_image is true
                trans_clip = self.transform_clip(trans_clip)
                orig_clip = self.transform_clip(orig_clip)

            if self.use_of and self.use_image:
                # Removing first frame from the video data to make it consistent with the flow data as flow data has 1 frame lesser than the video data
                orig_clip = orig_clip[:,1:,]
                trans_clip = trans_clip[:,1:,]
                # now concatinating along channel dim
                orig_clip = torch.cat((orig_clip, orig_clip_flow), dim=0)
                trans_clip = torch.cat((trans_clip, trans_clip_flow), dim=0)

            # print("After concat, concatenated clip dim: ", orig_clip.shape)
            if not self.use_image:
                yield orig_clip_flow, trans_clip_flow, transform_id
            else:
                yield orig_clip, trans_clip, transform_id
