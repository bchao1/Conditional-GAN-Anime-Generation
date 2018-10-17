# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:40:34 2018

@author: USER
"""

import os
import cv2
import pickle

import random
import torch


        
class Anime:
    def __init__(self, root_dir, tags, transform):
        with open(tags, 'rb') as file:
            self.tags = pickle.load(file)
        self.root_dir = root_dir
        self.transform = transform
        self.img_files = os.listdir(self.root_dir)
        self.dataset_len = len(self.img_files)
    
    def length(self):
        return self.dataset_len
    
    def get_item(self, idx):
        hair_tag, eye_tag = self.tags[idx]
        
        img_path = os.path.join(self.root_dir, str(idx) + '.jpg')
        img = cv2.imread(img_path)
        img = img[:, :, (2, 1, 0)]
        if self.transform:
            img = self.transform(img)
        return img, hair_tag, eye_tag

class Shuffler:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_len = self.dataset.length()
    
    def get_batch(self):
        indices = random.sample(range(self.dataset_len), self.batch_size)
        img_batch, hair_tags, eye_tags = [], [], []
        for i in indices:
            img, hair, eye = self.dataset.get_item(i)
            img_batch.append(img.unsqueeze(0))
            hair_tags.append(hair.unsqueeze(0))
            eye_tags.append(eye.unsqueeze(0))
        img_batch = torch.cat(img_batch, 0)
        hair_tags = torch.cat(hair_tags, 0)
        eye_tags = torch.cat(eye_tags, 0)
        
        return img_batch, hair_tags, eye_tags
