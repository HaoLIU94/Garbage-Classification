#!/usr/bin/env python
# coding: utf-8

import glob
import shutil
import os

src_dir = "dataset-resized"
dst_dir = "GARBAGES"

train = '/train/'
test = '/test/'
val = '/val/'
for cal in os.listdir(src_dir):
    images = []
    for jpgfile in glob.iglob(os.path.join(src_dir+"/"+cal, "*.jpg")):
        images.append(jpgfile)
    #images = sorted(images)
    
    direct_train = dst_dir+train+cal+"/" 
    direct_test = dst_dir+test+cal+"/"
    direct_val = dst_dir+val+cal+"/"
    os.makedirs(direct_train)
    os.makedirs(direct_val)
    os.makedirs(direct_test)
    count = 0
    l = len(images)
    for jpgfile in images:
        #Split 400 to train
        if count>=0 and count<0.7*l:
            shutil.copy(jpgfile, direct_train)
        #Split 50 to val
        if count>=0.7*l and count<0.8*l:
            shutil.copy(jpgfile, direct_val)
        #Split 250 to test
        if count>=0.9*l and count<l:
            shutil.copy(jpgfile, direct_test)
        count +=1

