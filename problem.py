# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
import torchvision
from torch.autograd import Variable 
from torchvision import transforms
from tqdm import tqdm

problem_title = 'Garbage-Classification'
_target_column_name = 'type'
_prediction_label_names = [0, 1, 2, 3, 4, 5] 
name = ["glass","paper","metal","plastic","cardboard","trash"]


# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
# workflow = rw.workflows.FeatureExtractorClassifier()
workflow = rw.workflows.Classifier()
#Soft_score_matrix generation
soft_score_matrix=[]
toadd= []
matcher = 0
for actual in name:
    for autre in name:
        if actual[0]==autre[0]:
            matcher = matcher + 0.25
        if actual[1]==autre[1]:
            matcher = matcher + 0.25
        if actual[2]==autre[2]:
            matcher = matcher + 0.25
        if actual[3]==autre[3]:
            matcher = matcher + 0.25
        toadd.append(matcher)
        matcher = 0
        
    #Ici on a fini une ligne, on va ajouter la ligne à la matrice
    soft_score_matrix.append(toadd)
    toadd = []
soft_score_matrix = np.array(soft_score_matrix)
        

# soft_score_matrix = np.array([
    # [1, 0.8, 0, 0, 0, 0],
    # [0.4, 1, 0.4, 0, 0, 0],
    # [0, 0.4, 1, 0.4, 0, 0],
    # [0, 0, 0.4, 1, 0.4, 0],
    # [0, 0, 0, 0.4, 1, 0.4],
    # [0, 0, 0, 0, 0.8, 1],
# ])


# true_false_score_matrix = np.array([
    # [1, 1, 1, 0, 0, 0],
    # [1, 1, 1, 0, 0, 0],
    # [1, 1, 1, 0, 0, 0],
    # [0, 0, 0, 1, 1, 1],
    # [0, 0, 0, 1, 1, 1],
    # [0, 0, 0, 1, 1, 1],
# ])

# score_types = [
    # # rw.score_types.SoftAccuracy(
        # # name='sacc', score_matrix=soft_score_matrix, precision=3),
    # rw.score_types.Accuracy(name='acc', precision=3),
    # # rw.score_types.SoftAccuracy(
        # # name='tfacc', score_matrix=true_false_score_matrix, precision=3),
# ]

score_types = [
    rw.score_types.SoftAccuracy(name='sacc', score_matrix=soft_score_matrix, precision=3),
    rw.score_types.Accuracy(name='acc')
    #rw.score_types.NegativeLogLikelihood(name='nll'),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path='.', f_name='data'):
    pass


def get_train_data(path='.'):
    # f_name = 'train.csv'
    # image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
    #                                   data_transforms) for x in ['train']}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
    #                                      shuffle=True, num_workers=4) for x in ['train']}
    X = np.load("data/train.npy")
    y = np.load("data/train_label.npy")

    return X,y 


def get_test_data(path='.'):

    X = np.load("data/test.npy")
    y = np.load("data/test_label.npy")

    return X,y 

