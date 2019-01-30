# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Garbage-Classification'
_target_column_name = 'type'
_prediction_label_names = [0, 1, 2, 3, 4, 5] 
name = ["glass","paper","metal","plastic","cardboard","trash"]


# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.ImageClassifier(16, 50, 1, len(_prediction_label_names))

score_types = [
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.BalancedAccuracy(name='balanced_acc')
]


def get_cv(folder_X, y):
    _, X = folder_X
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    df = pd.read_csv(os.path.join(path, 'data', f_name))
    X = df['id'].values
    y = df['class'].values
    folder = os.path.join(path, 'data', 'imgs')
    return (folder, X), y


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)
