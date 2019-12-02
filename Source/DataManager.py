# -*- coding: utf-8 -*-

import pandas as pd
import csv
import math
import numpy as np
import os

class DataManager:
    def __init__(self, nb_train, nb_test, normalisation=True, bias=True):
        self.nb_train = nb_train
        self.nb_test = nb_test
        self.normalisation = normalisation
        self.bias = bias
    
    def generer_donnees(self):
        base_dir = os.path.join( os.path.dirname ( __file__), '..' )
        csv = pd.read_csv(base_dir + "/Database/train.csv")
        data = np.array(csv, dtype=np.int)

        X_ = data[:,1:-1]
        y_ = data[:,-1:] 

        # Centrer et réduire les données (moyenne = 0, écart-type = 1)
        if self.normalisation:
            mean = np.mean(X_, axis=0)
            std = np.std(X_, axis=0)
            X_ = (X_ - mean) / std
        
        if self.bias:
            X_ = augment(X_)

        idx = np.random.permutation(len(X_Norm))

        train_idx = idx[:self.nb_train]
        test_idx = idx[-self.nb_test:]

        X_train = X_[train_idx]
        y_train = y_[train_idx]
        X_test = X_[test_idx]
        y_test = y_[test_idx]
        return X_train, y_train, X_test, y_test

def augment(X):
    if len(X.shape) == 1:
        return np.concatenate([X, [1.0]])
    else:
        return np.concatenate([X, np.ones((len(X), 1))], axis=1)
