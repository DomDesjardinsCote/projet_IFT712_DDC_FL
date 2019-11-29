# -*- coding: utf-8 -*-
print "HELLO DOM"

import pandas as pd
import csv
import numpy as np
import os

np.random.seed(0)

################## Load Data
base_dir = os.path.join( os.path.dirname ( __file__), '..' )
csv = pd.read_csv(base_dir + "/Database/train.csv")
#csv_test = pd.read_csv(base_dir + "/Database/test.csv")
data = np.array(csv, dtype=np.int)
#data_test = np.array(csv_train, dtype=np.int)

X_ = data[:,1:-1]
y_ = data[:,-1:] 
#X_test = data_test[:,1:-1]

################## Get attributes
sample_size_train = data.shape[0]
#sample_size_test = data_test.shape[0]
num_features = X_.shape[1]
features_name = list(csv.columns.values)
num_classes = np.unique(y_).shape[0]
print('Nombre de classes: ' + str(num_classes))
print('Nom des features: ' + str(features_name))
print('Nombre de features: ' + str(num_features))
print('Nombre de donnees: ' + str(sample_size_train))

# Centrer et réduire les données (moyenne = 0, écart-type = 1)
mean = np.mean(X_, axis=0)
std = np.std(X_, axis=0)
print ("mean :", mean)
print ("std :", std)
X_Norm = (X_ - mean) / std

# Séparer les données localement en (train, valid, test)