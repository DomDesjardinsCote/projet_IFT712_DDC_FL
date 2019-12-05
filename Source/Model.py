# -*- coding: utf-8 -*-
"""class Model:
    code for a general model
    def __init__(self):
        pass

    def train(self, x, t):
        pass

    def cross_validation(self, x, t):
        pass

    def prediction(self, x, t):
        pass

    def error(self, x, t):
        pass
"""
import numpy as np
from sklearn import tree            # Decision Tree Classifier
from sklearn.svm import SVC         # SVM Classifier
class ModelSVM:
    def __init__(self, kernel='rbf', degree=3, coef0=0, verbose=False, classes=7):
        """
        Initialize the parameters of the model SVM.
        Kernel = Name of a kernel (linear, poly, rbf, sigmoid)
        degree = Polynomial degree
        coef0= coef0 of the linear model
        verbose = See the trainning
        Reference : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        """
        print("Initialize model SVM")
        # Check if gamma='auto' is useful
        self.kernel = kernel
        self.mod = SVC(gamma='auto', kernel=kernel, degree=degree, coef0=coef0, verbose=verbose)
        self.w = 0
        self.w_0 = 0
       # self.mod.classes_(classes)

    def train(self, x, t):
        """
        Training the SVM model with the data.
        :param x: A numpy array of the data
        :param t: A numpy array of the classes
        Comments : Remove self.w and self.w_0?
        """
        print("Training of the Model SVM")
        self.mod.fit(x, t)
        self.w = self.mod.dual_coef_
        self.w_0 = self.mod.intercept_

    def cross_validation(self, x, t):
        """
        Do a cross validation on the hyperparameters.(Need to fine the hyper parameters for a cross-validation)
        :param x: Dataset
        :param t: Classes
        """
        print("TODO : Crossvalidation Model SVM...")

    def prediction(self, x):
        """
        Predict the classe from the data.
        :param x: One sample of data
        :return:  The predict classes from the SVM model.
        """
        t = self.mod.predict(x)
        return t

    def error(self, x, t):
        """
        Compute the error between the predicted class and the class from data
        if is not the same class : return 0
        if is the same class   : return 1
        :param x : One sample of data
        :param t : Class of the sample data
        """
        predict = self.mod.predict(x.reshape(1, -1))
        if t == predict:
            return 0
        else:
            return 1

