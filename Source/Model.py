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
from sklearn import decomposition   # For PCA
from sklearn import tree            # Decision Tree Classifier
from sklearn.svm import SVC         # SVM Classifier
from sklearn import metrics
import random
import math


class ModelSVM:
    def __init__(self, kernel='poly', degree=3, coef0=0, verbose=False, gamma='auto', reg=1/1000):
        """
        Initialize the parameters of the model SVM.
        Kernel = Name of a kernel (linear, poly, rbf, sigmoid)
        degree = Polynomial degree
        gamma = Hyperparameters for kernel (poly, rbf, sigmoid)
        coef0 = coef0 of the linear model
        verbose = See the training
        C = Inversely proportional to the regularization parameter
        Ref : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        """
        print("Initialize model SVM")
        # Check if gamma='auto' is useful
        self.kernel = kernel
        self.mod = SVC(gamma=gamma, kernel=kernel, degree=degree, coef0=coef0, verbose=verbose, C=reg)

    def train(self, x, t):
        """
        Training the SVM model with the data.
        :param x: A numpy array of the dataset
        :param t: A numpy array of the classes
        Comments : Remove self.w and self.w_0?
        """
        self.mod.fit(x, t)

    def cross_validation(self, x, t, k=5):
        """
        Do a k-fold cross validation on the hyper parameters.
        The hyper parameters are
        reg : 10^-3 to 10^3 in a log scale
        gamma : 10^-1 to 10^1 in a log scale
        degree:  1 to 10
        coef : 10^-1 to 10^1 in a log scale
        :param x: Dataset
        :param t: Classes
        """
        print("Cross validation of the  SVM Model...")

        # Initialize best error / hyperparameters
        best_error = float('inf')
        best_reg = 0
        best_gamma = 0
        best_deg = 0
        best_coef = 0

        # Cross-validation 80-20
        N = len(x)
        N_train = math.floor(0.8 * N)
        t = t.reshape((N,))

        #Initialize the grid search

        log_min_reg = np.log(0.001)
        log_max_reg = np.log(1000)
        reg_list = np.logspace(log_min_reg, log_max_reg, num=7, base=math.e)

        log_min_gamma = np.log(0.1)
        log_max_gamma = np.log(10)
        gamma_list = np.logspace(log_min_gamma, log_max_gamma, num=3, base=math.e)

        min_deg = 1
        max_deg = 4

        log_min_coef = np.log(0.1)
        log_max_coef = np.log(10)
        coef_list = np.logspace(log_min_coef, log_max_coef, num=3, base=math.e)

        for deg in range(min_deg, max_deg):
            for reg in reg_list:
                for gamma in gamma_list:
                    for coef in coef_list:
                        errors = np.zeros(k)

                        for j in range(k):
                            map_index = list(zip(x, t))
                            random.shuffle(map_index)
                            random_x, random_t = zip(*map_index)

                            train_x = random_x[:N_train]
                            valid_x = random_x[N_train:]
                            train_t = random_t[:N_train]
                            valid_t = random_t[N_train:]

                            self.mod = SVC(gamma=gamma, kernel='poly', degree=deg, coef0=coef, C=reg, cache_size=4000)
                            self.train(train_x, train_t)

                            error_valid = np.array([self.error(x_n, t_n)
                                                    for t_n, x_n in zip(valid_t, valid_x)])
                            errors[j] = error_valid.mean()

                        mean_error = np.mean(errors)
                        print(mean_error)
                        if mean_error < best_error:
                            best_error = mean_error
                            best_reg = reg
                            best_gamma = gamma
                            best_deg = deg
                            best_coef = coef
                            print("The new best hyper parameters are : ", best_reg, best_gamma, best_deg, best_coef)

        print("Best hyper parameters are : ", best_reg, best_gamma, best_deg, best_coef)
        print("Validation error : ", 100 * best_error, "%")
        self.mod = SVC(gamma=best_gamma, kernel='poly', degree=best_deg, coef0=best_coef, C=best_reg)
        self.train(x, t)

    def prediction(self, x):
        """
        Predict the classe from the data.
        :param x: One sample of data
        :return:  The predict classes from the SVM model.
        """
        t = self.mod.predict(x.reshape(1, -1))
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


class ModelDecisionTree:
    def __init__(self, max_depth=None, criterion='gini'):
        """
        Initialize the model of a Decision Tree Classifier
        """
        print("Initialize the model Decision Tree Classifier... ")
        self.mod = tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        self.pca = decomposition.PCA()

    def train(self, x, t):
        """
        Training the model of a Decision Tree Classifier
        :param x: A numpy array of the dataset
        :param t: A numpy array of a classes of the dataset.
        """
        self.pca.fit(x)
        x_red = self.pca.transform(x)
        self.mod.fit(x_red, t)

    def cross_validation(self, x, t, k=5):
        """
        Do a k-fold cross validation to choose the best hyper parameters for the Decision Tree model.
        The hyper parameters are
        criteria : 'gini'  or 'entropy'
        Max_depth : 2 to 40
        pca_dim :  1 to 20
        :param x: A numpy array of the dataset
        :param t: A numpy array of a classes of the dataset.
        :param k: A int for a k-fold cross validation.
        """
        print("Cross validation of the Decision Tree Classifier...")
        bestCriteria = ''
        bestMax_depth= 2
        bestPcaDim = 1
        bestError = float('inf')

        N = len(x)
        N_train = math.floor(0.8 * N)

        dicCriteria = ['gini', 'entropy']
        min_depth = 2
        max_depth = 40
        min_pcaDim = 1
        max_pcaDim = 20

        for crit in dicCriteria:
            for pcaDim in range(min_pcaDim, max_pcaDim):

                for d in range(min_depth, max_depth):
                    errors = np.zeros(k)

                    for j in range(k):
                        map_index = list(zip(x, t))
                        random.shuffle(map_index)
                        random_X, random_t = zip(*map_index)

                        train_x = random_X[:N_train]
                        valid_x = random_X[N_train:]
                        train_t = random_t[:N_train]
                        valid_t = random_t[N_train:]

                        self.mod = tree.DecisionTreeClassifier(max_depth=d, criterion=crit)
                        self.pca = decomposition.PCA(n_components=pcaDim)
                        self.train(train_x, train_t)
                        error_valid = np.array([self.error(x_n, t_n)
                                            for t_n, x_n in zip(valid_t, valid_x)])
                        errors[j] = error_valid.mean()

                    mean_error = np.mean(errors)
                    if mean_error < bestError:
                        bestError = mean_error
                        bestCriteria = crit
                        bestMax_depth = d
                        bestPcaDim = pcaDim
                        print("The new best hyper parameters are : ", bestMax_depth, bestCriteria, bestPcaDim)

        print("Best hyper parameters are : ", bestMax_depth, bestCriteria, bestPcaDim)
        print("Validation error : ", 100 * bestError, "%")
        self.mod = tree.DecisionTreeClassifier(max_depth=bestMax_depth, criterion=bestCriteria)
        self.pca = decomposition.PCA(n_components=bestPcaDim)
        self.train(x, t)

    def prediction(self, x):
        """
        Prediction of a class from one sample of data.
        :param x: One sample of data.
        :return: Predict the class the sample of data
        """
        x_red = self.pca.transform(x.reshape(1, -1))
        self.mod.predict(x_red)

    def error(self, x, t):
        """
        Compute the error between the predicted class and the class from data
        if is not the same class : return 0
        if is the same class   : return 1
        :param x : One sample of data
        :param t : Class of the sample data
        """
        x_red = self.pca.transform(x.reshape(1, -1))
        predict = self.mod.predict(x_red)
        if t == predict:
            return 0
        else:
            return 1

