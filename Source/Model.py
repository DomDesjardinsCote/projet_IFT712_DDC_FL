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
import math
import numpy as np
import random

from sklearn import tree
from sklearn.svm import SVC
import sklearn.linear_model
import sklearn.metrics
import sklearn.neural_network


class ModelSVM:
    def __init__(self, kernel='poly', degree=3, verbose=False, gamma='auto', reg=1/1000, random_state=0):
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
        self.random_state = random_state
        self.model = SVC(gamma=gamma, kernel=kernel, degree=degree, verbose=verbose, C=reg, random_state=random_state)

    def train(self, x, t):
        """
        Training the SVM model with the data.
        :param x: A numpy array of the dataset
        :param t: A numpy array of the classes
        Comments : Remove self.w and self.w_0?
        """
        self.model.fit(x, t)

    def cross_validation(self, x, t, k=5):
        """
        Do a k-fold cross validation on the hyper parameters.
        The hyper parameters are
        reg : 10^-3 to 10^3 in a log scale
        degree:  1 to 10
        :param x: Dataset
        :param t: Classes
        :param k: k-fold crossvalidation
        """
        print("Cross validation of the  SVM Model...")

        # Initialize best error / hyperparameters
        best_error = float('inf')
        best_reg = 0
        best_deg = 0

        # Cross-validation 80-20
        N = len(x)
        N_train = math.floor(0.8 * N)
        t = t.reshape((N,))

        #Initialize the grid search

        log_min_reg = np.log(0.001)
        log_max_reg = np.log(1000)
        reg_list = np.logspace(log_min_reg, log_max_reg, num=7, base=math.e)

        min_deg = 1
        max_deg = 4

        for deg in range(min_deg, max_deg):
            for reg in reg_list:
                errors = np.zeros(k)
                for j in range(k):
                    map_index = list(zip(x, t))
                    random.shuffle(map_index)
                    random_x, random_t = zip(*map_index)

                    train_x = random_x[:N_train]
                    valid_x = random_x[N_train:]
                    train_t = random_t[:N_train]
                    valid_t = random_t[N_train:]

                    self.model = SVC(gamma='auto', kernel='poly', degree=deg, C=reg, cache_size=1000)
                    self.train(train_x, train_t)

                    error_valid = np.array([self.error(x_n, t_n) for t_n, x_n in zip(valid_t, valid_x)])
                    errors[j] = error_valid.mean()

                    mean_error = np.mean(errors)
                    print(mean_error)
                    if mean_error < best_error:
                        best_error = mean_error
                        best_reg = reg
                        best_deg = deg
                        print("The new best hyper parameters are : ", best_reg, best_deg)

        print("Best hyper parameters are : ", best_reg, best_deg)
        print("Validation error : ", 100 * best_error, "%")
        self.model = SVC(gamma='auto', kernel='poly', degree=best_deg, C=best_reg)
        self.train(x, t)

    def prediction(self, x):
        """
        Predict the classe from the data.
        :param x: One sample of data
        :return:  The predict classes from the SVM model.
        """
        t = self.model.predict(x.reshape(1, -1))
        return t

    def error(self, x, t):
        """
        Compute the error between the predicted class and the class from data
        if is not the same class : return 0
        if is the same class   : return 1
        :param x : One sample of data
        :param t : Class of the sample data
        """
        predict = self.model.predict(x.reshape(1, -1))
        if t == predict:
            return 0
        else:
            return 1


class ModelDecisionTree:
    def __init__(self, max_depth=None, criterion='gini', random_state=0):
        """
        Initialize the model of a Decision Tree Classifier
        """
        print("Initialize the model Decision Tree Classifier... ")
        self.random_state = random_state
        self.model = tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)

    def train(self, x, t):
        """
        Training the model of a Decision Tree Classifier
        :param x: A numpy array of the dataset
        :param t: A numpy array of a classes of the dataset.
        """
        self.model.fit(x, t)

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
        bestError = float('inf')

        N = len(x)
        N_train = math.floor(0.8 * N)

        dicCriteria = ['gini', 'entropy']
        min_depth = 2
        max_depth = 40

        for crit in dicCriteria:
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

                        self.model = tree.DecisionTreeClassifier(max_depth=d, criterion=crit)
                        self.train(train_x, train_t)
                        error_valid = np.array([self.error(x_n, t_n)
                                            for t_n, x_n in zip(valid_t, valid_x)])
                        errors[j] = error_valid.mean()

                    mean_error = np.mean(errors)
                    if mean_error < bestError:
                        bestError = mean_error
                        bestCriteria = crit
                        bestMax_depth = d
                        print("The new best hyper parameters are : ", bestMax_depth, bestCriteria)

        print("Best hyper parameters are : ", bestMax_depth, bestCriteria, bestPcaDim)
        print("Validation error : ", 100 * bestError, "%")
        self.model = tree.DecisionTreeClassifier(max_depth=bestMax_depth, criterion=bestCriteria)
        self.train(x, t)

    def prediction(self, x):
        """
        Prediction of a class from one sample of data.
        :param x: One sample of data.
        :return: Predict the class the sample of data
        """
        t = self.model.predict(x)
        return t

    def error(self, x, t):
        """
        Compute the error between the predicted class and the class from data
        if is not the same class : return 0
        if is the same class   : return 1
        :param x : One sample of data
        :param t : Class of the sample data
        """
        predict = self.model.predict(x)
        if t == predict:
            return 0
        else:
            return 1

        print("Error Model 1...")


class Perceptron:
    def __init__(self, num_features, num_classes, lamb=0.0001, r_s=0):
        self.lamb = lamb
        self.model = sklearn.linear_model.Perceptron(penalty='l2', alpha=self.lamb, max_iter=1000, random_state=r_s)

    def train(self, x_train, y_train):

        self.model.fit(x_train, y_train)
        print "Result train accuracy:", self.model.score(x_train, y_train)
        prediction = self.prediction(x_train)
        Confusion_M = sklearn.metrics.confusion_matrix(y_train, prediction)
        print "Confusion matrix:", Confusion_M

    def prediction(self, x):
        if len(x.shape)==1:
            x = np.reshape(x, (1, x.shape[0]))
        predict = self.model.predict(x)

        return predict

    def cross_validation(self, X, y, k_fold=10):
        best_accuracy = 0.0
        N = X.shape[0]
        N_train = int(math.floor(0.8 * N))

        min_lamb = 0.000000001
        max_lamb = 2.0
        log_min_lamb = math.log(min_lamb)
        log_max_lamb = math.log(max_lamb)
        lamb_list = np.logspace(log_min_lamb, log_max_lamb, num=100, base=math.e)

        best_lamb = self.lamb

        for lamb in lamb_list:
            self.lamb = lamb
            print self.lamb
            accuracy = np.zeros((k_fold))
            for i in range(k_fold):
                map_index = list(zip(X, y))
                random.shuffle(map_index)
                random_X, random_y = zip(*map_index)

                train_X = np.array(random_X[:N_train])
                valid_X = random_X[N_train:]
                train_y = np.array(random_y[:N_train])
                valid_y = random_y[N_train:]

                self.train(train_X, train_y)
                accuracy[i] = self.model.score(valid_X, valid_y)

            mean_accuracy = np.mean(accuracy)
            print mean_accuracy
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_lamb = self.lamb
        self.lamb = best_lamb
        self.train(X, y)
        print "best lamb", best_lamb
        print "best accuracy", best_accuracy
        print "Result train accuracy:", self.model.score(X, y)
        prediction = self.prediction(X)
        Confusion_M = sklearn.metrics.confusion_matrix(y, prediction)
        print "Confusion matrix:", Confusion_M


class MLPerceptron:
    def __init__(self, num_features, hidden_layer_sizes, num_classes, activation='relu', reg=0.0001, r_s=0):
        self.reg = reg
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.model = sklearn.neural_network.MLPClassifier(self.hidden_layer_sizes,
                                                        activation=self.activation,
                                                        alpha=self.reg, max_iter=1000, random_state=r_s)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        accu = self.model.score(x_train, y_train)
        print "Result train accuracy:", self.model.score(x_train, y_train)
        prediction = self.prediction(x_train)
        Confusion_M = sklearn.metrics.confusion_matrix(y_train, prediction)
        print "Confusion matrix:", Confusion_M
        #print accu

    def prediction(self, x):
        if len(x.shape)==1:
            x = np.reshape(x, (1, x.shape[0]))
        return self.model.predict(x)

    def cross_validation(self, X, y, k_fold=5):
        best_accuracy = 0.0
        N = X.shape[0]
        N_train = int(math.floor(0.8 * N))

        min_lamb = 0.000000001
        max_lamb = 2.0
        log_min_lamb = math.log(min_lamb)
        log_max_lamb = math.log(max_lamb)
        lamb_list = np.logspace(log_min_lamb, log_max_lamb, num=10, base=math.e)

        best_lamb = self.reg

        for lamb in lamb_list:
            print "lamb", lamb
            self.reg = lamb
            accuracy = np.zeros((k_fold))
            for i in range(k_fold):
                map_index = list(zip(X, y))
                random.shuffle(map_index)
                random_X, random_y = zip(*map_index)

                train_X = np.array(random_X[:N_train])
                valid_X = random_X[N_train:]
                train_y = np.array(random_y[:N_train])
                valid_y = random_y[N_train:]

                self.train(train_X, train_y)
                accuracy[i] = self.model.score(valid_X, valid_y)
                

            mean_accuracy = np.mean(accuracy)
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_lamb = self.reg
            print mean_accuracy
        self.reg = best_lamb
        self.train(X, y)
        print "best lamb", best_lamb
        print "best accuracy", best_accuracy
        print "Result train accuracy:", self.model.score(X, y)
        prediction = self.prediction(X)
        Confusion_M = sklearn.metrics.confusion_matrix(y, prediction)
        print "Confusion matrix:", Confusion_M

class LogisticRegression:
    def __init__(self, lamb_inv=1.0, reg_penalty='l2', k_fold=5, r_s=0):
        self.lamb_inv = lamb_inv
        self.reg_penalty = reg_penalty
        self.k_fold = k_fold
        self.model = sklearn.linear_model.LogisticRegression(penalty=self.reg_penalty, C=self.lamb_inv,
                                                             max_iter=1000, random_state=r_s)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        accuracy_train = self.model.score(x_train, y_train)
        print accuracy_train

    def prediction(self, x):
        if len(x.shape)==1:
            x = np.reshape(x, (1, x.shape[0]))
        predict = self.model.predict(x)

        return predict

    def cross_validation(self, X, y, k_fold=10):
        self.model = sklearn.linear_model.LogisticRegressionCV(penalty=self.reg_penalty,
                                                               cv=self.k_fold, max_iter=1000)
        self.model.fit(X, y)
        print "Result train accuracy:", self.model.score(X, y)
        prediction = self.prediction(X)
        Confusion_M = sklearn.metrics.confusion_matrix(y, prediction)
        print "Confusion matrix:", Confusion_M

class Bagging:
    def __init__(self, all_model=None, number_model=50, learning_rate=1.0):
        self.all_model = all_model
        self.number_model = number_model
        self.learning_rate = learning_rate
    
    def train(self, x_train, y_train):
        for i in range(self.number_model):
            curr_model = self.all_model[i].model
            curr_model.fit(x_train, y_train)
            self.all_model.append(curr_model)

    def prediction(self, x):
        all_predict = np.zeros((x.shape[0], self.number_model))
        for i in range(self.number_model):
            curr_predict = self.all_model[i].prediction(x)
            all_predict[:,i] = curr_predict
        all_predict = all_predict.astype(np.int)
        final_predict = np.argmax(np.apply_along_axis(np.bincount, axis=1, arr=all_predict, minlength=8),axis=1)
        return final_predict