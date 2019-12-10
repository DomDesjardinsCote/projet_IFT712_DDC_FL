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
    def __init__(self, reg_penalty='l2', reg=0.001, k_fold=5, random_state=0):
        """
        Initialize the parameters of the perceptron model.
        reg_penalty = Type of regularization to be used on the parameters
        reg = Constant that multiplies the regularization term if used
        k_fold = The number of folds used for the cross-validation
        random_state = The seed to use when shuffling the data.
                       Useful to generate multiple model with same data for bagging
        """
        print("Initialize model Perceptron")
        self.reg_penalty = reg_penalty
        self.reg = reg
        self.k_fold = k_fold
        self.random_state = random_state
        self.model = sklearn.linear_model.Perceptron(penalty=reg_penalty,
                                                     alpha=self.reg,
                                                     max_iter=1000,
                                                     random_state=self.random_state)

    def train(self, x, t):
        """
        Training the Perceptron model with the data
        x : A numpy array of the features dataset (n_samples, n_features)
        t : A numpy array of the class label (n_sample)
        """
        self.model.fit(x, t)

    def prediction(self, x):
        """
        Predict the classes label given the features data
        x : A numpy array of the features for one sample data (n_features) or
            a set (n_samples, n_features)
        return : The predict classes label from the model
        Comments : The method "train" must have been call previously
        """
        if len(x.shape)==1:
            x = np.reshape(x, (1, x.shape[0]))
        predict = self.model.predict(x)
        return predict

    def cross_validation(self, x, t):
        """
        Do a k_fold cross validation on the hyper parameters.
        The hyper-parameters are
        reg : 10^-3 to 10^3 with a log scale
        Parameters
        x : A numpy array of the features dataset (n_samples, n_features)
        t : A numpy array of the class label (n_sample)
        """
        # Initialize accuracy / hyperparameters
        best_accuracy = 0.0
        best_reg = 0.0

        # Cross-validation 80-20
        N = X.shape[0]
        N_train = int(math.floor(0.8 * N))

        # Initialize the grid seach hyperparameters
        min_reg = 0.001
        max_reg = 1000
        log_min_reg = np.log(min_reg)
        log_max_reg = np.log(max_reg)
        reg_list = np.logspace(log_min_reg, log_max_reg, num=7, base=math.e)

        for reg in reg_list:
            accuracy = np.zeros((self.k_fold))
            for i in range(self.k_fold):
                map_index = list(zip(x, t))
                random.shuffle(map_index)
                random_x, random_t = zip(*map_index)

                train_x = random_x[:N_train]
                valid_x = random_x[N_train:]
                train_t = random_t[:N_train]
                valid_t = random_t[N_train:]

                self.model = sklearn.linear_model.Perceptron(penalty=self.reg_penalty,
                                                             alpha=reg,
                                                             max_iter=1000,
                                                             random_state=self.random_state)
                self.train(train_x, train_t)
                accuracy[i] = self.model.score(valid_x, valid_t)

            mean_accuracy = np.mean(accuracy)
            # print(mean_accuracy)
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_reg = reg
                print("The new best hyperparameters are : ", best_reg)

        print("Best hyperparameters are : ", best_reg)
        print("Valid Accuracy :", best_accuracy)
        self.reg = best_reg
        self.model = sklearn.linear_model.Perceptron(penalty=self.reg_penalty,
                                                     alpha=best_reg,
                                                     max_iter=1000,
                                                     random_state=self.random_state)
        self.train(x, t)


class MLPerceptron:
    def __init__(self, hidden_layer_sizes, activation='relu', reg=0.001, k_fold=5, random_state=0):
        """
        Initialize the parameters of the multi-layer Perceptron.
        hidden_layer_sizes = The ith element represents the number of neurons in the ith hidden layer
        activation = Activation function for the hidden layer
        reg = L2 penalty parameter
        k_fold = The number of folds used for the cross-validation
        random_state = The seed to use when shuffling the data.
                       Useful to generate multiple model with same data for bagging
        """
        print("Initialize model Multi-layer Perceptron")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.reg = reg
        self.k_fold = k_fold
        self.random_state = random_state
        self.model = sklearn.neural_network.MLPClassifier(self.hidden_layer_sizes,
                                                        activation=self.activation,
                                                        alpha=self.reg, max_iter=1000, 
                                                        random_state=self.random_state)

    def train(self, x, t):
        """
        Training the Multi-layer Perceptron model with the data
        x : A numpy array of the features dataset (n_samples, n_features)
        t : A numpy array of the class label (n_sample)
        """
        self.model.fit(x, t)

    def prediction(self, x):
        """
        Predict the classes label given the features data
        x : A numpy array of the features for one sample data (n_features) or
            a set (n_samples, n_features)
        return : The predict classes label from the model
        Comments : The method "train" must have been call previously
        """
        if len(x.shape)==1:
            x = np.reshape(x, (1, x.shape[0]))
        return self.model.predict(x)

    def cross_validation(self, x, t):
        """
        Do a k_fold cross validation on the hyper parameters.
        The hyper-parameters are
        reg : 10^-3 to 10^3 with a log scale
        Parameters
        x : A numpy array of the features dataset (n_samples, n_features)
        t : A numpy array of the class label (n_sample)
        """
        # Initialize accuracy / hyperparameters
        best_accuracy = 0.0
        best_reg = 0.0

        # Cross-validation 80-20
        N = X.shape[0]
        N_train = int(math.floor(0.8 * N))

        # Initialize the grid search hyperparameters
        min_reg = 0.001
        max_reg = 1000
        log_min_reg = np.log(min_reg)
        log_max_reg = np.log(max_reg)
        reg_list = np.logspace(log_min_reg, log_max_reg, num=7, base=math.e)

        for reg in reg_list:
            accuracy = np.zeros((self.k_fold))
            for i in range(self.k_fold):
                map_index = list(zip(x, t))
                random.shuffle(map_index)
                random_x, random_t = zip(*map_index)

                train_x = random_x[:N_train]
                valid_x = random_x[N_train:]
                train_t = random_t[:N_train]
                valid_t = random_t[N_train:]

                self.model = sklearn.neural_network.MLPClassifier(self.hidden_layer_sizes,
                                                        activation=self.activation,
                                                        alpha=reg, max_iter=1000, 
                                                        random_state=self.random_state)
                self.train(train_x, train_t)
                accuracy[i] = self.model.score(valid_x, valid_t)

            mean_accuracy = np.mean(accuracy)
            # print(mean_accuracy)
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_reg = reg
                print("The new best hyperparameters are : ", best_reg)

        print("Best hyperparameters are : ", best_reg)
        print("Valid Accuracy :", best_accuracy)
        self.reg = best_lamb
        self.model = sklearn.neural_network.MLPClassifier(self.hidden_layer_sizes,
                                                          activation=self.activation,
                                                          alpha=best_reg, max_iter=1000, 
                                                          random_state=self.random_state)
        self.train(x, t)


class LogisticRegression:
    def __init__(self, reg_penalty='l2', reg_inv=1.0, k_fold=5, random_state=0):
        """
        Initialize the parameters of the Logistic Regression for Classification.
        reg_penalty = Type of regularization to be used on the parameters
        reg_inv = Inverse of regularization strength
        random_state = The seed to use when shuffling the data.
                       Useful to generate multiple model with same data for bagging
        k_fold = The number of folds used for the cross-validation
        """
        print("Initialize model Logistic Regression")
        self.reg_penalty = reg_penalty
        self.reg_inv = reg_inv
        self.k_fold = k_fold
        self.random_state = random_state
        self.model = sklearn.linear_model.LogisticRegression(penalty=self.reg_penalty,
                                                             C=self.reg_inv,
                                                             max_iter=1000, 
                                                             random_state=self.random_state)

    def train(self, x, t):
        """
        Training the Logistic Regression classification model with the data
        x : A numpy array of the features dataset (n_samples, n_features)
        t : A numpy array of the class label (n_sample)
        """
        self.model.fit(x, t)

    def prediction(self, x):
        """
        Predict the classes label given the features data
        x : A numpy array of the features for one sample data (n_features) or
            a set (n_samples, n_features)
        return : The predict classes label from the model
        Comments : The method "train" must have been call previously
        """
        if len(x.shape)==1:
            x = np.reshape(x, (1, x.shape[0]))
        predict = self.model.predict(x)

        return predict

    def cross_validation(self, x, t):
        # Here, we can use the built in cross validation for logistic Regression
        self.model = sklearn.linear_model.LogisticRegressionCV(penalty=self.reg_penalty,
                                                               cv=self.k_fold, max_iter=1000)
        self.model.fit(x, t)


class Bagging:
    def __init__(self, base_model='LogisticRegression', number_model=50, 
                 hidden_layer_sizes=(100,), activation='relu',
                 kernel='poly', degree=3, coef0=0, gamma='auto',
                 criterion='gini', reg_penalty='l2', reg=0.001):
        """
        Initialise the parameters of a Bagging algorithm
        base_model = Base model on which we want to train multiple time
        ('Perceptron', 'MLPerceptron', 'ModelSVM', 'ModelDecisionTree', 'LogisticRegression')
        number_model = Number of base_model train to generate the Bagging model
        hidden_layer_sizes = The ith element represents the number of neurons in the ith hidden layer
                             Used if base_model is 'MLPerceptron'
        activation =  Activation function for the hidden layer
                      Used if base_model is 'MLPerceptron'
        kernel = Name of a kernel (linear, poly, rbf, sigmoid)
                 Used if base_model is 'ModelSVM'
        degree = Polynomial degree
                 Used if base_model is 'ModelSVM'
        coef0 = coef0 of the linear kernel
                Used if base_model is 'ModelSVM'
        gamma = Hyperparameters for kernel (poly, rbf, sigmoid)
                Used if base_model is 'ModelSVM'
        criterion = The function to measure the quality of a split
                    Used if base_model is 'ModelDecisionTree'
        reg_penalty = Type of regularization to be used on the parameters
                      Used if base_model is 'Perceptron' or 'LogisticRegression'
        reg = Constant that multiplies the regularization term if used
        """
        self.number_model = number_model

        # Initialise all_model list
        self.all_model = []
        for i in range(number_model):
            if base_model=='Perceptron':
                curr_model = Perceptron(reg_penalty=reg_penalty, reg=reg,
                                        random_state=i)
                self.all_model.append(curr_model.model)
            elif base_model=='MLPerceptron':
                curr_model = MLPerceptron(hidden_layer_sizes=hidden_layer_sizes,
                                                activation=activation, reg=reg, random_state=i)
                self.all_model.append(curr_model.model)
            elif base_model=='LogisticRegression':
                curr_model = LogisticRegression(reg_penalty=reg_penalty,
                                                      reg_inv=reg, random_state=i)
                self.all_model.append(curr_model.model)
            elif base_model=='ModelSVM':
                curr_model = ModelSVM(kernel=kernel, degree=degree, coef0=coef0,
                                            gamma=gamma, reg=reg, random_state=i)
                self.all_model.append(curr_model.model)
            elif base_model=='ModelDecisionTree':
                curr_model = ModelDecisionTree(criterion=criterion, random_state=i)
                self.all_model.append(curr_model.model)

    def train(self, x, t):
        """
        Training all model with the data
        x : A numpy array of the features dataset (n_samples, n_features)
        t : A numpy array of the class label (n_sample)
        """
        for i in range(self.number_model):
            curr_model = self.all_model[i]
            curr_model.fit(x, t)

    def prediction(self, x):
        """
        Predict the classes label given the features data
        x : A numpy array of the features for one sample data (n_features) or
            a set (n_samples, n_features)
        return : The predict classes label the most popular across all model
        Comments : The method "train" must have been call previously
        """
        if len(x.shape)==1:
            x = np.reshape(x, (1, x.shape[0]))

        all_predict = np.zeros((x.shape[0], self.number_model))
        for i in range(self.number_model):
            curr_predict = self.all_model[i].predict(x)
            all_predict[:,i] = curr_predict
        all_predict = all_predict.astype(np.int)
        final_predict = np.argmax(np.apply_along_axis(np.bincount, axis=1, arr=all_predict, minlength=8),axis=1)
        return final_predict