import numpy as np
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve

class ModelAnalyzer:
    def __init__(self):
        pass

    def plotLearningCurves(self, estimator, X, t, title=None, cv=5, train_sizes=np.linspace(.1, 1.0, 10)):
        """
            Plot the learning curves of a model.
            :param estimator: The model
            :param X: A numpy array of the dataset
            :param t: A numpy array of the classes from the dataset
            :param title: Title of the plot
            :param cv: Number of validation per train
            :param train_sizes: Number of training to do
        """
        plot.figure()
        plot.title(title)
        plot.xlabel("Training examples")
        plot.ylabel("Score")

        train_sizes, train_scores, test_scores = learning_curve(estimator, X, t, cv=cv, train_sizes=train_sizes)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plot.grid()

        plot.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha =0.1, color="r")

        plot.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plot.plot(train_sizes, train_scores_mean, '-o', color="r", label="Training Score")

        plot.plot(train_sizes, test_scores_mean, '-o', color="g", label="Validation Score")
        plot.legend(loc="best")

        return plot

    def confusionMatrix(self, t, pred):
        """
        Compute the confusion matrix.
        :param t: A numpy array of the classes from the dataset
        :param pred: A numpy array of the predicted classes from a model.
        :return: (class, class) numpy array a confusion matrix.
        """
        return confusion_matrix(y_true=t, y_pred=pred)

    def accuracy(self, t, pred):
        """
        Compute the accuracy of the model
        :param t: A numpy array of the classes from the dataset
        :param pred: A numpy array of the predicted classes from a model.
        :return: A float the accuracy.
        """
        confMatrix = confusion_matrix(y_true=t, y_pred=pred)
        tp = np.trace(confMatrix)

        accu = tp / len(t)
        return accu



