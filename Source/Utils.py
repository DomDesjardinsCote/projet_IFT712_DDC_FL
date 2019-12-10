import numpy as np
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score

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

        plot.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                          alpha =0.1, color="r")

        plot.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                          alpha=0.1, color="g")

        plot.plot(train_sizes, train_scores_mean, '-o', color="r", label="Training Score")

        plot.plot(train_sizes, test_scores_mean, '-o', color="g", label="Validation Score")
        plot.legend(loc="best")

        return plot

    def plotValidationCurve(self, estimator, x, t, title=None, param_name="gamma", param_range=np.logspace(-3, 3, 5),
                            cv=5, scaling="log", verbose=False):
        """
        Plot the validation curve.
            :param estimator: The model
            :param x: A numpy array of the dataset
            :param t: A numpy array of the classes from the dataset
            :param title: Title of the plot
            :param param_name: The name of the parameter
            :param param_range: Numpy array the list of value to cross validate
            :param cv: Cross-validation.
            :param scaling: The scaling of the x axis ("log" or "lin")
            :param verbose: Print the values of train scores means / std and test scores means / std.
        """
        plot.figure()
        plot.title(title)
        plot.xlabel(param_name)
        plot.ylabel("Score")

        train_scores, test_scores = validation_curve(estimator, X=x, y=t, param_name=param_name, param_range=param_range
                                                     , cv=cv, scoring="accuracy")

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        if verbose is True:
            print("The param is : ", param_name)
            print("The param range is : ", param_range)
            print("The train scores means are : ", train_scores_mean)
            print("The train scores std are : ", train_scores_std)
            print("The test scores means are : ", test_scores_mean)
            print("The test scores std are : ", test_scores_std)

        lw = 2

        if scaling == "log":
            plot.semilogx(param_range, train_scores_mean, label="Training score", color="orange", lw=lw)
        elif scaling == "lin":
            plot.plot(param_range, train_scores_mean, label="Training score", color="orange", lw=lw)

        plot.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                          alpha=0.15, color="orange", lw=lw)

        if scaling == "log":
            plot.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="blue", lw=lw)
        elif scaling == "lin":
            plot.plot(param_range, test_scores_mean, label="Cross-validation score", color="blue", lw=lw)

        plot.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                          alpha=0.15, color="blue")

        plot.legend(loc="best")
        return plot, test_scores_mean

    def confusionMatrix(self, t, pred):
        """
        Compute the confusion matrix.
        :param t: A numpy array of the classes from the dataset
        :param pred: A numpy array of the predicted classes from a model.
        :return: (class, class) numpy array a confusion matrix.
        """
        return confusion_matrix(y_true=t, y_pred=pred)

    def accuracy(self, t, pred, normalize=True):
        """
        Compute the accuracy of the model
        :param t: A numpy array of the classes from the dataset
        :param pred: A numpy array of the predicted classes from a model.
        :param normalize: Normalize the score.
        :return: A float the accuracy.
        """

        accu = accuracy_score(y_true=t, y_pred=pred, normalize=normalize)
        return accu

