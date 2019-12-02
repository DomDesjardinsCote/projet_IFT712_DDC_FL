# -*- coding: utf-8 -*-

# Execution in terminal :
#            python Classifier.py model_type validation
#            Example : python Classifier.py 1 1
#


import numpy as np
import sys
import DataManager as DM      ## remove comment when it's work##
import Model
import Utils
from sklearn.svm import SVC
import matplotlib.pyplot as plot
from DataManager import DataManager

def main():

    if len(sys.argv) < 5:
        error = "\n Error: The number of parameter is wrong\
                \n\n\t python Classifier.py model_type validation\
                \n\n\t model_type: 1 to 6 and 7 for all model\
                \n\t cross_validation: 0-1\
                \n\t Evaluation of the model : 0-1\
                \n\t Performance of the model(Learning Curve) : 0-1"
        print(error)
        return

    type_model = int(sys.argv[1])
    cross_val = bool(int(sys.argv[2]))
    eval_model = bool(int(sys.argv[3]))
    perf_model = bool(int(sys.argv[4]))

    # Load database
    my_data = DataManager(1000, 1000)
    x_train, y_train, x_test, y_test = my_data.generer_donnees()

    md = None
    title = None
    # Selection of the model
    if type_model == 1:
        print("TODO : Selecting model 1...")
        title = "Model 1"
    elif type_model == 2:
        title = "Model 2"
        md = Model.Perceptron(55,7)
    elif type_model == 3:
        md = Model.ModelDecisionTree()
        title = "Model 3"
    elif type_model == 4:
        md = Model.ModelSVM(kernel="poly", degree=2,  verbose=True)
        title = "Model 4"
    elif type_model == 5:
        print("TODO : Selecting model 5...")
        title = "Model 5"
    elif type_model == 6:
        print("TODO : Selecting model 6...")
        title = "Model 6"
    elif type_model == 7:
        print("TODO : Testing all models with same data.")
        title = "Model 7"
    else:
        print("Error : No model is train.")

    # Training of the model
    if cross_val is False:
        md.train(x_train, y_train)
    else:
        md.cross_validation(x_train, y_train)

    # Compute Train / Test
    # Prediction on the train dataset and the test dataset
    predictions_entrainement = np.array([md.prediction(x) for x in x_train])
    train_error = np.sum([md.error(x, t) for (x, t) in zip(x_train, y_train)])
    print("Train error = ", 100*train_error/len(y_train), "%")
    predictions_test = np.array([md.prediction(x) for x in x_test])
    test_error = np.sum([md.error(x, t) for (x, t) in zip(x_test, ty_test)])
    print("Test error = ", 100*test_error/len(y_test), "%")

    analyzer = Utils.ModelAnalyzer()

    if eval_model is True:
        print(analyzer.confusionMatrix(t_test, predictions_test))
        print("The accuracy is : ", analyzer.accuracy(y_test, predictions_test))

    x = np.append(x_train, x_test, axis=0)
    t = np.append(y_train, y_test)

    if perf_model is True:      # Change parameter for the exemple
        plt = analyzer.plotLearningCurves(md.mod, x, t, title=title)
        plt.show()

    # Exemple of a validation curves
    # print("...---BEGIN TEST---...")
    # print(np.linspace(1, 2, 2))
    # model1 = Model.ModelSVM(gamma="auto",  kernel="poly", degree=2)
    # plt = analyzer.plotValidationCurve(model1.mod, x, t, title="Test : validCurve", param_name="degree",
    #                                  param_range=np.linspace(1, 2, 2), verbose=True, scaling="lin")
    # plt.show()
    # print("...---END TEST---...")


if __name__ == "__main__":
    main()
