# -*- coding: utf-8 -*-

# Execution in terminal :
#            python Classifier.py model_type validation
#            Example : python Classifier.py 1 1
#


import numpy as np
import sys
import DataManager as DM      ## remove comment when it's work##
import Model

def main():

    if len(sys.argv) < 3:
        error = "\n Error: The number of parameter is wrong\
                \n\n\t python Classifier.py model_type validation\
                \n\n\t model_type: 1 to 6 and 7 for all model\
                \n\t cross_validation: 0-1\n"
        print(error)
        return

    type_model = int(sys.argv[1])
    cross_val = bool(int(sys.argv[2]))

    # Load database
    print("Loading Database...")
    dm = DM.DataManager(100, 100, normalisation=False)
    x_train, t_train, x_test, t_test = dm.generer_donnees()

    md = None
    # Selection of the model
    if type_model == 1:
        print("TODO : Selecting model 1...")
    elif type_model == 2:
        print("TODO : Selecting model 2...")
    elif type_model == 3:
        md = Model.ModelDecisionTree()
    elif type_model == 4:
        md = Model.ModelSVM(kernel="poly", degree=4,  verbose=True)
    elif type_model == 5:
        print("TODO : Selecting model 5...")
    elif type_model == 6:
        print("TODO : Selecting model 6...")
    elif type_model == 7:
        print("TODO : Testing all models with same data.")
    else:
        print("Error : No model is train.")

    # Training of the model
    if cross_val is False:
        md.train(x_train, t_train)
    else:
        md.cross_validation(x_train, t_train)

    # Compute Train / Test
    # Prediction on the train dataset and the test dataset
    predictions_entrainement = np.array([md.prediction(x) for x in x_train])
    train_error = np.sum([md.error(x, t) for (x, t) in zip(x_train, t_train)])
    print("Train error = ", 100*train_error/len(t_train), "%")
    predictions_test = np.array([md.prediction(x) for x in x_test])
    test_error = np.sum([md.error(x, t) for (x, t) in zip(x_test, t_test)])
    print("Test error = ", 100*test_error/len(t_test), "%")


if __name__ == "__main__":
    main()
