# -*- coding: utf-8 -*-

import numpy as np
import sys
#import DataManager as DM      ## remove comment when it's work##
import Model

def main():

    if len(sys.argv) < 3:
        error = "\n Error: The number of parameter is wrong\
                \n\n\t python Classifier.py model_type validation\
                \n\n\t model_type: 1 to 6\
                \n\t cross_validation: 0-1\n"
        print(error)
        return

    type_model = int(sys.argv[1])
    cross_val = bool(int(sys.argv[2]))

    # Load database
    print("TODO : Loading Database...")

    # Selection of the model
    if type_model == 1:
        print("TODO : Selecting model 1...")
    elif type_model == 2:
        print("TODO : Selecting model 2...")
    elif type_model == 3:
        print("TODO : Selecting model 3...")
    elif type_model == 4:
        print("TODO : Selecting model 4...")
    elif type_model == 5:
        print("TODO : Selecting model 5...")
    elif type_model == 6:
        print("TODO : Selecting model 6...")

    # Training of the model
    if cross_val is False:
        print("TODO : Training without cross-validation... ")
    else:
        print("TODO : Training with cross-validation")

    # Compute Train / Test
    print("TODO : Train error = ...")
    print("TODO : Test error = ...")


if __name__ == "__main__":
    main()