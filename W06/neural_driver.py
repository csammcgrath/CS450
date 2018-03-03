"""
Author: Sam McGrath
Class: CS 450
Instructor: Br. Burton
"""

import numpy as np

class NeuralNetworkClassifier():
    def __init__(self):
        pass

    def fit(self, data, targets):
        numRows, numColumns = data.shape
        numTargets = np.unique(targets)
        numHiddens = None
        numNodes = None

        print("How many hidden layers would you like to have?")
        numHiddens = int(input("> "))

        print("How many nodes per hidden layer?")
        numNodes = int(input("> "))

        return NeuralNetworkModel(numColumns, numTargets, numHiddens, numNodes, data, targets)


class NeuralNetworkModel():
    def __init__(self, numColumns, numTargets, numHiddens, numNodes, data, targets):
        self.numColumns = numColumns
        self.numTargets = numTargets
        self.numHiddens = numHiddens
        self.numNodes = numNodes
        self.data = data
        self.targets = targets

        

        





