"""
Author: Sam McGrath
Class: CS 450
Instructor: Br. Burton
"""

import numpy as np
import random
import math

# a node in a layer
class NeuronNode():
    def __init__(self, numInputs, target=None):
        self.inputWeights = []
        self.bias = -1
        self.target = target

        #0.5 for now but it will be changed soon
        self.threshold = 0.5

        #create a random set of weights of +- 0.1
        for index in range(numInputs + 1):
            self.inputWeights.append(random.uniform(-0.1, 0.1))

    #utilized for activation -- this is to be present for every node
    #at every layer
    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))

    def fire(self, row, target):
        value = 0

        for index in range(len(row)):
            value += (row[index] * self.inputWeights[index + 1])

        #ensuring that bias is present at every layer
        value += (self.bias * self.inputWeights[0])

        value = self.sigmoid(value)

        #if value is above threshold, fire!
        if (value > self.threshold):
            return True
        else:
            return False

#a layer of nodes
class Neurons():
    def __init__(self, cols, targets, numHiddenLayers):
        self.neuronLayer = []
        self.targets = targets
        self.uTargets = np.unique(targets)

        #create hidden layer(s)
        for hidden in numHiddenLayers:
            self.neuronLayer.append(NeuronNode(cols))

        #create output layer
        for target in self.uTargets:
            self.neuronLayer.append(NeuronNode(cols, target))

    def train(self, data):
        flag = False
        epochs = 0

        #keep going until no weights are changed or
        #we hit 10,000 epochs
        while not flag and epochs < 10000:
            flag = True
            epochs += 1

            #determine whether we should change the weights
            #might be used for weight changing..
            for index, row in enumerate(data):
                for node in self.neuronLayer:
                    if not (node.fire(row, self.targets[index])):
                        flag = False

#a combination of node layers to form a neural network.
class NeuralNetwork():
    def __init__(self):
        pass

    def fit(self, data, targets, numOfHiddenLayers=2):
        #retrieve number of rows and columns of dataset
        rows, cols = data.shape()

        #create a layer of neurons
        nn = Neurons(cols, targets, numOfHiddenLayers)

        nn.train(data)

        return NeuralNetworkDriver(targets)

class NeuralNetworkDriver():
    def __init__(self, targets):
        self.model = []
        self.targets = targets

    def predict(self, data):
        mostCommon = self.targets[self.getModeBins()]

        for row in data:
            self.model.append(mostCommon)

        return self.model

    def getModeBins(self):
        unique, positions = np.unique(self.targets, return_inverse=True)

        bins = np.bincount(positions)

        return max(bins, key=bins.count)