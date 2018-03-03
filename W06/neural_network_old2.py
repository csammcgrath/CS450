"""
Author: Sam McGrath
Class: CS 450
Instructor: Br. Burton
"""

import numpy as np
import random
import math

class Node():
    def __init__(self, numNextNodes):
        self.numNextNodes = numNextNodes
        self.weights = []

        for _ in range(0, numNextNodes):
            self.weights.append(random.uniform(-1, 1))

    def output(self, nodeInput):
        outputs = []
        
        for weight in self.weights:
            outputs.append(weight * nodeInput)
        
        return outputs

    #will be for next week
    def changeWeight(self):
        pass

class Layer():
    def __init__(self, numNodes, numNextNodes):
        self.nodes = []
        self.numNextNodes = numNextNodes

        for _ in range(0, numNodes):
            self.nodes.append(Node(numNextNodes))

    def activation(self, row):
        row.append(-1)
        activation = []

        for node in range(len(row)):
            output = 0
            for index in range(0, self.numNextNodes):
                output += self.nodes[node].output(row[node])

    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))

class Network():
    def __init__(self, numInputs, numHiddens, numNodes, numTargets, data, targets):
        self.layers = []
        self.hiddenLayers = []

        inputLayer = np.zeros(numInputs)
        self.layers.append(inputLayer)
        hiddenLayer = Layer(inputLayer.outputs, numHiddens)
        previousLayer = hiddenLayer

        for _ in range(numHiddens):
            nextLayer = Layer(previousLayer.outputs, numHiddens)
            self.layers.append(previousLayer)
            previousLayer = nextLayer

        outputLayer = Layer(self.layers[-1].outputs, numTargets)       
        self.layers.append(outputLayer)

    def train(self, row):
        for layer in self.layers:
            layer.nodeInputs

    #adjust weights based on the weights
    def adjustWeights(self):
        pass