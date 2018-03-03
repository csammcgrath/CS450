"""
Author: Sam McGrath
Class: CS 450
Instructor: Br. Burton
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plot

class Neuron():
    def __init__(self, numFeatures):
        #create weights
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(0, numFeatures + 1)]

        #each node will have its own error
        #this allows for easier error tracking 
        self.error = 0

    #determine activation value
    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))

    #calculate h to help determine activation value
    def calculateH(self, features):
        #ensuring that the bias is present in every layer
        featuresArray = np.append(features, [-1])

        total = 0

        #going through and getting the activation value
        for index, feature in enumerate(featuresArray):
            total += (self.weights[index] * feature)

        #ensure that the activation value is squashed
        totalSigmoid = self.sigmoid(total)

        return totalSigmoid

class NeuralNetwork():
    def __init__(self):
        self.learningRate = 0.1 #n in the error calculations
        self.layers = []        #array of nodes
        
        self.numEpochs = -1     #get num of epochs 
        self.numHidden = -1     #get num of layers
        self.features = None
        self.printEpoch = None  #boolean value whether to print the epochs
        self.accuracy = []      #store accuracy

        self.data = None
        self.targets = None

    def fit(self, data, targets, classes):
        self.data = data
        self.targets = targets
        self.classes = classes
        self.features = self.data.shape[1]

        #determine how many hidden layers/nodes 
        self.obtainInformation()

        #building layers
        for index in range(0, self.numHidden):
            self.layers.append(self.createLayer(self.numInputs(index), \
                               self.retrieveNumNodes(index)))

        self.train()

        return self

    def numInputs(self, layerNum):
        if (layerNum > 0):
            return len(self.layers[layerNum - 1]) + 1
        else:
            return self.features

    def plotGraph(self, accuracy):
        print("Accuracy", accuracy)
        plot.plot(accuracy)
        plot.title("Training Accuracy - MLP")
        plot.xlabel("# of epochs")
        plot.ylabel("Accuracy")
        plot.legend([accuracy], 'Iris')
        plot.show()

    #predict
    #
    # It calls predictClass and appends the results of the
    # function call to the predictions array
    def predict(self, data):
        model = []

        for (item) in data:
            model.append(self.predictClass(item))

        return model

    #predictClass
    #
    #gets the results and return the argmax 
    #of that results array
    def predictClass(self, item):
        results = self.getResults(item)

        #return max index
        return np.argmax(results[-1])

    #createLayer
    #
    #This function goes through and creates a
    #default number of nodes and returns the array
    #consisting of all of the nodes
    def createLayer(self, numFeatures, numNodes):
        nodeArray = []

        #build a layer of nodes
        for _ in range(0, numNodes):
            nodeArray.append(Neuron(numFeatures))

        return nodeArray

    #obtainInformation
    #
    #This function obtain the numbers of hidden layers and
    #epochs desired. The reason for number of epochs desired
    #is so I can experiment with the number of epochs and how
    #it affects the training of the neural network.
    def obtainInformation(self):
        try:
            print("How many hidden layers? (1, 2)")
            self.numHidden = int(input("> "))

            #it has been proven that if there are more than
            #2 hidden layers in a multi-layer perceptron 
            if (self.numHidden < 1 or self.numHidden > 2):
                raise Exception("please select either 1 or 2 hidden layers.")

            #I really don't want the neural network to overtrain
            #so I am putting a hard cap on the number of epochs
            print("How many epochs? (1 - 10,000)")
            self.numEpochs = int(input("> "))

            if (self.numEpochs < 1 or self.numEpochs > 10000):
                raise Exception("please choose between 1 and 10,000")
        except (ValueError) as err:
            print("ERROR: {}".format(err))

    # retrieveNumNodes
    #
    # Prompts the user for the number of nodes
    # for layer x
    def retrieveNumNodes(self, layerNum):
        numNode = -1

        try:
            print("How many nodes for Layer {}?".format(layerNum + 1))
            numNode = int(input("> "))

            if (numNode < 1):
                raise Exception("Please choose more than one node next time.")

        except (ValueError) as err:
            print("ERROR: {}".format(err))

        return numNode

    def train(self):
        accuracy = []

        for epoch in range(0, self.numEpochs):
            count = 0
            predictions = []

            for (dataPoint, target) in zip(self.data, self.targets):
                results = self.getResults(dataPoint)
                something = np.argmax(results[-1])
                predictions.append(something)
                
                # if (target[0] != something):
                self.updateNode(dataPoint, target, results)
            
            for (index, prediction) in enumerate(predictions):
                if (self.targets[index] == prediction):
                    count += 1

            #help keep track of accuracy levels for each epoch
            accuracy.append(getAccuracy(count, len(self.targets)))
                
    def updateNode(self, data, target, results):
        #we must update errors then weights
        self.updateErrors(target, results)
        self.updateWeights(data, results)

    def updateWeights(self, inputs, results):
        for (index, layer) in enumerate(self.layers):
            for (node) in layer:
                if (index > 0):
                    #here, we know that there is a previous layer that is
                    #NOT the input layer
                    self.updateNodeWeights(node, results[index - 1])
                else:
                    #here, we know that this is the second layer in the neural
                    #network so we can just use the input layer
                    self.updateNodeWeights(node, inputs.tolist())

    def updateNodeWeights(self, node, inputs):
        tempInputArray = np.append(inputs, [-1])

        newWeights = []

        #build array of new weights for the node itself
        for index, weight in enumerate(node.weights):
            newWeights.append(weight - (0.1) * (tempInputArray[index]) * node.error)

        #update the node's weights
        node.weights = newWeights

    def updateErrors(self, target, results):
        for (indexLayer, layer) in reversed(list(enumerate(self.layers))):
            for (indexNode, node) in enumerate(layer):
                #update the node error and let the getErrorNode take care of 
                #everything we need
                node.error = self.getErrorNode(indexNode, indexLayer, target, results)
                
    def getErrorNode(self, indexNode, indexLayer, target, results):
        #help determine if it is a hidden layer or output layer
        #since the error is calculated differently for each type of
        #layer
        # print(len(results))
        if (indexLayer < (len(results) - 1)):
            return self.getHiddenNodeError(results[indexLayer][indexNode], \
                                    self.getWeightsNode(indexNode, indexLayer), \
                                    self.getErrorsNode(indexNode, indexLayer))
        else:
            #since it is an output layer, we can give a target but
            #the formula requires it be a 1 or a 0 so we simply just
            #create a boolean through target == indexNode
            return self.getOutputNodeError(results[indexLayer][indexNode], \
                                    target == indexNode)

    def getWeightsNode(self, indexNode, indexLayer):
        returnWeightsArray = []

        for (node) in self.layers[indexLayer + 1]:
            returnWeightsArray.append(node.weights[indexNode])

        # print(returnWeightsArray)
        return returnWeightsArray

    def getErrorsNode(self, indexNode, indexLayer):
        returnErrorArray = []

        for (node) in self.layers[indexLayer + 1]:
            returnErrorArray.append(node.error[indexNode])

        return returnErrorArray
        
    def getOutputNodeError(self, result, target):
        return result * (1 - result) * (result - target)

    def getHiddenNodeError(self, result, weights, errors):
        total = 0

        for (index, weight) in enumerate(weights):
            total += (errors[index] * weight)

        return result * (1 - result) * total

    #getResults
    #
    #This function goes through and calculates
    #the h for each node in each layer and
    #returns the results.
    def getResults(self, input):
        results = []

        for (index, layer) in enumerate(self.layers):
            singleResults = []

            for (node) in layer:
                if (index > 0):
                    singleResults.append(node.calculateH(results[index - 1]))
                else:
                    singleResults.append(input)
            
            results.append(singleResults)

        return results

# getAccuracy()
#
# This function calculates and returns the accuracy
def getAccuracy(count, length):
    return (count / length) * 100