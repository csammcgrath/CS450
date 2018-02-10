import numpy as np
import math
import decision_tree
from node import Node
import operator

class DecisionTree():
    #nothing realy needs to be initiated here
    def __init__(self):
        pass

    #build the tee
    def fit(self, data, targets):
        return SamDecisionTree(data, targets)   

class SamDecisionTree():
    def __init__(self, data, targets):
      self.data = data
      self.targets = targets
      self.tree = decision_tree.build_tree(self.data, self.targets)

    #go through each item and begin traversal
    #tree traversal actually happens here
    def predict(self, data):
        predictions = []
      
        for item in data:
            predictions.append(self.predict(item))

        return predictions