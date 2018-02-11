import numpy as np
import pandas as pd
import math
import decision_tree as watdt
from node import Node
import operator

class DecisionTree():
    #nothing realy needs to be initiated here
    def __init__(self):
        pass

    #build the tee
    def fit(self, data, targets, headers):
        return SamDecisionTree(data, targets, headers)   

class SamDecisionTree():
    def __init__(self, data, targets, headers):
        self.data = data
        self.targets = targets
        self.headers = headers
        frames = [data, targets]
        final_data = pd.concat(frames, axis=1)
        
        self.tree = watdt.build_tree(final_data, headers[:-1])
    
    def __repr__(self):
        watdt.print_tree(self.tree)

    #go through each item and begin traversal
    #tree traversal actually happens here
    def predict(self, data):
        predictions = []
        
        for item, row in data.iterrows():
            attr = self.headers[:]
            predictions.append(self.predict_class(self.tree, row, attr))

        return predictions

    def predict_class(self, node, row, headers):
        for attribute in headers:
            if attribute == node.name:
                value = row[attribute]
                if value in node.children:
                    new_node = node.children[value]
                    if new_node.isLeaf():
                        return new_node.name
                    else:
                        headers.remove(attribute)
                        return self.predict_class(new_node, row, headers)
                else:
                    return 0
        return 1

    def print_tree(self, node):
        print(node.name),

        for (key, value) in node.children.items():
            print(key, "{"),
            self.print_tree(value),
            print("}"),
