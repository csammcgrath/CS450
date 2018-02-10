class Node():
    def __init__(self, name = "", value = -1, children = {}):
        self.name = name
        self.value = value
        self.children = children

    def isLeaf(self):
        return self.children == type(dict)

    def appendChild(self, attribute, value):
        self.children[attribute] = value