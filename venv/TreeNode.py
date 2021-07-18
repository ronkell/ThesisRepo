import numpy as np
class TreeNode():

    def __init__(self, parent, childnodes={}, Nc=0, value=0, belief=[],id=-500):
        self.parent = parent
        self.childnodes = childnodes
        self.countN = Nc
        self.value = value
        self.belief = belief
        self.id=id


    def printNode(self):
        print("parent ", self.parent)
        print("childnodes ", self.childnodes)
        print("countN ", self.countN)
        print("value ", self.value)
        print("belief ", self.belief)

    def copy(self):
        return TreeNode(self.parent, self.childnodes.copy(), self.countN, self.value, self.belief.copy())
