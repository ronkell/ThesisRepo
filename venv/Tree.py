from TreeNode import TreeNode
import numpy as np
class Tree():
    def __init__(self, initParams=[]):
        self.count = 0
        self.nodes = {}
        self.nodes[-1] = TreeNode(-1,id=self.count)
        self.currRoot=-1
        #self.tempnewroot=0
        self.history_of_roots=[]
        self.action_nodes_with_more_than_1_child=[]


    def isLeaf(self, h):
        if self.nodes[h].countN == 0:
            return True
        else:
            return False

    def ExpandTreeFrom(self, parent, action_obs, IsAction=False):
        if IsAction:
            # add node to the tree
            self.nodes[self.count] = TreeNode(parent, {}, 0, 0, -1,self.count)  # adding action node without belief state
            self.nodes[parent].childnodes[action_obs] = self.count
        else:
            self.nodes[self.count] = TreeNode(parent, {}, 0, 0, [],self.count)  # adding obs ndoe with beleif state
            self.nodes[parent].childnodes[action_obs] = self.count
        self.count += 1

    def UCB(self, nha, nh, vha, c=1):
        return vha + c * np.sqrt(np.log(nh) / nha)

    def childcount(self,node):
        return len(self.nodes[node].childnodes)

    def getObservationNode(self, h, sample_observation):
        if sample_observation not in list(self.nodes[h].childnodes.keys()):
            self.ExpandTreeFrom(h, sample_observation)
        next_node = self.nodes[h].childnodes[sample_observation]
        return next_node

    def prune(self, node,child_to_save):
        """
        recive node and delete all his childrens excepct one child
        :param node:
        :return:
        """
        childrens = self.nodes[node].childnodes
        # for key, child in childrens.items():
        #     if key is not child_to_save:
        #         self.pruneHelper(child)
        #         del childrens[key]
        # more efficent code:
        temp=list(childrens.keys())
        keys=temp.copy()
        for key in keys:
            if childrens[key] != child_to_save:
                self.pruneHelper(childrens[key])
                del childrens[key]



    def pruneHelper(self,node):
        """
        recive node and delete itself and all his childers.
        :param node:
        :return:
        """
        childrens = self.nodes[node].childnodes
        # if node == self.tempnewroot:
        #     print("here")
        del self.nodes[node]
        for _, child in childrens.items():
            self.pruneHelper(child)

    def make_new_root(self, new_root):
        self.currRoot=new_root
        self.history_of_roots.append(self.currRoot)
        action_father_node=self.nodes[new_root].parent
        if len(self.nodes[action_father_node].childnodes)>1:
            self.action_nodes_with_more_than_1_child.append(action_father_node)
    def getChildrens(self,node):
        return self.nodes[node].childnodes


    def prune_after_action(self, action, observation):
        action_node = self.nodes[self.currRoot].childnodes[action]

        # get new root after obs
        new_root = self.getObservationNode(action_node, observation)
        self.tempnewroot=new_root


        # remove from prev root the chosen child(action node) so i wont delete it.
        #del self.nodes[self.currRoot].childnodes[action_node] ## i dont know if del works using pop instead


        # prune the rest unnesecary  child nodes of the prev root, all but the chosen action node.
        self.prune(self.currRoot,action_node)


        # set the new root to root key=-1
        self.make_new_root(new_root)
        return self.currRoot
