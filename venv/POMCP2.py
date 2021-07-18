from Tree import Tree
from numpy.random import binomial, choice, multinomial
import numpy as np
class POMCP():
    def __init__(self, blackbox,problem, discountfactor=0.99, c=1, threshold=0.005, timeout=1000, num_of_patricles=1000):
        self.discountfactor = discountfactor
        self.blackbox = blackbox
        self.epsilon = threshold
        self.timeout = timeout
        self.num_of_patricles = num_of_patricles
        self.tree = Tree()
        self.c = c
        self.ends=[]
        self.problem=problem
        self.horizon=9
        #self.rollflag=False
        #self.saveroll=[[-1]]
        #self.rollcounter=0

    def initialize(self, I, A, O):
        """
        adding convertor from (index)->action/obs or (action/obs)->index
        :param I:
        :param A:
        :param O:
        :return:
        """
        self.initStates = I
        self.actions = A
        self.observations = O

    def convertToIndex(self,isAction,action_obs):
        if isAction:
            x = np.array(None); x[()] = action_obs
            index=np.where(self.actions==x)
            index2=index[0][0]
        else:
            x = np.array(None); x[()] = action_obs
            index = np.where(self.observations == x)
            index2 = index[0][0]
        return index2

    def convertFromIndex(self,isAction,action_obs_index):
        if isAction:
            return self.actions[action_obs_index]
        return self.observations[action_obs_index]

    def search(self, h=-1):
        prevBeliefState = self.tree.nodes[h].belief.copy()
        for i in range(self.timeout):
            if prevBeliefState == []:
                s = choice(self.initStates)
            else:
                s = choice(prevBeliefState)
            self.simulate(s, h, 0)

        action, child = self.searchBest(h, UseUCB=False)
        # maybe need to return also the best observation instaed of picking random from main.
        return action

    def rollout(self, s, h, depth):
        if (self.discountfactor ** depth < self.epsilon or self.discountfactor == 0 or depth>=self.horizon) and depth != 0:
            return 0
        accumlate_reward = 0
        valid_actions=self.problem.validactionsforrollout(s)
        #action = choice(self.actions)  # how to define a phirollout and do i need to add action and obs to the tree?
        action=choice(np.array(valid_actions))
        samplestate, sampleobs, reward = self.blackbox(s,action)
        if samplestate==-2:
            #self.rollflag=True
            #self.ends.append(h)
            return 0
        accumlate_reward += reward + self.discountfactor * self.rollout(samplestate, h, depth + 1)
        #if self.rollflag:
            #self.saveroll[self.rollcounter].append([s,action])
        return accumlate_reward

    def simulate(self, s, h, depth):
        if self.discountfactor ** depth < self.epsilon:
            return 0
        if self.tree.isLeaf(h):  # if h is  leaf
            for action in self.actions:
                self.tree.ExpandTreeFrom(h,action , IsAction=True)
            new_val = self.rollout(s, h, depth)
            #if self.rollflag:
                #self.rollflag=False
                #self.rollcounter +=1
                #self.saveroll.append([-1])
            self.tree.nodes[h].countN += 1
            self.tree.nodes[h].value = new_val
            return new_val

        acc_reward = 0
        next_action, next_node = self.searchBest(h, UseUCB=True)
        sample_state, sample_obs, sample_reward = self.blackbox(s, next_action)
        if sample_state==-2:
            return acc_reward
        Next_node = self.getObservationNode(next_node, sample_obs)
        acc_reward += sample_reward + self.discountfactor * self.simulate(sample_state, Next_node, depth + 1)
        self.tree.nodes[h].belief.append(s)  ## i am not sure its fit here
        if len(self.tree.nodes[h].belief)>self.num_of_patricles:
            self.tree.nodes[h].belief=self.tree.nodes[h].belief[1:]
        self.tree.nodes[h].countN += 1
        self.tree.nodes[next_node].countN += 1
        self.tree.nodes[next_node].value += (acc_reward - self.tree.nodes[next_node].value) / self.tree.nodes[
            next_node].countN
        #self.tree.nodes[Next_node].belief.append(sample_state)# need to remove this its just for trying
        return acc_reward

    def getObservationNode(self, h, sample_obs):
        if sample_obs not in self.tree.nodes[h].childnodes.keys():
            self.tree.ExpandTreeFrom(h, sample_obs)
        nextnode = self.tree.nodes[h].childnodes[sample_obs]
        return nextnode

    # searchBest action to take
    # UseUCB = False to pick best value at end of Search()
    def searchBest(self, h, UseUCB=True):
        maxval = None
        node_result = None
        action_result = None
        if UseUCB:
            if self.tree.nodes[h].belief != -1:
                childrens = self.tree.nodes[h].childnodes
                for action, child in childrens.items():
                    if self.tree.nodes[child].countN == 0:
                        return action, child
                    ucb = self.tree.UCB(self.tree.nodes[child].countN, self.tree.nodes[h].countN,
                                        self.tree.nodes[child].value, self.c)

                    # we save max
                    if maxval is None or maxval < ucb:
                        maxval = ucb
                        node_result = child
                        action_result = action
            return action_result, node_result
        else:
            if self.tree.nodes[h].belief != -1:
                children = self.tree.nodes[h].childnodes
                # pick the optimal value node for termination
                for action, child in children.items():
                    node_val = self.tree.nodes[child].value
                    # save max
                    if maxval is None or maxval < node_val:
                        maxval = node_val
                        node_result = child
                        action_result = action
            return action_result, node_result

    def PostSample(self, bh, action, observation, d):
        if bh == []:
            s = choice(self.initStates)
        else:
            s = choice(bh)
        # sample from transition distribution
        s_next, o_next, _ = self.blackbox(s, action)
        if s_next==-2:
            print(" need to fix this error in post sample")
            print("the  state we sent to blackbox is, ",s)
            return s
        #if o_next == observation or d == 100:
            #return s_next
        #self.PostSample(bh, action, observation, d + 1)
        if o_next == observation:
            return s_next
        result = self.PostSample(bh, action, observation, d + 1)
        return result

    def UpdateBelief(self, action, observation):
        #prior = self.tree.nodes[-1].belief.copy()
        #self.tree.nodes[-1].belief = []
        #for i in range(self.num_of_patricles):
         #   self.tree.nodes[-1].belief.append(self.PostSample(prior, action, observation, 0))
        prior = self.tree.nodes[self.tree.currRoot].belief.copy()
        self.tree.nodes[self.tree.currRoot].belief = []
        for i in range(self.num_of_patricles):
            self.tree.nodes[self.tree.currRoot].belief.append(self.PostSample(prior, action, observation, 0))
