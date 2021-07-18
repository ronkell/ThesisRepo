import random
import itertools
import numpy as np
from itertools import product
from timeit import default_timer as timer
# index description:
# nothing=0,    left=1,   right=2,    up=3,   down=4, sense=5,    push=6,     colpush=7 ** simulate the oteragent push=8
# actions= now its ai because its one agent acton while ai=0...9 one  from above
#state= [(xi,yi),(bx1,by1)..(bxk,byk),(Bx1,By1)...(Bxm,Bym)] size= 1+k+m for agent i

#obs oi=0,1 for one agent

class boxPushSingle():
    def __init__(self, numberOfAgents, gridSize, numberOfSmallBoxes, numberOfBigBoxes,traces):
        self.gridSize = gridSize  # tuple
        self.numberOfAgents = numberOfAgents
        self.numberOfSmallBoxes = numberOfSmallBoxes
        self.numberOfBigBoxes = numberOfBigBoxes
        self.actionSpace = [x for x in range(0,8)]
        self.observationSpace = [x for x in range(0,2)]
        self.counter = numberOfSmallBoxes + numberOfBigBoxes
        self.epsilon = 0.1
        self.countgood = 0
        self.countbad = 0
        self.traces=traces
        self.contextP=set()
        self.contextCP=set()
        self.relaxedContextP = set()
        self.relaxedContextCP = set()
        self.otherAgentsAction=dict()


    def blackbox(self, state, actionNumber):
        """
        :param state:[(xi,yi),(bx1,by1)..(bxk,byk),(Bx1,By1)...(Bxm,Bym)] for agent i
        :param action: ai action from agent i
        :return:(next state,observation,reward)
        """
        next_state = state.copy()
        reward = 0
        observation = 0
        action = self.actionSpace[actionNumber]
        chance_to_fail = random.uniform(0, 1)
        reward -= 1
        if action == 1 and next_state[0][0] > 0:
            next_state[0] = (next_state[0][0] - 1, next_state[0][1])
        if action == 2 and next_state[0][0] < self.gridSize - 1:
            next_state[0] = (next_state[0][0] + 1, next_state[0][1])
        if action == 3 and next_state[0][1] < self.gridSize - 1:
            next_state[0] = (next_state[0][0], next_state[0][1] + 1)
        if action == 4 and next_state[0][1] > 0:
            next_state[0] = (next_state[0][0], next_state[0][1] - 1)
        if action == 5 and state[0] in next_state[1:]:
            observation = 1

        if action == 6 :
            isInContext=self.checkContext(state,6)
            if isInContext == 1:
                reward += 300
                box_index = next_state.index(state[0],1) if state[0] in next_state[1:] else -1
                if box_index > -1:
                    next_state[box_index] = (-1, -1)

            elif isInContext == 0:
                reward += 1
            else:
                reward -= 50

        if action == 7:
            isInContext = self.checkContext(state, 7)
            if isInContext==1:
                reward += 500
                bigbox_index = next_state.index(state[0], 1+self.numberOfSmallBoxes) if state[0] in next_state[1+self.numberOfSmallBoxes:] else -1
                if bigbox_index>-1:
                    next_state[bigbox_index] = (-1, -1)

            elif isInContext == 0:
                reward += 1
            else:
                reward -= 50
        if action in self.otherAgentsAction.keys():
            tempstate=self.otherAgentsAction[action]
            box_index = next_state.index(tempstate,1) if tempstate in next_state[1:] else -1
            if box_index > -1:
                next_state[box_index] = (-1, -1)
                reward += 10
            else:
                reward -= 20


        return (next_state, observation, reward)

    def setContext(self,c,rc,cp,rcp):
        self.contextP=c
        self.contextCP=cp
        self.relaxedContextP = rc
        self.relaxedContextCP =rcp
    def checkContext(self,givenstate,action):
        state=tuple(givenstate)
        if action == 6:
            if state in self.contextP:
                return 1
            elif state in self.relaxedContextP:
                return 0
            else:
                return -1
        if action == 7:
            if state in self.contextCP:
                return 1
            elif state in self.relaxedContextCP:
                return 0
            else:
                return -1
    def addOtherAgentsActions(self,actionsToAdd,startindex):
        index=startindex
        for p in actionsToAdd:
            self.otherAgentsAction[index]=p
            self.actionSpace.append(index)
            index += 1

    def isFinalState(self, state):
        final_state_flag = True
        for i in range(0, self.numberOfSmallBoxes + self.numberOfBigBoxes):
            boxindex = 1 + i
            if state[boxindex] != (-1, -1):
                final_state_flag = False
        if final_state_flag:
            # endBlack = timer()
            # print(endBlack - startBlack)
            return -2
        else:
            return 0




