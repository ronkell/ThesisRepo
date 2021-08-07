import random
import itertools
import numpy as np
from itertools import product
from timeit import default_timer as timer
from adapter import adapter
import math
#actions per agent (0 idle),(1 left),(2 right),(3 up) (4 down) (5-x where x=5+number of boxes sense) (x+1-y where y=4*number of small boxes) (y+1 z where z=4*number of big boxes)
#obs per agent (0 idle) (1 no box) (2 yes box)
class gridBoxlinear():
    def __init__(self,numberOfAgents,grid_width,grid_height,numberOfSmallBoxes,numberOfBigBoxes,agents_init_locations,boxes_init_locations):
        self.grid_width=grid_width
        self.grid_height = grid_height
        self.numberOfAgents=numberOfAgents
        self.numberOfSmallBoxes=numberOfSmallBoxes
        self.numberOfBigBoxes=numberOfBigBoxes
        self.initialState=agents_init_locations+boxes_init_locations
        #self.realState=self.initialState.copy()
        self.objects_indexes=list(range(0,numberOfAgents+numberOfSmallBoxes+numberOfBigBoxes))
        move_idle_actions=list(range(0,5))
        self.moveindexes=(1,4)
        sense_actions_perbox=list(range(5,5+numberOfSmallBoxes+numberOfBigBoxes))
        pushstartindex=sense_actions_perbox[-1]+1
        self.senseindexes=(5,pushstartindex-1)
        push_actions_perbox=list(range(pushstartindex,pushstartindex+4*numberOfSmallBoxes))
        if len(push_actions_perbox)>0:
            collabpushstartindex=push_actions_perbox[-1]+1
        else:
            collabpushstartindex=pushstartindex
        self.pushindexes=(pushstartindex,collabpushstartindex-1)
        collab_push_perbox=list(range(collabpushstartindex,collabpushstartindex+4*numberOfBigBoxes))
        self.cpushindexes=(collabpushstartindex,collabpushstartindex+len(collab_push_perbox)-1)
        self.actions_per_agent=move_idle_actions+sense_actions_perbox+push_actions_perbox+collab_push_perbox
        actions_linear=move_idle_actions+sense_actions_perbox+push_actions_perbox
        actions_parelal=[(x,x) for x in collab_push_perbox]
        product_actions1 =list(product(actions_linear,[0]))
        product_actions2 = list(product([0], actions_linear[1:]))
        #push_collab_test_eliran=list(product(collab_push_perbox,collab_push_perbox))
        self.actionSpace=product_actions1+product_actions2+actions_parelal
        self.actionSpaceIndexed=list(range(0,len(self.actionSpace)))
        self.observationSpaceReally=[ele for ele in product(range(0,3), repeat=numberOfAgents)]
        self.observationSpace=[]
        for x in self.observationSpaceReally:
            if x[0]==0 or x[1]==0:
                self.observationSpace.append(x)
        self.counter=numberOfSmallBoxes+numberOfBigBoxes
        self.epsilon=0.1
        self.countgood=0
        self.countbad=0
        self.initialStateDisterbution=[]
        self.goal_loc=(grid_width-1,grid_height-1)
        start_or_goal=[ele for ele in product(range(0,2), repeat=self.numberOfSmallBoxes+self.numberOfBigBoxes)]
        for tup in start_or_goal:
            currstate=agents_init_locations.copy()
            for i in range(0,len(tup)):
                if tup[i]==0:
                    currstate.extend([boxes_init_locations[i]])
                elif tup[i]==1:
                    currstate.extend([self.goal_loc])
            self.initialStateDisterbution.append(currstate)
        self.move_cost=-3
        self.sense_cost=-1
        self.push_cost=-5
        self.cpush_cost=-5
        self.push_penalty=-20
        self.cpush_penalty=-20
        self.push_reward=300
        self.cpush_reward=600
        self.finish_reward=1000


    def canMove(self,loc,direction):
        """
        :param loc: (x,y)
        :param direction: 0 left 1 right 2 up 3 down
        :return: boolean if can move in that direction
        """
        if direction==0 and loc[0]>0:
            return True
        elif direction==1 and loc[0]<self.grid_width - 1:
            return True
        elif direction==2 and loc[1]<self.grid_height-1:
            return  True
        elif direction==3 and loc[1]>0:
            return True
        else:
            return False

    def move(self,state,object_ind,direction):
        """
        :param state:
        :param object_ind:
        :param direction:
        :return: next state
        """
        next_state=state.copy()
        if direction==0:
            next_state[object_ind] = (state[object_ind][0] - 1, state[object_ind][1])
            return next_state
        elif direction==1:
            next_state[object_ind] = (state[object_ind][0] + 1, state[object_ind][1])
            return next_state
        elif direction==2:
            next_state[object_ind] = (state[object_ind][0] , state[object_ind][1]+1)
            return next_state
        elif direction==3:
            next_state[object_ind] = (state[object_ind][0], state[object_ind][1] - 1)
            return next_state
        else:
            return state


    def sameLoc(self,loc1,loc2):
        if loc1==loc2:
            return True
        else:
            return False
    def Sense(self,state,index1,senseNumber):
        """
        :param state:
        :param index1: index of loc
        :param senseNumber: which box sense
        :return: observation 0 idle 1 no box 2 yes box
        """
        index_of_box=senseNumber-self.senseindexes[0]+self.numberOfAgents
        if self.sameLoc(state[index1],state[index_of_box]):
            return 2
        else:
            return 1

    def Push(self,state,agentind,pushnumber):
        next_state=state.copy()
        index_in_interval=pushnumber-self.pushindexes[0]
        box_index=math.floor(index_in_interval/4)+self.numberOfAgents
        direction_index=index_in_interval%4
        if self.sameLoc(state[agentind],state[box_index]) and self.canMove(state[agentind],direction_index):
            next_state=self.move(state,box_index,direction_index)
            return (next_state,True)
        else:
            return (state,False)

    def CPush(self,state,cpushnumber):
        next_state = state.copy()
        index_in_interval = cpushnumber - self.cpushindexes[0]
        box_index = math.floor(index_in_interval / 4)+self.numberOfAgents+self.numberOfSmallBoxes
        direction_index = index_in_interval % 4
        if self.sameLoc(state[0],state[box_index]) and self.sameLoc(state[0],state[1]) and  self.canMove(state[0],direction_index):
            next_state=self.move(state,box_index,direction_index)
            return (next_state,True)
        else:
            return (state,False)
    def checkGoal(self,state):
        for i in range(self.numberOfAgents,len(state)):
            if not self.sameLoc(state[i],self.goal_loc):
                return False
        return True

    def checkSubgoal(self,state,next_state):
        for i in range(self.numberOfAgents,len(state)):
            if not self.sameLoc(state[i],self.goal_loc) and self.sameLoc(next_state[i],self.goal_loc):
                if i<self.numberOfAgents+self.numberOfSmallBoxes:
                    return 1
                else:
                    return 2
            if self.sameLoc(state[i],self.goal_loc) and not self.sameLoc(next_state[i],self.goal_loc):
                if i<self.numberOfAgents+self.numberOfSmallBoxes:
                    return -2
                else:
                    return -3
        return 0

    def blackbox(self, state, action):
        """
        :param state:state
        :param action:(a1,a2,a3..an)
        :return:(next state,observation,reward)
        """
        next_state = state.copy()
        reward = [0]*self.numberOfAgents
        observation = [0] * self.numberOfAgents
        if self.checkGoal(state):
            return (next_state,tuple(observation),reward)
        for i in range(len(action)):
            if action[i]>=self.moveindexes[0] and action[i]<=self.moveindexes[1]:
                direction=action[i]-1
                reward[i] += self.move_cost
                if self.canMove(state[i],direction):
                    next_state=self.move(state,i,direction)



            if action[i]>= self.senseindexes[0] and action[i]<=self.senseindexes[1]:
                observation[i]=self.Sense(state,i,action[i])
                reward[i]+=self.sense_cost
                #if observation[i]==2 and  self.sameLoc(state[i],self.goal_loc):
                 #   reward[i]+=300


            if action[i]>=self.pushindexes[0] and action[i]<=self.pushindexes[1]:
                next_state,succ=self.Push(state,i,action[i])
                reward[i] += self.push_cost
                if succ==False:
                    reward[i]+=self.push_penalty


        if action[0] >= self.cpushindexes[0] and action[0] <= self.cpushindexes[1]:
            next_state,cpsucc=self.CPush(state,action[0])
            reward[0]+=self.cpush_cost
            reward[1]+=self.cpush_cost
            if cpsucc==False:
                reward[0] += self.cpush_penalty
                reward[1] += self.cpush_penalty

        if self.checkGoal(next_state):
            reward[0] += self.finish_reward
            reward[1] += self.finish_reward

        reward_multi=self.checkSubgoal(state,next_state)
        if reward_multi!=0:
            reward[0] += self.push_reward*reward_multi
            reward[1] += self.push_reward*reward_multi
        if action==(0,0):
            reward[0] -=5
            reward[1] -=5

        return (next_state,tuple(observation),reward)

    """def validactionsforrollout(self,state):
        available_actions=self.actionSpace.copy()
        available_indexes=self.actionSpaceIndexed.copy()
        for i in range(0,self.numberOfAgents):
            loc_agent_i = state[i]
            not_valid_for_agent_i = []
            if loc_agent_i[0] == 0:
                available_actions.remove()
            if loc_agent_i[0] == self.gridSize - 1:
                not_valid_for_agent_i.append(2)
            if loc_agent_i[1] == 0:
                not_valid_for_agent_i.append(4)
            if loc_agent_i[1] == self.gridSize - 1:
                not_valid_for_agent_i.append(3)"""












