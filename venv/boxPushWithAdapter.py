import random
import itertools
import numpy as np
from itertools import product
from timeit import default_timer as timer
from adapter import adapter
# index description:
# nothing=0,    left=1,   right=2,    up=3,   down=4, sense=5,    push=6,     colpush=7
# actions= [(a1,a2,a3),(a1,a2,a3)...]
#state= [(x1,y1),(x2,y2)...(xn,yn),(bx1,by1)..(bxk,byk),(Bx1,By1)...(Bxm,Bym)] size= n+k+m

#obs [o1,o2..on] where  oi=-1(not sure if needed i remove for now)  if no sense oi=0 if sense+no box or oi=1 if sense+box in loc.

class boxPushProblem():
   def __init__(self,numberOfAgents,gridSize,numberOfSmallBoxes,numberOfBigBoxes):
      self.gridSize=gridSize #tuple
      self.numberOfAgents=numberOfAgents
      self.numberOfSmallBoxes=numberOfSmallBoxes
      self.numberOfBigBoxes=numberOfBigBoxes
      self.initialState=[(random.randrange(0,gridSize),random.randrange(0,gridSize)) for i in range(0,numberOfAgents+numberOfSmallBoxes+numberOfBigBoxes)]
      self.realState=self.initialState.copy()
      self.actionSpace=[ele for ele in product(range(0, 8), repeat=numberOfAgents)]
      self.observationSpace=[ele for ele in product(range(0,2), repeat=numberOfAgents)]
      self.counter=numberOfSmallBoxes+numberOfBigBoxes
      self.epsilon=0.1
      self.countgood=0
      self.countbad=0
      self.initialStateDisterbution=[]
      self.boxadapter=adapter()



   def generateInitStates(self):
      locations_of_agents=[x for x in self.initialState[:self.numberOfAgents]]
      x_loc=list(range(0,self.gridSize))
      y_loc=list(range(0,self.gridSize))
      locations=list(product(x_loc,y_loc))
      all_possible_locations=[ele for ele in product(locations, repeat=(self.numberOfSmallBoxes+self.numberOfBigBoxes))]
      for x in all_possible_locations:
         self.initialStateDisterbution.append(locations_of_agents+list(x))
      return self.initialStateDisterbution



   def set_initState(self,s):
      self.initialState=s

   def isFinalState(self,state):
      final_state_flag = True
      for i in range(0, self.numberOfSmallBoxes + self.numberOfBigBoxes):
         boxindex = self.numberOfAgents + i
         if state[boxindex] != (-1, -1):
            final_state_flag = False
      if final_state_flag:
         # endBlack = timer()
         # print(endBlack - startBlack)
         return -2
      else:
         return 0

   def validactionsforrollout(self,stateNum):
      state = self.boxadapter.numberToState(stateNum)
      notvalidactions=[]
      for i in range(0,self.numberOfAgents):
         loc_agent_i=state[i]
         not_valid_for_agent_i=[]
         if loc_agent_i[0]==0:
            not_valid_for_agent_i.append(1)
         if loc_agent_i[0]==self.gridSize-1:
            not_valid_for_agent_i.append(2)
         if loc_agent_i[1] == 0:
            not_valid_for_agent_i.append(4)
         if loc_agent_i[1] == self.gridSize - 1:
            not_valid_for_agent_i.append(3)
         if i%2==0:
            notvalidactions.extend(list(product(not_valid_for_agent_i,list(range(0,8)))))
         else:
            notvalidactions.extend(list(product(list(range(0, 8)),not_valid_for_agent_i)))
      notvalidactions=set(notvalidactions)
      fine_actions=list(range(0,len(self.actionSpace)))
      for a in notvalidactions:
         fine_actions.remove(self.actionSpace.index(a))
      return fine_actions







   def blackbox(self,stateNum,actionNumber):
      """
      :param state:number]
      :param action:(a1,a2,a3..an) i change it for number and then i will convert it
      :return:(next state,observation,reward)
      """
      punishflag_for_colPUSH=False
      #startBlack = timer()
      state=self.boxadapter.numberToState(stateNum)
      final_state_flag=True
      for i in range(0,self.numberOfSmallBoxes+self.numberOfBigBoxes):
         boxindex=self.numberOfAgents+i
         if state[boxindex]!=(-1,-1):
            final_state_flag=False
      if final_state_flag:
         #endBlack = timer()
         #print(endBlack - startBlack)
         return (-2,-2,-2)

      next_state=state.copy()
      reward=0
      observation=[0]*self.numberOfAgents
      action=self.actionSpace[actionNumber]
      col_push_dict=dict() # (x,y)->counterAgent
      chance_to_fail=random.uniform(0,1)
      for i in range(0,len(action)):
         reward-=1
         if action[i]==1 and next_state[i][0] > 0:
            next_state[i]=(next_state[i][0]-1,next_state[i][1])
         if action[i]==2 and next_state[i][0] < self.gridSize-1:
            next_state[i]=(next_state[i][0]+1,next_state[i][1])
         if action[i]==3 and next_state[i][1] <self.gridSize-1:
            next_state[i]=(next_state[i][0],next_state[i][1]+1)
         if action[i]==4 and next_state[i][1] >0:
            next_state[i]=(next_state[i][0],next_state[i][1]-1)
         if action[i]==5 and state[i] in next_state[self.numberOfAgents:] :
            observation[i]=1
            if state[i] in next_state[self.numberOfAgents+self.numberOfSmallBoxes:]:
               reward+=2
         if action[i]==6:
            if state[i] in next_state[self.numberOfAgents:(self.numberOfAgents+self.numberOfSmallBoxes)]:
               if chance_to_fail<self.epsilon:
                  #print("fail ",chance_to_fail)
                  self.countbad += 1
               else:
                  self.countgood +=1
                  box_index = next_state.index(state[i], self.numberOfAgents)
                  next_state[box_index] = (-1, -1)
                  reward += 300
            else:
               reward -= 60

         if action[i]==7:
            if  punishflag_for_colPUSH == False:
               punishflag_for_colPUSH=True
            if state[i] in col_push_dict.keys():
               col_push_dict[state[i]] +=1
            else:
               col_push_dict[state[i]] = 1

      for i in range(0,self.numberOfBigBoxes):
         bigboxindex=self.numberOfAgents+self.numberOfSmallBoxes+i
         if state[bigboxindex] in col_push_dict.keys() and col_push_dict[state[bigboxindex]]>1:
            next_state[bigboxindex]=(-1,-1)
            reward+=500
            punishflag_for_colPUSH=False
      #****** big rewad if the next state is finish
      finishState=True
      for i in range(0, self.numberOfSmallBoxes + self.numberOfBigBoxes):
         boxindex = self.numberOfAgents + i
         if state[boxindex] != (-1, -1):
            finishState = False
      if finishState:
         reward += 1000
      if punishflag_for_colPUSH == True:
         reward -= 60
      next_state_num=self.boxadapter.stateToNumber(next_state)
      return (next_state_num,self.observationSpace.index(tuple(observation)),reward)






















