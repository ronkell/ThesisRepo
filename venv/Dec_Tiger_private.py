import random
import itertools
import numpy as np
from itertools import product
from timeit import default_timer as timer
from adapter import adapter

class Dec_Tiger_private():
    def __init__(self,traces,number_of_agents,agent_id ):
        self.numberOfAgents = number_of_agents
        self.states = [0, 1]  # 0 means tiger on the left 1 means tiger on the right
        self.actions_per_agent = [0, 1, 2]  # 0 listen, 1 open left, 2 open right
        self.TempactionSpace = [ele for ele in product(range(0, 3), repeat=self.numberOfAgents)]
        self.actionSpace=[]
        self.obs_per_agent = [0, 1]  # 0 means hearing tiger on left, 1 means hearing tiger on right
        self.observationSpace=self.obs_per_agent
        #self.observationSpace = [ele for ele in product(range(0, 2), repeat=self.numberOfAgents)]
        self.initialState = self.generate_state()
        self.traces=traces
        self.agent_id=agent_id
        for action in self.TempactionSpace:
            if action[1-self.agent_id]==0 and action[self.agent_id]!=0:
                continue
            else:
                self.actionSpace.append(action)




    def generateInitStates(self):
        return self.states

    def extract_ca_from_traces(self):
        for trace in self.traces:
            total_cost=0
            total_cost_per_agent=0
            toatl_reward=0
            ca_cost=0
            for i in range(0,len(trace['actions'])):

                action=self.actionSpace[trace['actions'][i]]



                if trace['rewards'][i]>0:
                    toatl_reward+=trace['rewards'][i]
                else:
                    total_cost+=trace['rewards'][i]
                    ca_cost+=trace['rewards'][i]






    def blackbox(self, stateNum, actionNumber):
        reward = 0
        observation = 0
        action_all_agents = self.actionSpace[actionNumber]
        counter_left=0
        counter_right=0
        is_final=False
        next_state=stateNum
        for i in range(0,len(action_all_agents)):
            action=action_all_agents[i]
            if action==0 and i==self.agent_id:
                reward-=1
                prob_of_right_obs = random.uniform(0, 1)
                if prob_of_right_obs<=0.85:
                    observation=stateNum
                else:
                    observation=abs(stateNum-1)
            if action==1:
                counter_left+=1
                is_final=True
            if action==2:
                counter_right+=1
                is_final=True

        if is_final:
            next_state=-2
            if stateNum == 0:
                if counter_left > 0:
                    reward+=10
                if counter_right>0:
                    penalty=400
                    #reward-=(penalty*pow(0.25,counter_right))
                    reward -= (penalty * 0.25* counter_right)
            elif stateNum==1:
                if counter_left > 0:
                    penalty = 400
                    #reward -= (penalty*pow(0.25,counter_left))
                    reward -= (penalty * 0.25* counter_left)
                if counter_right>0:
                    reward+=10


        return (next_state,self.observationSpace.index(tuple(observation)),reward)






