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
        self.publicAcitons=[1,2]
        self.TempactionSpace = [ele for ele in product(range(0, 3), repeat=self.numberOfAgents)]
        self.actionSpace=[]
        self.obs_per_agent = [0, 1]  # 0 means hearing tiger on left, 1 means hearing tiger on right
        self.observationSpace=self.obs_per_agent
        #self.observationSpace = [ele for ele in product(range(0, 2), repeat=self.numberOfAgents)]
        #self.initialState = self.generate_state()
        self.traces=traces
        self.agent_id=agent_id
        self.context=[{1:{0:0,1:0},2:{0:0,1:0}},{1:{0:0,1:0},2:{0:0,1:0}}]
        self.relaxed=[{1:{},2:{}},{1:{},2:{}}]
        for action in self.TempactionSpace:
            if action[1-self.agent_id]==0 and action[self.agent_id]!=0:
                continue
            else:
                self.actionSpace.append(action)




    def generateInitStates(self):
        return self.states

    def extract_ca_from_traces(self):
        e_i_t={0:{1:{0:[],1:[]},2:{0:[],1:[]}},1:{1:{0:[],1:[]},2:{0:[],1:[]}}} #mapping between e to list  re    e= number of agent,number of action,number of state/context hard coded
        for trace in self.traces:
            total_cost_per_agent=[trace['total_cost']/2,trace['total_cost']/2]#cit
            total_reward_per_agent=[trace['total_reward']/2,trace['total_reward']/2]#rit           need to change it for to suit the paper
            ca_costs=[0,0]
            for i in range(0,trace['trace_len']):
                action = self.TempactionSpace[trace['actions'][i]]
                for j in range(0,len(action)):
                    if action[j] in self.publicAcitons:
                        we=ca_costs[j]/total_cost_per_agent[j]
                        re=we*total_reward_per_agent[j]
                        ca_costs[j]=0
                        e_i_t[j][action[j]][trace['states'][i]].append(re)
                    else:
                        if trace['rewards'][i]<0:
                            ca_costs[j]+=trace['rewards'][i]/2
        for i in range(0,2):
            for j in range(1,3):
                for k in range(0,2):
                    if len(e_i_t[i][j][k])>0:
                        self.context[i][j][k]=sum(e_i_t[i][j][k])/len(e_i_t[i][j][k])







                """if trace['rewards'][i]>0 or trace['next_states'] not in trace['states'][0:i+1]:
                    contribute_steps[i]=ca_cost
                    ca_cost=0
                if trace['rewards'][i]>0:
                    toatl_reward+=trace['rewards'][i]
                else:
                    total_cost+=trace['rewards'][i]
                    total_cost_per_agent[0]+=trace['rewards'][i]/2
                    total_cost_per_agent[1] += trace['rewards'][i] / 2
                    ca_cost+=trace['rewards'][i]
                action=self.actionSpace[trace['actions'][i]]
                for j in len(action):
                    if action[j] in self.publicAcitons:
                        self.context[j][action[j]][trace['states'][i]]='info'
                        self.context[j][action[j]][trace['states'][i]] = 'info'
            total_reward_per_agent[0]+= toatl_reward*(total_cost_per_agent[0]/total_cost)
            total_reward_per_agent[1] += toatl_reward * (total_cost_per_agent[1] / total_cost)"""








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






