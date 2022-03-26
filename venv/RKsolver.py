import random
import itertools
import numpy as np
from itertools import product
from timeit import default_timer as timer
from adapter import adapter
from gridBoxlinear import gridBoxlinear
from POMCP2 import POMCP
from collections import Counter
import math
from numpy.random import binomial, choice, multinomial
import pandas as pd
import extracteliran

class RKsolver():
    def __init__(self,problemname):
        self.problem_name=problemname
        self.problem_instance=None
        self.problem_adapter=None
        self.pomcp_solver=None
        self.tree=None
        self.traces=[]
        self.problem_dict_names={'boxMove':self.boxMoveSetup}
        self.testflag=False
        setup=self.problem_dict_names[problemname]
        setup()

    def boxMoveSetup(self):
        self.problem_instance = gridBoxlinear(2, 2, 2, 1, 1, [(0, 1), (1, 0)], [(0, 0), (0, 0)])
        self.problem_adapter=adapter(self.problem_instance)
        initstates=self.problem_adapter.initstates(self.problem_instance.initialStateDisterbution)
        init_state_array=np.array(list(range(0,initstates)))
        actions,obs=self.problem_adapter.initacitonsobs(self.problem_instance.actionSpace,self.problem_instance.observationSpace)
        self.pomcp_solver=POMCP(self.problem_adapter.blackbox,self.problem_adapter, discountfactor=0.95,c=1100,timeout=500, horizon=15)
        self.pomcp_solver.initialize(init_state_array, np.array(self.problem_adapter.actionslistindexes), np.array(self.problem_adapter.obslistindexes))
        self.tree=self.pomcp_solver.tree
        self.counter_train = 0
        self.list_of_hisory_belief = []

    def printBelief(self,info):
        for key,item in info.items():
            print(f'bstate {self.problem_adapter.numberToState(key)} : {item}')

    def train(self,init_state):
        real_state = init_state
        time = 0
        sumRewards = 0
        actioncounts = 0
        root = self.tree.currRoot
        actionNode_realState = []
        while time <= 15:
            time += 1
            print("current state: ", self.problem_adapter.numberToState(real_state))
            beleif_info = dict(Counter(self.tree.nodes[root].belief).items())
            if beleif_info != {}:
                self.list_of_hisory_belief.append(beleif_info)
            print("belief state:")
            self.printBelief(beleif_info)
            if self.problem_instance.checkGoal(self.problem_adapter.numberToState(real_state)):
                print("process finished")
                break

            action = self.pomcp_solver.search(root)
            actioncounts += 1
            print("action chosen: ", self.problem_adapter.numbertoAction(action))
            next_state, observation, reward = self.problem_adapter.blackbox(real_state, action)
            print("next state: ", self.problem_adapter.numberToState(next_state))
            print("observation recievd : ", self.problem_adapter.numbertoObservation(observation))
            print("reward : ", reward)
            sumRewards += reward
            real_state = next_state
            root = self.tree.prune_after_action(action, observation)
            # temp1=getBelief(ab.tree.nodes[root].belief)
            self.pomcp_solver.UpdateBelief(action, observation)
            # temp2=getBelief(ab.tree.nodes[root].belief)
            if len(self.tree.nodes[self.tree.nodes[root].parent].childnodes) > 1:
                print(f"action that added to expand is {self.problem_adapter.numbertoAction(action)} ")
                actionNode_realState.append((self.tree.nodes[root].parent, next_state, action))
            if self.problem_instance.checkGoal(self.problem_adapter.numberToState(real_state)):
                print("process finished")
                break
        print("the sum of rewards ", sumRewards)
        print("number of action until goal", actioncounts)

        succ_flag = True if time < 16 else False
        return actionNode_realState, succ_flag

    def belief_similiarity(self,curr_belief_dict, list_of_prev_belief_dicts):
        min_diff = 100000000
        for root_dict in list_of_prev_belief_dicts:
            diff = 0
            keys = set(curr_belief_dict.keys()).union(root_dict.keys())
            for key in keys:
                diff += pow((curr_belief_dict.get(key, 0) - root_dict.get(key, 0)), 2)
            min_diff = min(min_diff, math.sqrt(diff))
        return min_diff

    def build_tree(self):
        startALLtimer = timer()
        print()
        print(f"train number {self.counter_train}")
        self.counter_train += 1
        more_nodes, flag = self.train(self.problem_adapter.stateToNumber(self.problem_instance.initialState))
        if flag is False:
            raise Exception("failed to train")


        print("********* start expanding *****************")
        while (len(more_nodes) > 0):
            node = more_nodes.pop(0)
            childrens = self.tree.getChildrens(node[0])
            for key, child in childrens.items():
                if child not in self.tree.history_of_roots:
                    self.tree.currRoot = child
                    self.pomcp_solver.UpdateBelief(node[2], key)
                    curr_info = dict(Counter(self.tree.nodes[child].belief).items())
                    diff_in_bstate = self.belief_similiarity(curr_info, self.list_of_hisory_belief)
                    print()
                    print(f"train number {self.counter_train}")
                    self.counter_train += 1
                    print(f"min_diff is {diff_in_bstate}")
                    if diff_in_bstate < 50:
                        continue
                    print(f"obs in this expand {self.problem_adapter.numbertoObservation(key)}")
                    temp, flag = self.train(node[1])
                    if flag:
                        # nodes_list_length += len(temp)
                        more_nodes.extend(temp)
            # index +=1
        endAllTimer = timer()
        print("time of all proccess  ", endAllTimer - startALLtimer)

    def printTree(self,tree, root):
        childs_list = [root]
        while (len(childs_list) > 0):
            curr_node = childs_list.pop(0)
            print(f"node number {curr_node}")
            print(f"node belief state {dict(sorted(Counter(tree.nodes[curr_node].belief).items()))}")
            action_child = tree.nodes[curr_node].childnodes
            if len(action_child) > 1:
                print(
                    "ERRRRRRRRRRRRORRRRRRR")  # in the pruned tree every obs node suppoused to have only 1 action child
            if len(action_child) == 0:
                continue
            action_number, node_id = list(action_child.items())[0]
            print(f"action suggested {problem.actionSpace[action_number]}")
            for key, obs_node in tree.nodes[node_id].childnodes.items():
                print(f"got observation {problem.observationSpace[key]} the suit node is {obs_node}")
                if key == -2:
                    continue
                childs_list.append(obs_node)




    def genrateTraces(self,num_of_traces):
        print("****** generate traces**********")
        self.problem_adapter.sep_rewards=True
        for i in range(0, num_of_traces):
            print(f"start trace {i}")
            trace = {'actions': [], 'states': [], 'rewards': [], 'observations': [], 'next_states': [], 'bstates': [],
                     'trace_len': 0,
                     'total_cost': 0, 'total_reward': 0,'total_costs':[0,0],'total_rewards':[0,0]}
            state = choice(self.pomcp_solver.initStates)
            # beleif_state={0:500,1:500}
            root = -1
            flag = False
            count=0
            while (not flag):
                count+=1
                print(f"state is {self.problem_adapter.numberToState(state)}")
                d = dict(sorted(Counter(self.tree.nodes[root].belief).items()))
                print(f"node belief state")
                self.printBelief(d)
                action_child = self.tree.nodes[root].childnodes
                if len(action_child) > 1:
                    print(
                        "error in traces tree")  # in the pruned tree every obs node suppoused to have only 1 action child
                if len(action_child) == 0:
                    continue
                action_number, node_id = list(action_child.items())[0]
                print(f"action suggested {self.problem_adapter.numbertoAction(action_number)}")
                next_state, observation, reward = self.problem_adapter.blackbox(state, action_number)
                print("next state: ", self.problem_adapter.numberToState(next_state))
                print("observation recievd : ", self.problem_adapter.numbertoObservation(observation))
                print("reward : ", reward)
                trace['actions'].append(action_number)
                trace['states'].append(state)
                trace['next_states'].append(next_state)
                trace['observations'].append(observation)
                trace['rewards'].append(reward)
                for j in range(0,len(reward)):
                    if reward[j] > 0:
                        trace['total_rewards'][j] += reward[j]
                        trace['total_reward'] += reward[j]
                    else:
                        trace['total_costs'][j] += reward[j]
                        trace['total_cost'] += reward[j]
                trace['bstates'].append(d)

                if self.problem_instance.checkGoal(self.problem_adapter.numberToState(next_state)) or count>20:
                    flag = True
                    trace['trace_len'] = len(trace['actions'])
                    if self.problem_instance.checkGoal(self.problem_adapter.numberToState(next_state)):
                        self.traces.append(trace)

                else:
                    if observation not in self.tree.nodes[node_id].childnodes:
                        print("hello here")
                        break;
                    state = next_state
                    root = self.tree.nodes[node_id].childnodes[observation]
            print()



    def genrateTracesbyHand(self,num_of_traces):
        print("****** generate traces**********")
        self.problem_adapter.sep_rewards=True
        for i in range(0, num_of_traces):
            print(f"start trace {i}")
            trace = {'actions': [], 'states': [], 'rewards': [], 'observations': [], 'next_states': [],
                     'trace_len': 0,
                     'total_cost': 0, 'total_reward': 0,'total_costs':[0,0],'total_rewards':[0,0]}
            state = choice(self.pomcp_solver.initStates)
            flag = False
            count=0
            policyGraph=extracteliran.s
            while (not flag):
                count+=1
                print(f"state is {self.problem_adapter.numberToState(state)}")
                action=policyGraph['action']
                print(f"action suggested {action}")
                action_number=self.problem_adapter.actiontoNumber(action)
                next_state, observation, reward = self.problem_adapter.blackbox(state, action_number)
                print("next state: ", self.problem_adapter.numberToState(next_state))
                obs=self.problem_adapter.numbertoObservation(observation)
                next_policy_graph_state=policyGraph['next'][obs]
                policyGraph=extracteliran.mymap[next_policy_graph_state] if next_policy_graph_state!='final' else policyGraph
                print("observation recievd : ", obs)
                print("reward : ", reward)
                trace['actions'].append(action_number)
                trace['states'].append(state)
                trace['next_states'].append(next_state)
                trace['observations'].append(observation)
                trace['rewards'].append(reward)
                for j in range(0,len(reward)):
                    if reward[j] > 0:
                        trace['total_rewards'][j] += reward[j]
                        trace['total_reward'] += reward[j]
                    else:
                        trace['total_costs'][j] += reward[j]
                        trace['total_cost'] += reward[j]


                if self.problem_instance.checkGoal(self.problem_adapter.numberToState(next_state)) or count>20:
                    flag = True
                    trace['trace_len'] = len(trace['actions'])
                    if self.problem_instance.checkGoal(self.problem_adapter.numberToState(next_state)):
                        self.traces.append(trace)

                else:
                    state = next_state

            print()


"""

    def nextaction(self,bstate,prob,stepcount):
        #!
        if stepcount==1:
            return (4,0)
        #2
        if stepcount==2:
            return (5,0)
        #3
        if bstate==[(0,0),(1,0),(0,0),(1,1)] and prob==0.5:
            return (6,0)
        if bstate==[(0,0),(1,0),(1,1),(1,1)] and prob==0.5:
            return (6,0)
        #4
        if bstate==[(0,0),(1,0),(0,0),(0,0)] and prob==1:
            return (8,0)
        if bstate==[(0,0),(1,0),(0,0),(1,1)] and prob==1:
            return (8,0)
        if bstate==[(0,0),(1,0),(1,1),(1,1)] and prob==1:
            return (0,0)
        if bstate==[(0,0),(1,0),(1,1),(0,0)] and prob==1:
            return (0,1)
        #5
        if bstate==[(0,0),(1,0),(1,0),(1,1)] and prob==0.8:
            return (5,0)

    def updatebsate(self,prev_bstate,action,obs,prob):
        if prev_bstate[0]==(0,1) and action(4,0):
            prev_bstate[0]=(0,0)
            return (prev_bstate,-1)
        if prev_bstate[0]==(0,0) and action(2,0):
            prev_bstate[0]=(1,0)
            return (prev_bstate,-1)
        if prev_bstate[0]==(1,0) and action(1,0):
            prev_bstate[0]=(0,0)
            return (prev_bstate,-1)
        if prev_bstate[1]==(1,0) and action(1,0):
            prev_bstate[1]=(0,0)
            return (prev_bstate,-1)

        if prev_bstate[0]==(0,0) and action==(5,0) and prob==0.25:
            if obs==(2,0):
                prev_bstate[2]=(0,0)
                return (prev_bstate,0.5)
            elif obs ==(1,0):
                prev_bstate[2]=(1,1)
                return (prev_bstate,0.5)
        if prev_bstate[0]==(0,0) and action==(6,0):
           if obs==(2,0):
               prev_bstate[3]=(0,0)
               return (prev_bstate,1)
           elif obs==(1,0):
               prev_bstate[3]=(1,1)
               return (prev_bstate,1)
        if prev_bstate[0]==(1,0) and action==(6,0):
           if obs==(2,0):
               prev_bstate[3]=(1,0)
               return (prev_bstate,1)

        if prev_bstate[0]==(0,0) and action==(8,0):
            prev_bstate[2]=(1,0)
            return (prev_bstate,0,8)

        if prev_bstate[1]==(1,0) and action==(0,5):
            if obs==(2,0):
                prev_bstate[2]=(1,0)
                return (prev_bstate,1)
        if prev_bstate[1]==(1,0) and action==(0,9):
            prev_bstate[2]=(1,1)
            return (prev_bstate,0,8)
        if prev_bstate==[(0,0),(0,0),(1,1),(0,0)] and action==(12,12):
            prev_bstate[3]=(1,0)
            return (prev_bstate,0.8)
        if prev_bstate[(1,0),(1,0),(1,1),(1,0)] and action==(13,13):
            prev_bstate[3]=(1,1)
            return (prev_bstate,0.8)
"""



















