from POMCP2 import POMCP
from boxPushWithAdapter import boxPushProblem
import numpy as np
from numpy.random import binomial, choice, multinomial
from timeit import default_timer as timer
import random
import pandas as pd
from adapter import adapter
from Dec_Tiger import Dec_Tiger
from collections import Counter
import math

problem=Dec_Tiger(2)
a=problem.generateInitStates()
init_states=np.arange(0,len(a),1)
actions=np.arange(0,len(problem.actionSpace),1)
obs=np.arange(0,len(problem.observationSpace),1)
ab = POMCP(problem.blackbox,problem, discountfactor=0.95,c=110,horizon=2)
ab.initialize(init_states, actions, obs)
tree=ab.tree
counter_train=0
list_of_hisory_belief=[]
def train(init_state):
    real_state = init_state
    time = 0
    sumRewards = 0
    actioncounts = 0
    #start = timer()
    root = ab.tree.currRoot
    actionNode_realState=[]
    while time <= 15:
        time += 1
        print("current state: ", real_state)
        beleif_info=dict(Counter(tree.nodes[root].belief).items())
        if beleif_info!={}:
            list_of_hisory_belief.append(beleif_info)
        print(f"belief state: {beleif_info}")
        if real_state == -2:
            print("process finished")
            break
        #temp0 = getBelief(ab.tree.nodes[root].belief)
        action = ab.search(root)
        actioncounts += 1
        print("action chosen: ", problem.actionSpace[action])
        next_state, observation, reward = problem.blackbox(real_state, action)
        print("next state: ", next_state)
        print("observation recievd : ", problem.observationSpace[observation])
        print("reward : ", reward)
        sumRewards += reward

        real_state = next_state
        root = ab.tree.prune_after_action(action, observation)
        #temp1=getBelief(ab.tree.nodes[root].belief)
        ab.UpdateBelief(action, observation)
        #temp2=getBelief(ab.tree.nodes[root].belief)
        if len(ab.tree.nodes[ab.tree.nodes[root].parent].childnodes)>1:
            print(f"action that added to expand is {problem.actionSpace[action]} ")
            actionNode_realState.append((ab.tree.nodes[root].parent,next_state,action))
        if next_state == -2:
            print("process finished")
            break
    print("the sum of rewards ", sumRewards)
    print("number of action until goal", actioncounts)
    #print("precentege of fail ", problem.countbad / (problem.countbad + problem.countgood))
    succ_flag= True if time<16 else False
    return actionNode_realState,succ_flag


def belief_similiarity(curr_belief_dict,list_of_prev_belief_dicts):
    min_diff=100000000
    for root_dict in list_of_prev_belief_dicts:
        diff=0
        keys = set(curr_belief_dict.keys()).union(root_dict.keys())
        for key in keys:
            diff+=pow((curr_belief_dict.get(key,0)-root_dict.get(key,0)),2)
        min_diff=min(min_diff,math.sqrt(diff))
    return min_diff





startALLtimer = timer()
print()
print(f"train number {counter_train}")

counter_train+=1
more_nodes,flag=train(problem.initialState)
print("********* start expanding *****************")

#index=0
#nodes_list_length=len(more_nodes)
while(len(more_nodes)>0):
    node=more_nodes.pop(0)
    childrens=tree.getChildrens(node[0])
    for key,child in childrens.items():
        if child not in tree.history_of_roots:
            tree.currRoot=child
            ab.UpdateBelief(node[2], key)
            curr_info = dict(Counter(tree.nodes[child].belief).items())
            diff_in_bstate=belief_similiarity(curr_info,list_of_hisory_belief)

            print()
            print(f"train number {counter_train}")
            counter_train += 1
            print(f"min_diff is {diff_in_bstate}")
            if diff_in_bstate < 50:
                continue
            print(f"obs in this expand {problem.observationSpace[key]}")
            temp,flag=train(node[1])
            if flag:
                #nodes_list_length += len(temp)
                more_nodes.extend(temp)
    #index +=1
endAllTimer = timer()
print("time of all proccess  ",endAllTimer - startALLtimer)

def printTree(tree,root):
    childs_list = [root]
    while(len(childs_list)>0):
        curr_node=childs_list.pop(0)
        print(f"node number {curr_node}")
        print(f"node belief state {dict(sorted(Counter(tree.nodes[curr_node].belief).items()))}")
        action_child=tree.nodes[curr_node].childnodes
        if len(action_child)>1:
            print("ERRRRRRRRRRRRORRRRRRR")  # in the pruned tree every obs node suppoused to have only 1 action child
        if len(action_child)==0:
            continue
        action_number,node_id=list(action_child.items())[0]
        print(f"action suggested {problem.actionSpace[action_number]}")
        for key,obs_node in tree.nodes[node_id].childnodes.items():
            print(f"got observation {problem.observationSpace[key]} the suit node is {obs_node}")
            if key == -2:
                continue
            childs_list.append(obs_node)

printTree(tree,-1)





