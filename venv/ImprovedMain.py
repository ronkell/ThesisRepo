from POMCP2 import POMCP
from boxPushWithAdapter import boxPushProblem
import numpy as np
from numpy.random import binomial, choice, multinomial
from timeit import default_timer as timer
import random
import pandas as pd
from adapter import adapter
from collections import Counter


problem=boxPushProblem(2,2,2,1)
a=problem.generateInitStates()
boxadapter=problem.boxadapter
number_of_init_states=boxadapter.initstates(a)
init_states=np.arange(0,number_of_init_states,1)
actions=np.arange(0,len(problem.actionSpace),1)
obs=np.arange(0,len(problem.observationSpace),1)

ab = POMCP(problem.blackbox,problem, discountfactor=0.95,c=300)
ab.initialize(init_states, actions, obs)
tree=ab.tree
#real_state = problem.initialState

def getBelief(state_list):
    bdict=dict()
    for s in state_list:
        bdict.setdefault(s,0)
        bdict[s] +=1
    return bdict

def train(init_state):
    real_state = init_state
    time = 0
    sumRewards = 0
    actioncounts = 0
    #start = timer()
    root = ab.tree.currRoot
    actionNode_realState=[]
    while time <= 15:
        print(f"iter number {time}")
        time += 1
        print("current state: ", real_state)
        print(f"belief state: {dict(sorted(Counter(tree.nodes[root].belief).items()))}")
        if problem.isFinalState(real_state) == -2:
            print("process finished")
            break
        #temp0 = getBelief(ab.tree.nodes[root].belief)
        action = ab.search(root)
        actioncounts += 1
        print("action chosen: ", problem.actionSpace[action])
        next_state, observation, reward = problem.blackbox(boxadapter.stateToNumber(real_state), action)
        print("next state: ", boxadapter.numberToState(next_state))
        print("observation recievd : ", problem.observationSpace[observation])
        print("reward : ", reward)
        sumRewards += reward
        if next_state == -2:
            print("process finished")
            break
        real_state = boxadapter.numberToState(next_state)
        root = ab.tree.prune_after_action(action, observation)
        #temp1=getBelief(ab.tree.nodes[root].belief)
        ab.UpdateBelief(action, observation)
        #temp2=getBelief(ab.tree.nodes[root].belief)
        if len(ab.tree.nodes[ab.tree.nodes[root].parent].childnodes)>1:
            actionNode_realState.append((ab.tree.nodes[root].parent,next_state))


        """if time >12:
            print("before ")
            for key,val in temp1.items():
                print("state ", boxadapter.numberToState(key)," count ", val)
            print("after")
            for key,val in temp2.items():
                print("state ", boxadapter.numberToState(key)," count ", val)"""

    #end = timer()
    #print("time ",end - start)
    print("the sum of rewards ", sumRewards)
    print("number of action until goal", actioncounts)
    print("precentege of fail ", problem.countbad / (problem.countbad + problem.countgood))
    succ_flag= True if time<16 else False
    return actionNode_realState,succ_flag


startALLtimer = timer()
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
            temp,flag=train(boxadapter.numberToState(node[1]))
            if flag:
                #nodes_list_length += len(temp)
                more_nodes.extend(temp)
    #index +=1
endAllTimer = timer()
print("time of all proccess  ",endAllTimer - startALLtimer)




