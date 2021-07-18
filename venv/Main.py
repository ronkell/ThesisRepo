from POMCP import POMCP
from boxPushProblem import boxPushProblem
import numpy as np
from numpy.random import binomial, choice, multinomial
from timeit import default_timer as timer
import random
import pandas as pd

d={'time_step':[],'xa1':[],'ya1':[],'xa2':[],'ya2':[],'xb1':[],'yb1':[],'xb2':[],'yb2':[],'xb3':[],'yb3':[],'action1':[],'action2':[]}
df=pd.DataFrame(columns=d)
print(df)
dfcount=0
problem=boxPushProblem(2,2,2,1)
# turn into numpy **************
#init_state = np.empty(2*len([problem.initialState]), dtype=object)
init_state = np.empty(len([problem.initialState]), dtype=object)
init_state[:]=[problem.initialState]
#adding more init state
#initstate_2nd=[(random.randrange(0,3),random.randrange(0,3)) for i in range(0,2)]
#initstate_2nd.extend(problem.initialState[2:])
#init_state[1]=initstate_2nd
#init_state=np.append(init_state,np.array([initstate_2nd]))
#actions = np.empty(len(problem.actionSpace), dtype=object)
#actions[:]=problem.actionSpace
#obs = np.empty(len(problem.observationSpace), dtype=object)
#obs[:]=problem.observationSpace
actions=np.arange(0,len(problem.actionSpace),1)
obs=np.arange(0,len(problem.observationSpace),1)
# ******************************
"""
ab=POMCP(problem.blackbox,discountfactor = 0.95)
ab.initialize(init_state,actions,obs)
real_state=problem.initialState
time=0
sumRewards=0
actioncounts=0;
start = timer()
root=ab.tree.currRoot
policy=dict()
while time<=15:
    time+=1
    print("current state: ",real_state)
    if problem.isFinalState(real_state)==-2:
        print("process finished")
        break
    #actiontimeStart=timer()
    action=ab.search(root)
    actioncounts +=1
    #actiontimeEnd = timer()
    #print("finding action time: ",actiontimeEnd - actiontimeStart)
    print("action chosen: ",problem.actionSpace[action])
    blist = np.empty(len(ab.tree.nodes[root].belief), dtype=object)
    blist[:] = ab.tree.nodes[root].belief
    policyState = choice(blist)
    policyState=tuple(policyState)
    policy[policyState]=[action,real_state]
    #print(ab.tree.nodes[-1].belief)
    next_state,observation,reward=problem.blackbox(real_state,action)
    print("next state: ", next_state)
    print("observation recievd : ", problem.observationSpace[observation])
    print("reward : ",reward)
    sumRewards+=reward
    if next_state==-2:
        print("process finished")
        break
    real_state=next_state
    #prune_belieftimeStart = timer()
    root=ab.tree.prune_after_action(action, observation)
    ab.UpdateBelief(action,observation)
    #prune_belieftimeEnd = timer()
    #print("prune + update belief: ", prune_belieftimeEnd - prune_belieftimeStart)
end = timer()
print(end-start)
print("the sum of rewards ",sumRewards)
print("number of action until goal", actioncounts)
print("policy from belief to action")
print("precentege of fail ", problem.countbad/(problem.countbad+problem.countgood))
for state,pair in policy.items():
    print("belief state ",state,"real state ",pair[1]," action ",problem.actionSpace[pair[0]])
#print(ab.ends)
#for i in range(0,len(ab.ends)):
#    print(ab.tree.nodes[ab.ends[i]].belief)"""
"""for i in range(0,len(ab.saveroll)):
    for j in range(0,len(ab.saveroll[i])):
        if ab.saveroll[i][j] != -1:
            print(ab.saveroll[i][j][0], " ",problem.actionSpace[ab.saveroll[i][j][1]])
    print("***************")"""
traces=[]
for j in range(0,10):
    ab = POMCP(problem.blackbox, discountfactor=0.95)
    ab.initialize(init_state, actions, obs)
    real_state = problem.initialState
    time = 0
    sumRewards = 0
    actioncounts = 0;
    start = timer()
    root = ab.tree.currRoot
    policy = dict()
    while time <= 15:
        time += 1
        print("current state: ", real_state)
        if problem.isFinalState(real_state) == -2:
            print("process finished")
            break
        # actiontimeStart=timer()
        action = ab.search(root)
        actioncounts += 1
        # actiontimeEnd = timer()
        # print("finding action time: ",actiontimeEnd - actiontimeStart)
        print("action chosen: ", problem.actionSpace[action])
        blist = np.empty(len(ab.tree.nodes[root].belief), dtype=object)
        blist[:] = ab.tree.nodes[root].belief
        policyState = choice(blist)
        policyState = tuple(policyState)
        policy[policyState] = [action, real_state]
        # print(ab.tree.nodes[-1].belief)
        next_state, observation, reward = problem.blackbox(real_state, action)
        print("next state: ", next_state)
        print("observation recievd : ", problem.observationSpace[observation])
        print("reward : ", reward)
        sumRewards += reward
        if next_state == -2:
            print("process finished")
            break
        real_state = next_state
        # prune_belieftimeStart = timer()
        root = ab.tree.prune_after_action(action, observation)
        ab.UpdateBelief(action, observation)
        # prune_belieftimeEnd = timer()
        # print("prune + update belief: ", prune_belieftimeEnd - prune_belieftimeStart)
    end = timer()
    print(end - start)
    print("the sum of rewards ", sumRewards)
    print("number of action until goal", actioncounts)
    print("policy from belief to action")
    print("precentege of fail ", problem.countbad / (problem.countbad + problem.countgood))
    k=0
    for state, pair in policy.items():
        print("belief state ", state, "real state ", pair[1], " action ", problem.actionSpace[pair[0]])
        df.loc[dfcount]=[k,state[0][0],state[0][1],state[1][0],state[1][1],state[2][0],state[2][1],state[3][0],state[3][1],state[4][0],state[4][1],problem.actionSpace[pair[0]][0],problem.actionSpace[pair[0]][1]]
        dfcount+=1
        k+=1
    traces.append(policy)
df.to_csv("tracesagent1.csv")


