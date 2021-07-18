from POMCP import POMCP
from boxPushProblem import boxPushProblem
from boxPushSingle import boxPushSingle
import numpy as np
from numpy.random import binomial, choice, multinomial
from timeit import default_timer as timer
import random
import pandas as pd

traces = pd.read_csv('tracesagent1.csv')
#traces = traces.iloc[:, 1:2].values
traces=traces.drop('Unnamed: 0',axis=1)
print(traces)
contextP=set()
contextrelasxedP=set()
contextCP=set()
contextrelasxedCP=set()
actionForOtheragent=set()
d={'time_step':[],'xa1':[],'ya1':[],'xb1':[],'yb1':[],'xb2':[],'yb2':[],'xb3':[],'yb3':[],'action1':[]}
df=pd.DataFrame(columns=d)
dfcount=0
for i in range(len(traces)):
    if traces.loc[i,"action1"] == 6:
        contextP.add(((traces.loc[i, "xa1"], traces.loc[i, "ya1"]),(traces.loc[i, "xb1"], traces.loc[i, "yb1"]),
                      (traces.loc[i, "xb2"], traces.loc[i, "yb2"]),(traces.loc[i, "xb3"], traces.loc[i, "yb3"])))
        contextrelasxedP.add(((traces.loc[i, "xa1"], traces.loc[i, "ya1"])))
    if traces.loc[i,"action1"] == 7 and traces.loc[i,"action2"] == 7:
        contextCP.add(((traces.loc[i, "xa1"], traces.loc[i, "ya1"]), (traces.loc[i, "xb1"], traces.loc[i, "yb1"]),
                      (traces.loc[i, "xb2"], traces.loc[i, "yb2"]), (traces.loc[i, "xb3"], traces.loc[i, "yb3"])))
        contextrelasxedCP.add(((traces.loc[i, "xa1"], traces.loc[i, "ya1"])))
    if traces.loc[i,"action2"] == 6:
        actionForOtheragent.add((traces.loc[i, "xa2"], traces.loc[i, "ya2"]))

print(contextP)
print(contextCP)

problem=boxPushSingle(1,2,2,1,traces)
problem.setContext(contextP,contextrelasxedP,contextCP,contextrelasxedCP)
problem.addOtherAgentsActions(actionForOtheragent,8)
init_state = np.empty(len([1]), dtype=object)
init_state[:]=[[(1,0),(1,0),(1,0),(1,0)]]
actions=np.arange(0,len(problem.actionSpace),1)
obs=np.arange(0,len(problem.observationSpace),1)
traces=[]

ab = POMCP(problem.blackbox, discountfactor=0.95)
ab.initialize(init_state, actions, obs)
real_state = [(1,0),(1,0),(1,0),(1,0)]
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

k=0
for state, pair in policy.items():
    print("belief state ", state, "real state ", pair[1], " action ", problem.actionSpace[pair[0]])
    df.loc[dfcount]=[k,state[0][0],state[0][1],state[1][0],state[1][1],state[2][0],state[2][1],state[3][0],state[3][1],problem.actionSpace[pair[0]]]
    dfcount+=1
    k+=1
traces.append(policy)
df.to_csv("traceSingleAgent1.csv")




