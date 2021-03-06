from gridBoxlinear import gridBoxlinear
from RKsolver import RKsolver
import pickle
#prob=gridBoxlinear(2, 2, 2, 1, 1, [(0, 1), (1, 0)], [(0, 0), (0, 0)])
#prob=gridBoxlinear(2,2,2,0,2,[(0,0),(1,1)],[(0,1),(1,0)])
solver=RKsolver('boxMove')
""" just process pickle need to remove the comment because this code is good

solver.build_tree()
solver.genrateTraces(100)
traces=solver.traces
with open('tracesBP.pickle', 'wb') as handle:
    pickle.dump(traces, handle, protocol=pickle.HIGHEST_PROTOCOL)"""


def gen_traces(sim_id):
    solver.build_tree()
    solver.genrateTraces(100)
    traces = solver.traces
    with open(f'tracesBP{sim_id}.pickle', 'wb') as handle:
        pickle.dump(traces, handle, protocol=pickle.HIGHEST_PROTOCOL)

def gen_multitrace(sim_id,numberofmodels):
    full_traces=[]
    for i in range(numberofmodels):
        mysolv=RKsolver('boxMove')
        try:
            mysolv.build_tree()
        except Exception as e:
            print(e)
            continue
        mysolv.genrateTraces(100)
        full_traces.extend(mysolv.traces)
    with open(f'tracesBP{sim_id}.pickle', 'wb') as handle:
        pickle.dump(full_traces, handle, protocol=pickle.HIGHEST_PROTOCOL)
def disterbute_traces(sim_id):
    agent1dict = {}
    agent2dict = {}
    count = 0;
    with open(f'tracesBP{sim_id}.pickle', 'rb') as handle:
        mydict = pickle.load(handle)
        for trace in mydict:
            temptrace1 = {'actions': [], 'observations': []}
            temptrace2 = {'actions': [], 'observations': []}
            for i in range(len(trace['actions'])):
                action = solver.problem_adapter.numbertoAction(trace['actions'][i])
                temptrace1['actions'].append(action[0])
                temptrace2['actions'].append(action[1])
                obs = solver.problem_adapter.numbertoObservation(trace['observations'][i])
                temptrace1['observations'].append(obs[0])
                temptrace2['observations'].append(obs[1])
            agent1dict[count] = temptrace1
            agent2dict[count] = temptrace2
            count += 1


    with open(f'tracesBP{sim_id}agent1.pickle', 'wb') as handle:
        pickle.dump(agent1dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'tracesBP{sim_id}agent2.pickle', 'wb') as handle:
        pickle.dump(agent2dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
    sim_id=input('enter simulation id')
    #gen_traces(sim_id)
    gen_multitrace(sim_id,15)
    disterbute_traces(sim_id)





