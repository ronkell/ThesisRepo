from gridBoxlinear import gridBoxlinear
from RKsolver import RKsolver
import pickle

solver=RKsolver('boxMove')

def gen_multitrace(sim_id,numberoftraces=100):
    full_traces=[]
    mysolv=RKsolver('boxMove')
    mysolv.genrateTracesbyHand(numberoftraces)
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
    sim_id = input('enter simulation id')
    gen_multitrace(sim_id,10000)
    disterbute_traces(sim_id)