import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import argmax
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

from RKsolver import RKsolver
from numpy.random import binomial, choice, multinomial
import pandas as pd
import random
import itertools
def load_and_preprocess(pickle_path):
    with open(pickle_path, 'rb') as handle:
        agentdict = pickle.load(handle)
    return agentdict


def building_seq_data(agent_dict,len_of_seq=3):
    sequences=[]
    sequences_with_index=[]
    for index,trace in agent_dict.items():
        actions=trace['actions']
        observations=trace['observations']
        time_step=list(range(len(actions)))
        padding=[-1]*(len_of_seq-1)
        padded_actions=padding+actions
        padded_observations=padding+observations
        padded_tmestep=padding+time_step

        for i in range(len_of_seq,len(padded_actions)+1):
            action_seq=padded_actions[i-len_of_seq:i]
            obs_seq=padded_observations[i-len_of_seq:i]
            time_step_seq=padded_tmestep[i-len_of_seq:i]
            indexes=list(range(i-len_of_seq,i))
            obs_action_seq=[]
            timestep_obs_action_seq=[]
            for j in range(len(action_seq)):
                obs_action_seq.append((obs_seq[j],action_seq[j]))
                timestep_obs_action_seq.append((time_step_seq[j], obs_seq[j], action_seq[j]))
            sequences.append(obs_action_seq)
            sequences_with_index.append(timestep_obs_action_seq)
    return sequences,sequences_with_index


def train_split(sequences):
    x_train=[]
    y_train=[]
    for i in range(len(sequences)):
        x_train.append(sequences[i][:-1])
        y_train.append(sequences[i][-1][-1])
    return x_train,y_train

def init_one_hots():
    obs_data = np.array([-1, 0, 1, 2])
    onehot_obs_encoder = OneHotEncoder(sparse=False)
    obs_encoded = obs_data.reshape(len(obs_data), 1)
    obs_encoded = onehot_obs_encoder.fit_transform(obs_encoded)
    obs_to_onehot = {}
    index = -1
    for vec in obs_encoded:
        obs_to_onehot[index] = vec
        index += 1
    actions = np.array([-1,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    onehot_actions_encoder = OneHotEncoder(sparse=False)
    actions_encoded = actions.reshape(len(actions), 1)
    actions_encoded = onehot_actions_encoder.fit_transform(actions_encoded)
    index = -1
    actions_to_onehot={}
    for vec in actions_encoded:
        actions_to_onehot[index] = vec
        index += 1
    return obs_to_onehot,actions_to_onehot,onehot_actions_encoder


def turn_to_vec(x_pre,y_pre,obs_to_onehot,actions_to_onehot):
    x_train=[]
    y_train=[]
    for i in range(len(x_pre)):
        window_x=x_pre[i]
        curr_vec_window=[]
        curr_y=actions_to_onehot[y_pre[i]]
        for j in range(len(window_x)):
            tmp_obs=obs_to_onehot[window_x[j][0]]
            tmp_action=actions_to_onehot[window_x[j][1]]
            combine=np.concatenate((tmp_obs,tmp_action))
            curr_vec_window.append(combine)
        x_train.append(curr_vec_window)
        y_train.append(curr_y)
    return np.array(x_train),np.array(y_train)

def turn_to_vec_temp(x_pre,y_pre,obs_to_onehot,actions_to_onehot):
    """

    need to delete it, its only to compute the one hot for y, neeeded if using the timestep version
    if using withnot timestep use the other turn to vec func
    """
    x_train=[]
    y_train=[]
    for i in range(len(x_pre)):
        #window_x=x_pre[i]
        #curr_vec_window=[]
        curr_y=actions_to_onehot[y_pre[i]]
        """for j in range(len(window_x)):
            tmp_obs=obs_to_onehot[window_x[j][0]]
            tmp_action=actions_to_onehot[window_x[j][1]]
            combine=np.concatenate((tmp_obs,tmp_action))
            curr_vec_window.append(combine)
        x_train.append(curr_vec_window)"""
        y_train.append(curr_y)
    return np.array(x_train),np.array(y_train)

def turn_window(window_x,obs_to_onehot,actions_to_onehot):
    x_train = []
    curr_vec_window = []
    for j in range(len(window_x)):
        tmp_obs = obs_to_onehot[window_x[j][0]]
        tmp_action = actions_to_onehot[window_x[j][1]]
        combine = np.concatenate((tmp_obs, tmp_action))
        curr_vec_window.append(combine)
    x_train.append(curr_vec_window)
    return np.array(x_train)


def buildmoel():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(None, 20)))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(16, activation='softmax'))
    # compiling the network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def buildmoelwithEmbbeding(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size,3,input_length=3))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(16, activation='softmax'))
    # compiling the network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model,x_train,y_train,epochs_num=100):
    history=model.fit(x_train, y_train, epochs=epochs_num, verbose=1)
    return history

def predict(model,x_test):
    preds = model.predict(x_test)
    preds_classes = np.argmax(preds, axis=-1)
    return preds_classes

def prepare_data_for_train(agent_dict,window_size=3,obs_to_onehot=None, actions_to_onehot=None):
    sequences, _ = building_seq_data(agent_dict, window_size)
    x_train_beforevec, y_train_beforevec = train_split(sequences)
    x_train, y_train = turn_to_vec(x_train_beforevec, y_train_beforevec,obs_to_onehot, actions_to_onehot)
    return x_train,y_train

def prepare_data_for_train_forEMb(agent_dict,window_size=3,obs_to_onehot=None, actions_to_onehot=None):
    #sequences, _ = building_seq_data(agent_dict, window_size)     THIS IS NORMAL SEQUENCE THE LINE AFTER IS WITH TIMESTEP
    _, sequences = building_seq_data(agent_dict, window_size)
    x_train_beforevec, y_train_beforevec = train_split(sequences)
    a1_tokenizer=dict()
    count=0
    x_train=[]
    _, y_train = turn_to_vec_temp(x_train_beforevec, y_train_beforevec, obs_to_onehot, actions_to_onehot)
    count1=1
    a1_tokenizer[(-1,-1,-1)]=0
    for i in range(25):
        for j in range(3):
            for k in range(15):
                a1_tokenizer[(i,j,k)]=count1
                count1+=1
    for i in range(len(x_train_beforevec)):
        templst=[]
        for tup in x_train_beforevec[i]:
            if tup not in a1_tokenizer:
                a1_tokenizer[tup]=count
                count+=1
            templst.append(a1_tokenizer[tup])
        x_train.append(np.array(templst))
    return np.array(x_train),y_train,a1_tokenizer




def sim(path1,path2):
    solver = RKsolver('boxMove')
    obs_to_onehot, actions_to_onehot,action_encoder= init_one_hots()
    agent1 = load_and_preprocess(path1)
    #a1_x_train, a1_y_train = prepare_data_for_train(agent1, 4,obs_to_onehot, actions_to_onehot)
    a1_x_train, a1_y_train,a1_tokenizer = prepare_data_for_train_forEMb(agent1, 4,obs_to_onehot, actions_to_onehot)
    #a1_model = buildmoel()
    a1_model = buildmoelwithEmbbeding(len(a1_tokenizer))
    h1=train_model(a1_model, a1_x_train, a1_y_train, 4)

    agent2 = load_and_preprocess(path2)
    #a2_x_train, a2_y_train = prepare_data_for_train(agent2, 4,obs_to_onehot, actions_to_onehot)
    a2_x_train, a2_y_train, a2_tokenizer = prepare_data_for_train_forEMb(agent2, 4, obs_to_onehot, actions_to_onehot)
    #a2_model = buildmoel()
    a2_model = buildmoelwithEmbbeding(len(a2_tokenizer))
    h2=train_model(a2_model, a2_x_train, a2_y_train, 4)
    #solutions=simulate(solver,10,a1_model,a2_model,obs_to_onehot,actions_to_onehot,action_encoder,a1_tokenizer,a2_tokenizer)
    solutions = simulate_v1(solver, 10, a1_model, a2_model, obs_to_onehot, actions_to_onehot, action_encoder, a1_tokenizer,
                         a2_tokenizer)
    return solutions

def NextWindow(curr_window,action,obs):
    next_window=curr_window[1:].copy()
    next_window.append((obs,action))
    return next_window

def NextWindow_with_time(curr_window,action,obs,time):
    next_window=curr_window[1:].copy()
    next_window.append((time,obs,action))
    return next_window

def simulate(solver,num_of_traces,a1_model,a2_model,obs_to_onehot,actions_to_onehot,action_encoder,a1_tokenizer,a2_tokenizer):
    print("****** start simulation**********")
    traces=[]
    solver.problem_adapter.sep_rewards=True
    for i in range(0, num_of_traces):
        print(f"start trace {i}")
        trace = {'actions': [], 'states': [], 'rewards': [], 'observations': [], 'next_states': [],
                 'trace_len': 0,'visual_state':[],'action_1':[],'action_2':[],'obs_1':[],'obs_2':[],'end_cause':'succ',
                 'visual_next_state':[],
                 'total_cost': 0, 'total_reward': 0,'total_costs':[0,0],'total_rewards':[0,0]}
        state = choice(solver.pomcp_solver.initStates)
        sliding_win_before_vec_agent1=[(-1,-1),(-1,-1),(-1,-1)]
        sliding_win_before_vec_agent2 = [(-1, -1), (-1, -1), (-1, -1)]
        window_agent1=turn_window(sliding_win_before_vec_agent1,obs_to_onehot,actions_to_onehot)
        window_agent2 = turn_window(sliding_win_before_vec_agent2, obs_to_onehot, actions_to_onehot)
        flag = False
        count=0
        while (not flag):
            count+=1
            print(f"state is {solver.problem_adapter.numberToState(state)}")
            #need to check if its not a problem the network can produce the action -1 which is the network nope action that
            #doesnt exist in the problem action space, maybe later add if action ==-1 change to 0
            a1_action_pred=np.argmax(a1_model.predict(window_agent1),axis=-1)[0]-1
            a2_action_pred=np.argmax(a2_model.predict(window_agent2),axis=-1)[0]-1
            print(f"action suggested {(a1_action_pred,a2_action_pred)}")
            if a1_action_pred!=0 and a2_action_pred!=0 and a1_action_pred!=a2_action_pred:
                action_number=(a1_action_pred,a2_action_pred)
            else:
                action_number = solver.problem_adapter.actiontoNumber((a1_action_pred, a2_action_pred))
            next_state, observation, reward = solver.problem_adapter.blackbox(state, action_number)
            print("next state: ", solver.problem_adapter.numberToState(next_state))
            obs=solver.problem_adapter.numbertoObservation(observation)
            print("observation recievd : ", obs)
            print("reward : ", reward)
            trace['actions'].append(action_number)
            trace['visual_state'].append(solver.problem_adapter.numberToState(state))
            trace['action_1'].append(a1_action_pred)
            trace['action_2'].append(a2_action_pred)
            trace['obs_1'].append(obs[0])
            trace['obs_2'].append(obs[1])
            trace['states'].append(state)
            trace['next_states'].append(next_state)
            trace['visual_next_state'].append(solver.problem_adapter.numberToState(next_state))
            trace['observations'].append(observation)
            trace['rewards'].append(reward)
            for j in range(0, len(reward)):
                if reward[j] > 0:
                    trace['total_rewards'][j] += reward[j]
                    trace['total_reward'] += reward[j]
                else:
                    trace['total_costs'][j] += reward[j]
                    trace['total_cost'] += reward[j]
            if solver.problem_instance.checkGoal(solver.problem_adapter.numberToState(next_state)) or count>20:
                flag = True
                trace['trace_len'] = len(trace['actions'])
                if count>20 and not solver.problem_instance.checkGoal(solver.problem_adapter.numberToState(next_state)):
                    trace['end_cause']='fail'
                traces.append(trace)
            else:
                state = next_state
                sliding_win_before_vec_agent1=NextWindow(sliding_win_before_vec_agent1,a1_action_pred+1,obs[0])
                window_agent1 = turn_window(sliding_win_before_vec_agent1, obs_to_onehot, actions_to_onehot)
                sliding_win_before_vec_agent2 = NextWindow(sliding_win_before_vec_agent2,a2_action_pred+1,obs[1])
                window_agent2 = turn_window(sliding_win_before_vec_agent2, obs_to_onehot, actions_to_onehot)
    return traces


def tokenize(window,agent_tokenizer):
    converted_window=[]
    for i in range(len(window)):
        converted_window.append(agent_tokenizer[window[i]])
    return np.array([np.array(converted_window)])

def simulate_v1(solver,num_of_traces,a1_model,a2_model,obs_to_onehot,actions_to_onehot,action_encoder,a1_tokenizer,a2_tokenizer):
    """
    same as simulate but using the embeddings
    :param solver:
    :param num_of_traces:
    :param a1_model:
    :param a2_model:
    :param obs_to_onehot:
    :param actions_to_onehot:
    :param action_encoder:
    :param a1_tokenizer:
    :param a2_tokenizer:
    :return:
    """
    print("****** start simulation**********")
    traces=[]
    solver.problem_adapter.sep_rewards=True
    for i in range(0, num_of_traces):
        print(f"start trace {i}")
        trace = {'actions': [], 'states': [], 'rewards': [], 'observations': [], 'next_states': [],
                 'trace_len': 0,'visual_state':[],'action_1':[],'action_2':[],'obs_1':[],'obs_2':[],'end_cause':'succ',
                 'visual_next_state':[],
                 'total_cost': 0, 'total_reward': 0,'total_costs':[0,0],'total_rewards':[0,0]}
        state = choice(solver.pomcp_solver.initStates)
        window_agent1_before_tokenize=[(-1,-1,-1),(-1,-1,-1),(-1,-1,-1)]
        window_agent1=tokenize(window_agent1_before_tokenize,a1_tokenizer)
        window_agent2_before_tokenize = [(-1,-1,-1),(-1,-1,-1),(-1,-1,-1)]
        window_agent2=tokenize(window_agent2_before_tokenize,a2_tokenizer)
        flag = False
        count=0
        while (not flag):
            count+=1
            print(f"state is {solver.problem_adapter.numberToState(state)}")
            #need to check if its not a problem the network can produce the action -1 which is the network nope action that
            #doesnt exist in the problem action space, maybe later add if action ==-1 change to 0
            a1_action_pred=np.argmax(a1_model.predict(window_agent1),axis=-1)[0]-1
            a2_action_pred=np.argmax(a2_model.predict(window_agent2),axis=-1)[0]-1
            print(f"action suggested {(a1_action_pred,a2_action_pred)}")
            if a1_action_pred!=0 and a2_action_pred!=0 and a1_action_pred!=a2_action_pred:
                action_number=(a1_action_pred,a2_action_pred)
            else:
                action_number = solver.problem_adapter.actiontoNumber((a1_action_pred, a2_action_pred))
            next_state, observation, reward = solver.problem_adapter.blackbox(state, action_number)
            print("next state: ", solver.problem_adapter.numberToState(next_state))
            obs=solver.problem_adapter.numbertoObservation(observation)
            print("observation recievd : ", obs)
            print("reward : ", reward)
            trace['actions'].append(action_number)
            trace['visual_state'].append(solver.problem_adapter.numberToState(state))
            trace['action_1'].append(a1_action_pred)
            trace['action_2'].append(a2_action_pred)
            trace['obs_1'].append(obs[0])
            trace['obs_2'].append(obs[1])
            trace['states'].append(state)
            trace['next_states'].append(next_state)
            trace['visual_next_state'].append(solver.problem_adapter.numberToState(next_state))
            trace['observations'].append(observation)
            trace['rewards'].append(reward)
            for j in range(0, len(reward)):
                if reward[j] > 0:
                    trace['total_rewards'][j] += reward[j]
                    trace['total_reward'] += reward[j]
                else:
                    trace['total_costs'][j] += reward[j]
                    trace['total_cost'] += reward[j]
            if solver.problem_instance.checkGoal(solver.problem_adapter.numberToState(next_state)) or count>20:
                flag = True
                trace['trace_len'] = len(trace['actions'])
                if count>20 and not solver.problem_instance.checkGoal(solver.problem_adapter.numberToState(next_state)):
                    trace['end_cause']='fail'
                traces.append(trace)
            else:
                state = next_state
                window_agent1_before_tokenize=NextWindow_with_time(window_agent1_before_tokenize,a1_action_pred,obs[0],count-1)
                window_agent1 = tokenize(window_agent1_before_tokenize, a1_tokenizer)
                window_agent2_before_tokenize = NextWindow_with_time(window_agent2_before_tokenize,a2_action_pred,obs[1],count-1)
                window_agent2 = tokenize(window_agent2_before_tokenize, a2_tokenizer)
    return traces


def sane_check():
    obs_to_onehot, actions_to_onehot, action_encoder = init_one_hots()
    agent1 = load_and_preprocess('tracesBPagent1_2.pickle')
    a1_x_train, a1_y_train = prepare_data_for_train(agent1, 4,obs_to_onehot, actions_to_onehot)
    print(a1_x_train.shape)
    print(a1_y_train.shape)
    a1_model = buildmoel()
    train_model(a1_model, a1_x_train, a1_y_train, 100)
    a1_preds = predict(a1_model, a1_x_train[:21])
    print(a1_preds)

    agent2 = load_and_preprocess('tracesBPagent2_2.pickle')
    a2_x_train, a2_y_train = prepare_data_for_train(agent2, 4,obs_to_onehot, actions_to_onehot)
    print(a2_x_train.shape)
    print(a2_y_train.shape)
    a2_model = buildmoel()
    train_model(a2_model, a2_x_train, a2_y_train, 100)
    a2_preds = predict(a2_model, a2_x_train[:21])
    print(a2_preds)

def checktrace(path):
    teampolicy=load_and_preprocess(path)
    a1=load_and_preprocess('tracesBP1601agent1.pickle')
    a2=load_and_preprocess('tracesBP1601agent2.pickle')
    a3=load_and_preprocess('tracesBP1301agent1.pickle')
    solver = RKsolver('boxMove')
    print(teampolicy)

def infusetraces(paths,id):
    infused_agent1={}
    infused_agent2 = {}
    start=0
    for path in paths:
        a1=load_and_preprocess(path[0])
        a2=load_and_preprocess(path[1])
        for i,trace in a1.items():
            infused_agent1[start+i]=trace
        for i,trace in a2.items():
            infused_agent2[start+i]=trace
        start+=len(a1)


    with open(f'infusedBP{id}_agent1.pickle', 'wb') as handle:
        pickle.dump(infused_agent1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'infusedBP{id}_agent2.pickle', 'wb') as handle:
        pickle.dump(infused_agent2, handle, protocol=pickle.HIGHEST_PROTOCOL)

ps=[('traces/tracesBPagent1_2.pickle','traces/tracesBPagent2_2.pickle'),('tracesBP1301agent1.pickle','tracesBP1301agent2.pickle'),('tracesBP1601agent1.pickle','tracesBP1601agent2.pickle')]
if __name__ == '__main__':
    #checktrace('tracesBP1301.pickle')
    #infusetraces(ps,2)
    path1='tracesBP2603agent1.pickle'
    path2='tracesBP2603agent2.pickle'
    sol_traces=sim(path1,path2)
    i=0
    for tr in sol_traces:
        i+=1
        print(f'trace {i}')
        print(tr['end_cause'])
        print(tr['visual_state'])
        print(tr['action_1'])
        print(tr['action_2'])
        print(tr['obs_1'])
        print(tr['obs_2'])
        print('')
























