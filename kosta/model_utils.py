# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:02:17 2021

@author: kosta
"""

import kosta.hyperparams as hp
import torch
import torch.nn as nn
import numpy as np
from pfrl.pfrl.agents import DDPG
from pfrl.pfrl.nn import MLP,ConcatObsAndAction
from pfrl.pfrl.replay_buffers import ReplayBuffer
from pfrl.pfrl.policies import DeterministicHead
from pfrl.pfrl.replay_buffer import batch_experiences

class Storage:
    def __init__(self):
        self.states = []
        self.delta_ts = []
        self.actions = []
    def add_sample(self,state,delta_t,action):
        self.states.append(state)
        self.delta_ts.append(delta_t)
        self.actions.append(action)
    def sample(self,i):
        return self.states[i],self.delta_ts[i],self.actions[i]
    def get_size(self):
        return len(self.actions)
    
def collect_data(data_size,env):
    storage = Storage()
    is_full = False
    while not is_full:
        done = False
        s_curr = env.reset()
        while not done:
            action = np.random.uniform(0,1.00001)
            action = 1 if action > 1 else action
            action = 0 if action < 0 else action
            s_next,reward,done,_, price_reward, comfort_reward = env.step(action) 
            # delta_t = s_next[10] - s_curr[10]
            # delta_t = np.clip(delta_t,-hp.max_delta_t,hp.max_delta_t)/hp.max_delta_t
            delta_t = action
            storage.add_sample(s_curr,delta_t,action)
            s_curr = s_next
            if storage.get_size() == data_size:
                is_full=True
                break
            if storage.get_size() % 1000 == 0:
                print(f'\r{storage.get_size()}/{data_size}',end='')
    print('')
    return storage

class ScaleOut(nn.Module):
    def __init__(self,scaling):
        super().__init__()
        self.scaling=scaling
    def forward(self,X):
        return self.scaling*X

# +
def init_network2(network2):
    network2[0] = init_pfrl_MLP(network2[0])
    return network2
def init_policy(policy):
    policy[0] = init_pfrl_MLP(policy[0])
    return policy
def init_q_func(q_func):
    q_func[1] = init_pfrl_MLP(q_func[1])
    return q_func

def init_pfrl_MLP(mlp):
    for i in range(len(mlp.hidden_layers)):
        nn.init.kaiming_uniform_(mlp.hidden_layers[i].weight, mode='fan_in', nonlinearity='relu')
        # bias is already set to 0 in pfrl.nn.MLP
    nn.init.kaiming_uniform_(mlp.output.weight, mode='fan_in', nonlinearity='relu')
    return mlp
def get_network2():
    """
    Create network2 with hyperparameters from hyperparams.py
    """
    network2 = nn.Sequential(
        MLP(
            in_size=hp.net2_in_size,
            out_size=hp.net2_out_size,
            hidden_sizes=hp.net2_hidden_sizes,
            nonlinearity=hp.net2_nonlinearity
        ),
        nn.Sigmoid()
    )
    print('Network 2 is created with parameters from hyperparams.py')
    if hp.use_he_normalization:
        network2 = init_network2(network2)
        print('Network 2 is initialized with unfiorm HE initialization')
    return network2
def get_ddpg_agent_and_rb(env=None):
    """
    Create ddpg agent replay buffer with hyperparams from hyperparameters.py
    """
    # initialize replay buffer and DDPG agent
    replay_buffer = ReplayBuffer(
        capacity=hp.rb_capacity,
        num_steps=hp.rb_num_steps
    )
    # define policy and q function
    print('Max delta_t is:',hp.max_delta_t)
    policy = nn.Sequential(
        MLP(
            in_size=hp.pol_in_size,
            out_size=hp.pol_out_size,
            hidden_sizes=hp.pol_hidden_sizes,
            nonlinearity=hp.pol_nonlinearity
        ),
        nn.Tanh(),
#        ScaleOut(hp.max_delta_t),
        DeterministicHead()
    )
    if hp.use_he_normalization:
        policy = init_policy(policy)
    q_func = nn.Sequential(
        ConcatObsAndAction(),
        MLP(
            in_size=hp.q_in_size,
            out_size=hp.q_out_size,
            hidden_sizes=hp.q_hidden_sizes,
            nonlinearity=hp.q_nonlinearity
        )
    )
    if hp.use_he_normalization:
        q_func = init_q_func(q_func)
    if hp.apply_burning_func:
        #burning_func = lambda:np.random.normal(loc=0, scale=hp.max_delta_t) 
        #burning_func = lambda:np.random.uniform(-hp.max_delta_t, hp.max_delta_t) 
        if env is None:
            raise ValueError('You should give env to this functions')
        print('Collecting data for burning function..')
        storage = collect_data(hp.rb_init_capacity,env)
        print('COLECTED')
        def burning_func():
            delta_t = np.random.choice(storage.delta_ts)
            return delta_t
    else:
        burning_func = None
    # define DDPG agent
    ddpg_agent = DDPG(
        policy=policy,
        q_func=q_func,
        actor_optimizer=hp.pol_optim(policy.parameters()),
        critic_optimizer=hp.q_optim(q_func.parameters()),
        gamma=hp.gamma,
        replay_buffer=replay_buffer,
        explorer=hp.explorer,
        gpu=hp.device['id'],
        burnin_action_func = burning_func
    )
    print('DDPG agent is created with parameters from hyperparams.py')
    if hp.use_he_normalization:
        print('Policy and q func are initilized with uniform HE initialization')
    return ddpg_agent, replay_buffer
