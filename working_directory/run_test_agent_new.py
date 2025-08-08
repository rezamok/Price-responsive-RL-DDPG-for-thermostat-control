# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:18:14 2021

@author: kosta
"""

import sys
import os
# import sys

# Append the path to the sys.path list
os.chdir('C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL')

import json
sys.path.insert(0,'..')
import numpy as np
# Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)
# imports
from kosta.environment import get_env
from kosta.one_for_many import test_agent
from kosta.reward import compute_reward
from kosta.environment import agent_kwargs,model_kwargs,data_kwargs
from agents.agents import RBAgent, BangBangAgent,RBAgent_old_three, UnavoidableAgent
# change working directory
if sys.platform == "win32":
    os.chdir(os.path.abspath(os.path.join(os.path.dirname("__file__"),'..')))
else:
    sys.path.append("../")

import kosta.hyperparams as hp
import numpy as np
from statistics import mean
from kosta.checkpoint_utils import Alg1Logger,plot_test_data,load_from_disk,Alg2Logger
from kosta.model_utils import get_network2, get_ddpg_agent_and_rb

NUM_EPISODES = 1 # We will agent for this number of episodes
#NUM_EPISODES = 1 # We will agent for this number of episodes

# EPISODE_CHECKPOINT = 1700 # FROM alg1 directory
EPISODE_CHECKPOINT = 120# FROM alg1 directory
# sequences_and_init_temps_path = 'C:\\Users\\kosta\\Desktop\\master-thesis\\RL\\DRL\\kosta\\checkpoints\\delta_t_norm7_Vrewardshape_1day\\alg1\\new3_tests_chkpt_1700_episodes_10_days_3\\sequences_and_init_temps.txt'# None if you want random sequences
sequences_and_init_temps_path = None
if __name__ == '__main__':
    env,data_kwargs,model_kwargs,agent_kwargs= get_env(env_model=hp.env_model, set_room="272")
    network2 = get_network2()
    network2 = network2.to(hp.device['name'])
    network2.eval()
    ddpg_agent,_ = get_ddpg_agent_and_rb(env) 
    logger = Alg1Logger(hp.checkpoint_dir,False)
    if sequences_and_init_temps_path is not None:
        # sequences,init_temps = logger.load_sequences_and_init_temps(sequences_and_init_temps_path)
        sequences = np.array([[1, 25000]])
        init_temps = np.array([[23]])
        print("test_Seq")
        print(sequences)
        print(init_temps)
        # assert (len(sequences) == NUM_EPISODES)
#        delete_ids=[0,1,2,4,5,7,9,10,17,19]
#        for i in sorted(delete_ids,reverse=True):
#            del sequences[i]
#            del init_temps[i]
#        NUM_EPISODES-=len(delete_ids)
    else:
        sequences,init_temps = None,None
        sequences = np.array([[8000, 9500]])
        init_temps = np.array([[23]])
    rb_agent = UnavoidableAgent(data_kwargs,model_kwargs,agent_kwargs,compute_reward=compute_reward)
    
    sequences = np.array([[6200, 6800]])
    #sequences = np.array([[4000, 7000]])

    init_temps = np.array([[23]])
#==============================================================================
    test_dir = os.path.join(
        logger.save_dir,
        f'prez2_tests_chkpt_{EPISODE_CHECKPOINT}_episodes_{NUM_EPISODES}_days_{int(hp.threshold_length/96)}'
    )
    print('NON-RANDOM agent!')
    # load weights
    network2,ddpg_agent.q_function,ddpg_agent.policy,_,_,_ = logger.load_weights(
        EPISODE_CHECKPOINT,
        network2,
        ddpg_agent.q_function,
        ddpg_agent.policy
    )
    # UNCOMMENT NEXT LINE ONLY IF YOU KNOW WHAT YOU ARE DOING (it will load network2 from alg2)
    # network2,_ = Alg2Logger(hp.checkpoint_dir,False).load_weights(hp.device['name'],network2)
    sequences,init_temps,stats_ddpg,stats_rb = test_agent(
        env=env,
        network2=network2,
        ddpg_agent=ddpg_agent,
        rb_agent=rb_agent,
        test_dir=test_dir,
        num_episodes=NUM_EPISODES,
        sequences=sequences,
        init_temps=init_temps
    )
    logger.save_sequences_and_init_temps(
        sequences,
        init_temps,
        save_path=os.path.join(test_dir,'sequences_and_init_temps.txt')
    )
    

    

