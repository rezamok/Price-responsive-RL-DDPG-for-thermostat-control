# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:18:14 2021

@author: kosta
"""

import sys
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
from agents.agents import RBAgent
# change working directory
if sys.platform == "win32":
    os.chdir(os.path.abspath(os.path.join(os.path.dirname("__file__"),'..')))
else:
    sys.path.append("../")

import kosta.hyperparams as hp
import numpy as np
from statistics import mean
from kosta.checkpoint_utils import Alg1Logger,plot_test_data
from kosta.model_utils import get_network2, get_ddpg_agent_and_rb

NUM_EPISODES = 100 # We will agent for this number of episodes
EPISODE_CHECKPOINT = 600 # FROM alg1 directory
PRINT_PER_EPISODE=False

if __name__ == '__main__':
    env = get_env(env_model=hp.env_model)
    network2 = get_network2()
    network2 = network2.to(hp.device['name'])
    network2.eval()
    ddpg_agent,_ = get_ddpg_agent_and_rb() 
    logger = Alg1Logger(hp.checkpoint_dir,False)
#==============================================================================
# Test NON-RANDOM agent
    test_dir = os.path.join(logger.save_dir,'tests_not_random_agent')
    print('NON-RANDOM agent!')
    # load weights
    network2,ddpg_agent.q_function,ddpg_agent.policy,_,_,_ = logger.load_weights(
        EPISODE_CHECKPOINT,
        network2,
        ddpg_agent.q_function,
        ddpg_agent.policy
    )
    sequences,stats = test_agent(
        env=env,
        network2=network2,
        ddpg_agent=ddpg_agent,
        test_dir=test_dir,
        num_episodes=NUM_EPISODES,
        sequences=None,
        print_per_episode=PRINT_PER_EPISODE
    )
#==============================================================================
    # Test Rule Base agent
    test_dir = os.path.join(logger.save_dir,'tests_rb_agent')
    os.makedirs(test_dir,exist_ok=True)
    rb_agent = RBAgent(data_kwargs,model_kwargs,agent_kwargs,compute_reward=compute_reward)
    episode_rewards = []
    comfort_violations = []
    prices = []
    for sequence in sequences:
        reward, _ = rb_agent.run(sequence,render=False)
        episode_rewards.append(reward)
        comfort_violations.append(rb_agent.env.comfort_violations)
        prices.append(rb_agent.env.prices)
    avg_comf_viol = mean([mean(x) for x in comfort_violations])
    max_comf_viol = max([max(list(map(abs, x))) for x in comfort_violations])
    avg_price = mean([mean(x) for x in prices])
    max_price = max([max(list(map(abs, x))) for x in prices])
    avg_reward = mean(episode_rewards)
    stats_rb = {
        'avg_reward': avg_reward,
        'avg_comf_viol': avg_comf_viol,
        'max_comf_viol': max_comf_viol,
        'avg_price': avg_price,
        'max_price': max_price
    }
    with open(os.path.join(test_dir,'stats.json'),'w') as fp:
        json.dump(stats_rb,fp)
    print('---'*5)
    print('Average reward is: {:.4f}'.format(avg_reward))
    print('Average comfort violation is: {:.4f}'.format(avg_comf_viol))
    print('Maximal absolute comfort violation is: {:.4f}'.format(max_comf_viol))
    print('Average price is: {:.4f}'.format(avg_price))
    print('Maximal absolute price is: {:.4f}'.format(max_price))
    print('---'*5)
#==============================================================================
# Print results  
    print('Trained agent compared to Rule Based agent:')
    print('Percentage of average comfort violation difference: {:.3f}%'.format(
          (avg_comf_viol-stats['avg_comf_viol'])/avg_comf_viol*100
    ))
    print('Percentage of average price difference: {:.3f}%'.format(
        (avg_price-stats['avg_price'])/avg_price*100
    ))
    print('Difference between maximal comfort violations: {:.3f} degrees'.format(
        max_comf_viol - stats['max_comf_viol']
    ))
    print('Difference between maximal prices: {:.3f}'.format(
         max_price -  stats['max_price']
    ))

    

    

