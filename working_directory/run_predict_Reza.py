"""
Created on Mon November  19, 2024

@author: Reza
"""

import sys
import os

# Append the path to the sys.path list
os.chdir('C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL')
import pandas as pd
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
from kosta.one_for_many2 import predict_agent

EPISODE_CHECKPOINT = 460# FROM alg1 directory

# Call the trained RL environment
env,data_kwargs,model_kwargs,agent_kwargs= get_env(env_model=hp.env_model, set_room="272")
network2 = get_network2()
network2 = network2.to(hp.device['name'])
network2.eval()
ddpg_agent,_ = get_ddpg_agent_and_rb(env) 
logger = Alg1Logger(hp.checkpoint_dir,False)
sequences,init_temps = None,None
#==============================================================================
predict_path = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/Predicts'





# load weights
network2,ddpg_agent.q_function,ddpg_agent.policy,_,_,_ = logger.load_weights(
    EPISODE_CHECKPOINT,
    network2,
    ddpg_agent.q_function,
    ddpg_agent.policy
)
obs_path = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/saves/Data_predict'

# Load PCNN model
import os
os.chdir(r'C:\Users\remok\OneDrive - Danmarks Tekniske Universitet\Skrivebord\RL TRV\Models\pcnnNew - complete\pcnnNew - complete\pcnn')

import numpy as np
import torch
from pcnn.parameters import parameters
from pcnn.util import load_data
from pcnn.model import Model
import matplotlib.pyplot as plt

model_kwargs = parameters(unit='UMAR',
                            to_normalize=True,
                            name="PCNN1",
                            heating=1,
                            cooling=0,
                            seed=0,
                            overlapping_distance=4,
                            warm_start_length=12,
                            maximum_sequence_length=96*3,
                            minimum_sequence_length=48,
                            learn_initial_hidden_states=True,
                            decrease_learning_rate=True,
                            validation_percentage=15,
                            test_percentage=15,
                            learning_rate=0.0005,
                            feed_input_through_nn=True,
                            input_nn_hidden_sizes=[32],
                            lstm_hidden_size=64,
                            lstm_num_layers=2,
                            layer_norm=True,
                            batch_size=256,
                            output_nn_hidden_sizes=[32],
                            division_factor=10.,
                            verbose=0)

module = 'PCNN'


X_columns = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
             'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 272', 'Case']
Y_columns = ['Temperature 272']

# Corresponding columns
case_column = -1
out_column = 1
neigh_column = 3
temperature_column = 2
power_column = -2

# Info to use in D
inputs_D = [0, 4, 5, 6, 7, 8]

topology = None # Not needed in the single-zone case

# Trying to load a model or not, if yes the last one or the best one
load = True
load_last = False

data = load_data('Book2')

pcnn = Model(data=data, interval=15, model_kwargs=model_kwargs, inputs_D=inputs_D,
            module=module, rooms='272', case_column=case_column, out_column=out_column, neigh_column=neigh_column,
            temperature_column=temperature_column, power_column=power_column,
            Y_columns=Y_columns, X_columns=X_columns, topology=topology, load_last=load_last,
            load=load)


actions, temperatures, demand = predict_agent(env,network2,pcnn,X_columns,ddpg_agent,test_dir = predict_path, obs_path=obs_path)

# Calculate hourly demand
demand_h = [sum(demand[i:i+4]) for i in range(0, len(demand), 4)]
temperatures_h = [np.average(temperatures[i:i+4]) for i in range(0, len(temperatures), 4)]

# Save the baseline demand and temperatures
save_path = r"C:\Users\remok\OneDrive - Danmarks Tekniske Universitet\Skrivebord\RL TRV\Models\Price-Responsive RL\saves\Data_predict\Y.csv"
save_path_temp = r"C:\Users\remok\OneDrive - Danmarks Tekniske Universitet\Skrivebord\RL TRV\Models\Price-Responsive RL\saves\Data_predict\Temperatures_Y.csv"
demand_h = pd.DataFrame(demand_h, columns=["Values"])
demand_h.to_csv(save_path, index=False, header=False)
temperatures_h = pd.DataFrame(temperatures_h, columns=["Values"])
temperatures_h.to_csv(save_path_temp, index=False, header=False)


# Visualizatoin
plt.plot(temperatures)
plt.figure()
plt.plot(actions)


    

