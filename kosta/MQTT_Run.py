import os
os.chdir('C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL')

import json
import sys
sys.path.insert(0,'..')
import kosta.hyperparams as hp
import torch
from kosta.environment import get_env
import pandas as pd
import numpy as np
from opcua_empa.live_control_Mina import Controller
from kosta.environment import agent_kwargs,model_kwargs,data_kwargs
from kosta.reward import compute_reward
import datetime as dt
from dotenv import load_dotenv
from kosta.model_utils import get_network2, get_ddpg_agent_and_rb
from kosta.checkpoint_utils import Alg1Logger,plot_test_data,load_from_disk,Alg2Logger

# Specify the directories
checkpoint_dir = "C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/checkpoints"
save_path = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/Experiments'
load_episode = 460

class RLAgent:
    def __init__(self,name,env,compute_reward,checkpoint_dir,load_episode,transfer):
        self.env=env
        self.compute_reward = compute_reward
        self.name=name
        # load from checkpoint
        network2 = get_network2()
        network2 = network2.to(hp.device['name'])
        network2.eval()
        ddpg_agent,_ = get_ddpg_agent_and_rb(env) 
        logger = Alg1Logger(checkpoint_dir,False)
        # load weights
        network2,ddpg_agent.q_function,ddpg_agent.policy,_,_,_ = logger.load_weights(
            load_episode,
            network2,
            ddpg_agent.q_function,
            ddpg_agent.policy
        )
        if transfer:
            data_type = '_real_data' if hp.use_historical_data else '_env_data'
            logger2 = Alg2Logger(os.path.join(checkpoint_dir,'room_'+hp.transfer_room[0]+data_type), False)
            network2,_ = logger2.load_weights(hp.device['name'],network2)
        self.network2 = network2
        self.ddpg_agent = ddpg_agent
    def predict(self,observation):
        observation = observation.astype(np.float32)
        # observation = [np.float32(val) for val in observation]
        with self.ddpg_agent.eval_mode():
            delta_t = torch.tensor(self.ddpg_agent.batch_act([observation])).reshape((1,))
            print("delta_t")
            print(delta_t)
        with torch.no_grad():
            delta_t = delta_t.to(torch.float32)
            observation = torch.tensor(observation)
            action = delta_t
        return action, delta_t
    
# Initialize the controller
env, _, _, _= get_env(env_model=hp.env_model,set_room=hp.main_room)

agent = RLAgent(
     name='DDPG',
     env=env,
     compute_reward=compute_reward,
     checkpoint_dir=checkpoint_dir,
     load_episode=load_episode,
     transfer=False
 )


# Functions
def get_current_hour_price(filename="Prices.csv"):
    # Read CSV file
    df = pd.read_csv(filename, header=None)  # Assuming no headers in the file
    
    # Get current hour (24-hour format)
    current_hour = dt.datetime.now().hour  # e.g., 17 for 5 PM
    
    # Ensure the file has exactly 24 rows
    if len(df) != 24:
        raise ValueError("Prices.csv should have exactly 24 rows.")
    
    # Get the value from the current hour's row
    price_value = df.iloc[current_hour, 0]  # Assuming values are in the first column
    
    return price_value

def p2t_sp_hvac(sp_power, sc_=False, sh_=False):
    """
    Convert power set point to temperature set point
    :param sp_power: Power set point in percentage (0-100%)
    :param sc_: space cooling?
    :param sh_: space heating?
    :return: temperature set point
    """
    now = dt.datetime.now()
    quo_quarter = now.minute // 15  # which quarter it is right now
    if sh_:
        # Heating case
        if quo_quarter == 0 or quo_quarter == 2:
            if ((now.minute * 60 + now.second) % (15 * 60)) / (15 * 60) * 100 <= np.abs(sp_power):
                sp_temp = 28
            else:
                sp_temp = 16
        else:
            if ((now.minute * 60 + now.second) % (15 * 60)) / (15 * 60) * 100 <= 100 - np.abs(sp_power):
                sp_temp = 16
            else:
                sp_temp = 28
        # If the duty cycle is too low, then no point to turn on the valve
        if np.abs(sp_power) < 5:
            sp_temp = 16
    elif sc_:
        # Cooling case
        if quo_quarter == 0 or quo_quarter == 2:
            if ((now.minute * 60 + now.second) % (15 * 60)) / (15 * 60) * 100 <= np.abs(sp_power):
                sp_temp = 16
            else:
                sp_temp = 28
        else:
            if ((now.minute * 60 + now.second) % (15 * 60)) / (15 * 60) * 100 <= 100 - np.abs(sp_power):
                sp_temp = 28
            else:
                sp_temp = 16
        # If the duty cycle is too low, then no point to turn on the valve
        if np.abs(sp_power) < 5:
            sp_temp = 28
    else:
        sp_temp = 22

    return sp_temp


# Get controller parameters
if __name__ == '__main__':
    env, _, _, _= get_env(env_model=hp.env_model,set_room=hp.main_room)
    
# Set up the controller
controller = Controller(data_kwargs,model_kwargs,agent_kwargs, compute_reward,checkpoint_dir=checkpoint_dir,
                        load_episode=460,transfer=False,max_experiment_time=60*60*24*6, mode=0., backup=True,
                        save_path = save_path, env = env)

# Read observation data
Gsolar = 0
Tout = -5
Tzone = 16 
price_now = get_current_hour_price()
Month_sin = 0.5
Month_cos = 0.866025404
Weekday = dt.datetime.now().weekday()
Timestep_sin = np.sin((np.pi/12)*(dt.datetime.now().hour + dt.datetime.now().minute/60))
Timestep_cos = np.cos((np.pi/12)*(dt.datetime.now().hour + dt.datetime.now().minute/60))
Case = 1
observations = pd.DataFrame(columns=['Solar irradiation','Outside temperature','Temperature 272','Month sin','Month cos','Weekday','Timestep sin','Timestep cos','Case','Price'],
                            data=[[Gsolar,Tout,Tzone,Month_sin,Month_cos,Weekday,Timestep_sin,Timestep_cos,Case,price_now]])
observations_n = 0.8*(observations - env.min_[observations.columns])/(env.max_[observations.columns]- env.min_[observations.columns]) + 0.1
data = np.append(observations_n,[0.1,0.9,0])

# Find actions
action = agent.predict(data)
action = torch.tensor([1.0]) if action[0].item() > 0.9 else action[0]
action = torch.tensor([0.0]) if action[0].item() < 0.1 else action[0]

# Translate actions to temperature setpoints
Tset = p2t_sp_hvac(action*100, sc_=False, sh_=True)

def write_tset_to_json(tset, filename="decisions.json"):
    """Writes Tset value to a JSON file in the format: {'sp_sh_r274': Tset}"""
    data = {"sp_sh_r274": tset}
    
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)

write_tset_to_json(Tset)
print(action.item())
print(Tset)



