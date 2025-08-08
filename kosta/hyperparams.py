"""
Created on Wed Jun 16 12:31:13 2021

@author: kosta
"""
import torch.nn.functional as F
import torch
from pfrl.pfrl.explorers.additive_ou import AdditiveOU


# checkpoint_dir = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/checkpoints6_Future2'
# checkpoint_dir = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/checkpoints6_Future3'
# checkpoint_dir = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/checkpoints6_Future22'
# checkpoint_dir = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/checkpoints6_Future22'
checkpoint_dir = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/checkpoints5_1'
# checkpoint_dir = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/checkpoints6_Future22'
# checkpoint_dir = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/checkpoints7_Future1'

# checkpoint_dir = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/checkpoints_RL1'

rooms = '272'
transfer_room = '274'
main_room = '272'
use_historical_data = True
dynamic_bounds=False
# dynamic_temp_bounds = [20,22,23]

temp_bounds = [21,24]
# temp_bounds = [23.95,24.05]
price_levels=[1]
interval = 15
price_type="Constant"

n_autoregression=14
threshold_length=96*5
save_replay_buffer=True
uk_coeff_scaling = 1
energy_scale= 0.06/47.49*100
# energy_scale= 0.06/47.49*100*10

# reward_scale= 10
reward_scale= 1

continue_training=False
load_episode=1560

freeze_network2=False
# For visualization
max_delta_t_degrees = 1
max_delta_t = 1
min_Price = 0
max_Price = 110
use_he_normalization = True

heating=True
cooling=False
strong_case = False
weak_case = False

max_episodes=int(1e6)
batch_size = 32
device = {'name':'cpu', 'id':None}#
env_model = 'ARX'
chkpt_freq_alg1 = 20 # in episodes
chkpt_freq_alg2 = 2
alg1_eval_every = 20 # in episodes
alg1_eval_episodes = 20


# policy network hyperparams
pol_in_size=14
pol_out_size=1
pol_hidden_sizes=[128,256,256,32]

pol_nonlinearity=F.relu
# pol_optim = lambda params: torch.optim.Adam(params, lr=2e-5)
pol_optim = lambda params: torch.optim.Adam(params, lr=2e-4)
# pol_optim = lambda params: torch.optim.Adam(params, lr=1e-4)



target_update_freq = 240 # in iterations
apply_burning_func = True

# q function hyperparams
q_in_size=15
q_out_size=1
q_hidden_sizes=[128,256,256,32]


q_nonlinearity=F.relu
# q_optim = lambda params: torch.optim.Adam(params, lr=2e-5)
q_optim = lambda params: torch.optim.Adam(params, lr=1e-4)
# q_optim = lambda params: torch.optim.Adam(params, lr=0.5e-4)


# replay buffer params
rb_capacity = 5e4
rb_num_steps = 1
rb_init_capacity = 5000

# explorer = AdditiveOU(mu=0,theta=0.3,sigma=0.2,start_with_mu=False)
explorer = AdditiveOU(mu=0,theta=0.2,sigma=0.2,start_with_mu=False)

# gamma = 0.97
gamma = 0.99

# network2 hyperparams
net2_in_size=13
net2_out_size=1
net2_hidden_sizes=[128,256,256]
net2_nonlinearity=F.relu
net2_optim = lambda params: torch.optim.Adam(params, lr=1e-4,weight_decay=0)
net2_epochs = 350
net2_loss_fcn = 'mae'


