"""
Created on Wed Jun 16 12:31:13 2021

@author: kosta
"""
import torch.nn.functional as F
import torch
from pfrl.pfrl.explorers.additive_ou import AdditiveOU


checkpoint_dir = 'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/checkpoints'

rooms = '272'
transfer_room = '274'
main_room = '272'
use_historical_data = True
dynamic_bounds=False
dynamic_temp_bounds = [20,22,23]

temp_bounds = [23,26]  #[20, 22, 23, 25]
price_levels=[1]
interval = 15
price_type="Constant"

# n_autoregression=20
# n_autoregression=15
n_autoregression=14
threshold_length=96*5
save_replay_buffer=True
uk_coeff_scaling = 1
# energy_scale=9
# energy_scale=12
# energy_scale=0.05
# energy_scale= 0.040/850
# energy_scale= 0.00250/850*0
energy_scale= 0.1/47.49*1

# reward_scale=0.1 # Best so far
reward_scale=60
# reward_scale=1

continue_training=False
# continue_training = True
load_episode=800

freeze_network2=False
# max_delta_t_degrees = 0.25 # in degrees
# max_delta_t = 0.8/(15/max_delta_t_degrees)
max_delta_t_degrees = 1
max_delta_t = 1
min_Price = 0
max_Price = 50
use_he_normalization = True

heating=True
cooling=False
strong_case = False
weak_case = False

max_episodes=int(1e6)
# batch_size = 5000
batch_size = 32
device = {'name':'cpu', 'id':None}#
# device = {'name':'cuda:0', 'id':0}
#device = {'name':'cuda:0'}
env_model = 'ARX'
# chkpt_freq_alg1 = 50 # in episodes
chkpt_freq_alg1 = 20 # in episodes
chkpt_freq_alg2 = 2
# alg1_eval_every = 50 # in episodess
alg1_eval_every = 20 # in episodes
alg1_eval_episodes = 20

# policy network hyperparams
# pol_in_size=16
pol_in_size=13
pol_out_size=1
# pol_hidden_sizes=[128,64,64] # Best so far
pol_hidden_sizes=[128,64,64,32] # Best so far

# pol_hidden_sizes=[128,256,400]#[100,200,200]#[128,256,256,256,400]#[128,128,256]

# pol_hidden_sizes=[128,64,64,32]#[100,200,200]#[128,256,256,256,400]#[128,128,256]
# pol_hidden_sizes=[128,256,256,256,400]#[128,128,256]
# pol_hidden_sizes=[128,256,256,256,512]#[128,128,256]

pol_nonlinearity=F.relu
pol_optim = lambda params: torch.optim.Adam(params, lr=2e-6)
# pol_optim = lambda params: torch.optim.Adam(params, lr=1e-5)


target_update_freq = 240 # in iterations
apply_burning_func = True

# q function hyperparams
# q_in_size=17
q_in_size=14
q_out_size=1
# q_hidden_sizes=[128,64,32]# Best so far
q_hidden_sizes=[128,64,64,32]# Best so far

# q_hidden_sizes=[128,256,400]#[100,200,200]#[128,256,256,256,400]#[128,128,256]

# q_hidden_sizes=[128,64,64,32]#[128,128,256] # [128,256,256,256,400]
# q_hidden_sizes=[128,256,256,256,512]
# q_hidden_sizes=[128,256,256,256,512]

q_nonlinearity=F.relu
q_optim = lambda params: torch.optim.Adam(params, lr=1e-6)
# q_optim = lambda params: torch.optim.Adam(params, lr=1e-5)

# replay buffer params
# rb_capacity = 2.5e5
rb_capacity = 5e4
rb_num_steps = 1
# rb_init_capacity = 32
# rb_init_capacity = 2000
rb_init_capacity = 5000

# explorer
# explorer = AdditiveOU(mu=0,theta=0.15,sigma=0.15,start_with_mu=False) # Taken from Bauman's code
# explorer = AdditiveOU(mu=0,theta=0.2,sigma=0.2,start_with_mu=False) # Best so far

# explorer = AdditiveOU(mu=0,theta=0.3,sigma=0.7,start_with_mu=False) # Taken from Bauman's code
# explorer = AdditiveOU(mu=0,theta=0.8,sigma=0.8,start_with_mu=False) # Taken from Bauman's code
explorer = AdditiveOU(mu=0,theta=0.8,sigma=0.8,start_with_mu=False) # Taken from Bauman's code

#explorer = additive_ou(mu=0,theta=0.15,sigma=0.05,start_with_mu=False) # Taken from Bauman's code
gamma = 0.97
# gamma = 0.95

# network2 hyperparams
# net2_in_size=17
net2_in_size=13
net2_out_size=1
net2_hidden_sizes=[128,256,256]#[128,256,256,256,400]
net2_nonlinearity=F.relu
net2_optim = lambda params: torch.optim.Adam(params, lr=1e-4,weight_decay=0)
# net2_epochs = 1000
net2_epochs = 350
net2_loss_fcn = 'mae'


