# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:48:15 2021

@author: kosta
"""
from pfrl.pfrl.nn import MLP
import kosta.hyperparams as hp
from kosta.model_utils import get_network2
import os
import torch
from statistics import mean
import matplotlib.pyplot as plt
import json
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import shutil
import pickle
import numpy as np
import scipy.io

def load_from_disk(txt_path):
    with open(txt_path, "rb") as f:
        values =  pickle.load(f)
    return values

def save_to_disk(txt_path, values):
    if os.path.isfile(txt_path):
        os.remove(txt_path)
    with open(txt_path, "wb") as fp:   
        pickle.dump(values, fp)
    return

def load_second_network(checkpoint_dir,network2=None):
    """
    Load the second network weights from 'network2.pt' in the checkpoint_dir.
    If network2 is None, network 2 will be created with parameters from 
    hyperparams.py
    """
    load_path = os.path.join(checkpoint_dir, 'best_chkpt.pt')
    if os.path.exists(load_path):
        state_dict = torch.load(load_path,map_location=hp.device['name'])
        if network2 is None:
            network2 = get_network2()
        network2.load_state_dict(state_dict['network2'])
        print(f"Network 2 weights are loaded from:\n'{load_path}'")
    else:
        raise ValueError(f"There is no network2.pt in '{checkpoint_dir}'")
    return network2

class Alg1Logger:
    """
    Logger for train_bot_networks function in one_for_many.py - which is algorithm 1
    in One for Many paper.
    """
    def __init__(self,checkpoint_dir, checkpoint_hyperparams=True):
        self.checkpoint_dir = checkpoint_dir
        self.save_dir = os.path.join(checkpoint_dir,'alg1')
        self.plots_dir = os.path.join(self.save_dir,'plots')
        os.makedirs(self.save_dir,exist_ok=True)
        os.makedirs(self.plots_dir,exist_ok=True)
        self.logs = {
            'net2loss_e': [], # network 2 episode loss
            'reward_e': [], # episode rewards
            'eval_reward':[],
            'eval_avg_comf_viol': [],
            'eval_avg_price': [],
            'eval_reward_rb':[],
            'eval_avg_comf_viol_rb': [],
            'eval_avg_price_rb': [],
            'eval_reward_instant': [],
            'eval_reward_rb_instant': [],
            'eval_reward_price': [],
            'eval_reward_comfort': [],
            'eval_reward_price_rb': [],
            'eval_reward_comfort_rb': []
         }
        if checkpoint_hyperparams:
            # copy hyperparams.py to checkpoint directory
            try:
                dest = os.path.join(self.save_dir,'hyperparams.py')
                shutil.copyfile(hp.__file__[1:],dest)
                print(f"Hyperparams are checkpointed at:\n'{dest}'")
            except:
                print('Failed to checkpoint hyperparams.py')
    def update_logs(self, episode_reward, net2_losses, eval_stats, eval_stats_rb):
        if len(net2_losses) != 0:
            self.logs['net2loss_e'].append(mean(net2_losses))
        self.logs['reward_e'].append(episode_reward)
        if eval_stats is not None:
            # self.logs['eval_reward_instant'].append(eval_stats['reward'])
            self.logs['eval_reward'].append(eval_stats['avg_reward'])
            self.logs['eval_avg_comf_viol'].append(eval_stats['avg_comf_viol'])
            self.logs['eval_avg_price'].append(eval_stats['avg_price'])
            self.logs['eval_reward_rb'].append(eval_stats_rb['avg_reward'])
            # self.logs['eval_reward_rb_instant'].append(eval_stats_rb['reward'])
            self.logs['eval_avg_comf_viol_rb'].append(eval_stats_rb['avg_comf_viol'])
            self.logs['eval_avg_price_rb'].append(eval_stats_rb['avg_price'])
            
            self.logs['eval_reward_price'].append(eval_stats['price_reward'])
            self.logs['eval_reward_comfort'].append(eval_stats['comfort_reward'])
            self.logs['eval_reward_price_rb'].append(eval_stats_rb['price_reward'])
            self.logs['eval_reward_comfort_rb'].append(eval_stats_rb['comfort_reward'])
        return
    def save_logs(self):
        save_path = os.path.join(self.save_dir,'logs.json')
        with open(save_path, 'w') as fp:
            json.dump(self.logs, fp)
        print(f"Logs are saved at:\n'{save_path}'")
        return
    def load_logs(self):
        load_path = os.path.join(self.save_dir,'logs.json')
        if os.path.exists(load_path):
            with open(load_path,'r') as fp:
                self.logs = json.load(fp)
            print(f'Logs are loaded from {load_path}')
            return self.logs
        else:
            raise ValueError(f'No logs in {load_path}')
    def save_sequences_and_init_temps(self,sequences,init_temps,save_path=None):
        if save_path is None:
            save_path = os.path.join(self.save_dir,'sequences_and_init_temps.txt')
        save_to_disk(
            save_path,
            {'sequences':sequences, 'init_temps':init_temps}
        )
    def load_sequences_and_init_temps(self,load_path=None):
        if load_path is None:
            load_path = os.path.join(self.save_dir,'sequences_and_init_temps.txt')
        if os.path.exists(load_path):
            load_dict= load_from_disk(txt_path=load_path)
            print(f'Sequences and init_temps are loaded from {load_path}')
            return load_dict['sequences'], load_dict['init_temps']
        else:
            raise ValueError(f'No sequences and init_temps in {load_path}')
    def save_replay_buffer(self,replay_buffer,episode):
        replay_buffer.save(os.path.join(self.save_dir,f'replay_buffer_{episode}.txt'))
    def load_replay_buffer(self,replay_buffer,load_episode):
        load_path = os.path.join(self.save_dir,f'replay_buffer_{load_episode}.txt')
        if os.path.exists(load_path):   
            replay_buffer.load(load_path)
            print(f'Replay buffer is loaded from {load_path}')
            return replay_buffer
        else:
            raise ValueError(f'No replay buffer in {load_path}')
    def plot_logs(self, episode):
        plt.figure()
        plt.plot(self.logs['reward_e'])
        plt.title('Cumulative reward per episode')
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.savefig(os.path.join(self.plots_dir,f'reward_{episode}.png'))
        plt.cla()
        plt.clf()
        if len(self.logs['eval_reward']) >= 2:
            plt.figure()
            plt.plot(self.logs['eval_reward'], c='tab:blue')
            plt.plot(self.logs['eval_reward_rb'], c='tab:green')
            plt.legend(['eval_reward', 'eval_reward_rb'])
            plt.title('Eval cumulative reward per episode')
            plt.xlabel('eval iter')
            plt.ylabel('reward')
            plt.savefig(os.path.join(self.plots_dir,f'eval_rewad_{episode}.png'))
            plt.cla()
            plt.clf()
        if len(self.logs['eval_reward']) >= 2:
            plt.figure()
            plt.plot(self.logs['eval_reward_comfort'], linestyle='-', c='tab:blue')
            plt.plot(self.logs['eval_reward_price'], linestyle='-', c='tab:green')
            plt.plot(self.logs['eval_reward_comfort_rb'], linestyle='--', c='tab:blue')
            plt.plot(self.logs['eval_reward_price_rb'], linestyle='--', c='tab:green')
            plt.legend(['RL comfort reward', 'RL cost reward', 'RB comfort reward', 'RB cost reward'])
            plt.title('cumulative reward terms per episode')
            plt.xlabel('eval iter')
            plt.ylabel('reward value')
            plt.savefig(os.path.join(self.plots_dir,f'Both_rewards_{episode}.png'))
            plt.cla()
            plt.clf()
        # if len(self.logs['eval_reward_instant']) >= 2:
        #     plt.figure()
        #     plt.plot(self.logs['eval_reward_instant'], c='tab:blue')
        #     plt.plot(self.logs['eval_reward_rb_instant'], c='tab:green')
        #     plt.legend(['eval_reward', 'eval_reward_rb'])
        #     plt.title('Eval cumulative reward per episode')
        #     plt.xlabel('eval iter')
        #     plt.ylabel('reward')
        #     plt.savefig(os.path.join(self.plots_dir, f'eval_rewad_instant_{episode}.png'))
        #     plt.cla()
        #     plt.clf()
        if len(self.logs['eval_avg_comf_viol']) >= 2:
            plt.figure()
            plt.plot(self.logs['eval_avg_comf_viol'], c='tab:blue')
            plt.plot(self.logs['eval_avg_comf_viol_rb'], c='tab:green')
            plt.legend(['eval_reward', 'eval_reward_rb'])
            plt.title('Eval average comfort violation')
            plt.xlabel('eval iter')
            plt.ylabel('violation in degrees')
            plt.savefig(os.path.join(self.plots_dir,f'eval_comf_viol_{episode}.png'))
            plt.cla()
            plt.clf()
        if len(self.logs['eval_avg_price']) >= 2:
            plt.figure()
            plt.plot(self.logs['eval_avg_price'], c='tab:blue')
            plt.plot(self.logs['eval_avg_price_rb'], c='tab:green')
            plt.legend(['eval_reward', 'eval_reward_rb'])
            plt.title('Eval average price')
            plt.xlabel('eval iter')
            plt.ylabel('price')
            plt.savefig(os.path.join(self.plots_dir,f'eval_price_{episode}.png'))
            plt.cla()
            plt.clf()
        plt.close("all")
        
    def save_weights(self,episode,network2,q_func,policy,opt_net2,opt_q,opt_pol):
        state_dict = {
            'network2':network2.state_dict(),
            'q_func':q_func.state_dict(),
            'policy':policy.state_dict(),
            'opt_network2':opt_net2.state_dict(),
            'opt_q_func':opt_q.state_dict(),
            'opt_policy':opt_pol.state_dict()
        }
        save_path = os.path.join(self.save_dir, f'episode_{episode}.pt')
        torch.save(state_dict, save_path)
        print(f"Weights are saved at:\n'{save_path}'")
        return
    def load_weights(self,episode,network2,q_func,policy,opt_net2=None,opt_q=None,opt_pol=None):
        load_path = os.path.join(self.save_dir, f'episode_{episode}.pt')
        if os.path.exists(load_path):
            state_dict = torch.load(load_path,map_location=hp.device['name'])
            if network2 is not None:
                network2.load_state_dict(state_dict['network2']) 
            q_func.load_state_dict(state_dict['q_func']) 
            policy.load_state_dict(state_dict['policy']) 
            if opt_net2 is not None:
                opt_net2.load_state_dict(state_dict['opt_network2'])
            if opt_q is not None:
                opt_q.load_state_dict(state_dict['opt_q_func'])
            if opt_pol is not None:
                opt_pol.load_state_dict(state_dict['opt_policy'])
            print(f"Checkpoint loaded from {load_path}")
            return network2,q_func,policy,opt_net2,opt_q,opt_pol
        else:
            raise ValueError(f"There is no checkpoint in {load_path}")
class Alg2Logger:
    """
    Logger for train_second_network function in one_for_many.py - which is algorithm 2
    in One for Many paper.
    """
    def __init__(self,checkpoint_dir, checkpoint_hyperparams=True):
        self.checkpoint_dir = checkpoint_dir
        self.save_dir = os.path.join(checkpoint_dir,'alg2')
        self.plots_dir = os.path.join(self.save_dir,'plots')
        os.makedirs(self.plots_dir,exist_ok=True)
        os.makedirs(self.save_dir,exist_ok=True)
        self.logs = {
            'train loss': [], # network 2 episode loss
            'val loss': [] # episode rewards
         }
        if checkpoint_hyperparams:
            # copy hyperparams.py to checkpoint directory
            try:
                dest = os.path.join(self.save_dir,'hyperparams.py')
                shutil.copyfile(hp.__file__[1:],dest)
                print(f"Hyperparams are checkpointed at:\n'{dest}'")
            except:
                print('Failed to checkpoint hyperparams.py')
    def update_logs(self, train_losses, val_losses):
        self.logs['train loss'].append(mean(train_losses))
        self.logs['val loss'].append(mean(val_losses))
        return
    def plot_logs(self):
        # plt.figure()
        plt.figure(figsize=(8, 4))
        plt.plot(self.logs['train loss'],c='tab:blue')
        plt.plot(self.logs['val loss'],c='tab:red')
        # plt.title('training curves')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend(['Train losses','Validation losses'])
        plt.grid()
        plt.ylim(0, 0.15)
        plt.xlim(-10, 300)
        plt.savefig(os.path.join(self.plots_dir,'loss.png'))
    def save_logs(self):
        save_path = os.path.join(self.save_dir,'logs.json')
        with open(save_path, 'w') as fp:
            json.dump(self.logs, fp)
        print(f"Logs are saved at:\n'{save_path}'")
        return
    def save_weights(self,network2,opt_net2):
        state_dict = {
            'network2':network2.state_dict(),
            'opt_network2':opt_net2.state_dict()
        }
        save_path = os.path.join(self.save_dir, f'best_chkpt.pt')
        torch.save(state_dict, save_path)
        print(f"Weights are saved at:\n'{save_path}'")
        return
    def load_weights(self,device,network2,opt=None):
        load_path = os.path.join(self.save_dir, f'best_chkpt.pt')
        if os.path.exists(load_path):
            state_dict = torch.load(load_path,map_location=device)
            network2.load_state_dict(state_dict['network2']) 
            if opt is not None:
                opt.load_state_dict(state_dict['opt_network2'])
            print(f"Checkpoint loaded from {load_path}")
            network2 = network2.to(device)
            return network2,opt
        else:
            raise ValueError(f"There is no checkpoint in {load_path}")
    def load_logs(self):
        load_path = os.path.join(self.save_dir,'logs.json')
        with open(load_path,'r') as fp:
            logs = json.load(fp)
        print(f'Logs are loaded from {load_path}')
        return logs
def plot_test_data(save_dir,episode,actions,actions_rb,delta_ts,temperatures,
                   temperatures_rb,bounds,T_ambs, Price, irradiances,case,dates,
                   comf_viol,price,comf_viol_rb,price_rb,demand,demand_rb):
    """
    Plot test data from test_agent function in one_for_many.py
    """
    comf_viol_perc = (comf_viol_rb-comf_viol)/comf_viol_rb*100
    if comf_viol_rb == 0:
        if comf_viol == 0:
            comf_viol_perc = '0.00%'
        else:
            comf_viol_perc = "can't calculate percentage"
    else:
        comf_viol_perc = f'{comf_viol_perc:.2f}%'
    price_perc = (price_rb-price)/price_rb*100
    if price_rb == 0:
        if price == 0:
            price_perc = '0.00%'
        else:
            price_perc = "can't calculate percentage"
    else:
        price_perc = f'{price_perc:.2f}%'
    title_fontsize = 35
    y_ticks_fontsize = 27
    x_ticks_fontsize = 25
    label_size = 30
    linewidth = 7
    legend_size = 25
    fig = plt.figure(figsize=[45,45])
    plt.title(
        f'{case}\nImprovement over the baseline: comfort = '+comf_viol_perc+', price = '+price_perc,
        fontsize=title_fontsize+10, pad=70
    )
    # Plot temperatures

    ax = fig.add_subplot(411)
    ax.plot(temperatures,label='RL',linewidth=linewidth)
    ax.plot(temperatures_rb,label='RB',linewidth=linewidth)
    ax.set_title('Room temperature',fontsize=title_fontsize)
    ax.set_ylabel('Temperature (°C)',fontsize=label_size)
    ax.tick_params(axis='y', labelsize=y_ticks_fontsize)
    ax.tick_params(axis='x', labelsize=x_ticks_fontsize)
    ax.axhline(bounds[0],color='black',label='bounds',linewidth=linewidth)
    for bound in bounds[1:]:
        ax.axhline(bound,color='black',label=None,linewidth=linewidth)
    ax.legend(prop={'size': legend_size},loc='upper left')
    
    # Plot actions
    actions_values = [float(action.item()) for action in actions]
    ax = fig.add_subplot(412)
    ax.plot(actions_values, linewidth=linewidth, label='RL')
    #ax.plot(actions,linewidth=linewidth,label='DDPG (network2)')
    ax.plot(actions_rb,linewidth=linewidth,label='RB')
    ax.tick_params(axis='y', labelsize=y_ticks_fontsize)
    ax.tick_params(axis='x', labelsize=x_ticks_fontsize)
    ax.set_ylim(0.1,0.9)
    ax.set_title('Valve openning',fontsize=title_fontsize)
    ax.legend(prop={'size': legend_size},loc='upper right')
    
    # Plot dynamic prices
    ax = fig.add_subplot(413)
    ax.plot(dates,Price,linewidth=linewidth,color='gray')
    ax.tick_params(axis='y', labelsize=y_ticks_fontsize)
    ax.tick_params(axis='x', labelsize=x_ticks_fontsize)
    ax.set_ylim(hp.min_Price,hp.max_Price)
    ax.set_ylabel('Price (CHF/kWh)',fontsize=label_size)
    ax.set_title('Dynamic price',fontsize=title_fontsize)

    # Plot Ambient temperature and irradiance
    ax = fig.add_subplot(414)
    ax.tick_params(axis='y', labelsize=y_ticks_fontsize)
    ax.tick_params(axis='x', labelsize=x_ticks_fontsize)
    ax.plot(dates,T_ambs,linewidth=linewidth,color='orange')
    ax_new = ax.twinx()
    ax_new.tick_params(axis='y', labelsize=y_ticks_fontsize)
    ax_new.plot(dates,irradiances,linewidth=linewidth,color='green')
    ax_new.set_ylabel('Irradiance (W/m^2)',fontsize=label_size,color='green')

    fig.autofmt_xdate()
    ax.set_ylabel('Ambient temperature (°C)',fontsize=label_size,color='orange')
    ax.set_xlabel("Time", fontsize=label_size)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%h-%d %H:%M'))
    fig.savefig(os.path.join(save_dir,f'results_ep_{episode}_{case}'),bbox_inches='tight')
    plt.cla()
    plt.close(fig)
    plt.close("all")
    file_path = os.path.join(save_dir,f'results_ep_{episode}_{case}.mat')
    # scipy.io.savemat(file_path, {'temperatures': temperatures, 'temperatures_rb': temperatures_rb , 'actions_values':actions_values,'actions_rb':actions_rb,'dates': dates,'T_ambs':T_ambs ,'irradiances':irradiances , 'Price':Price,'comf_viol_perc':comf_viol_perc,'price_perc':price_perc })
    scipy.io.savemat(file_path, {'temperatures': temperatures, 'temperatures_rb': temperatures_rb ,
                                 'actions_values':actions_values,'actions_rb':actions_rb,'dates': dates,
                                 'T_ambs':T_ambs ,'irradiances':irradiances , 'Price':Price,'comf_viol_perc':comf_viol_perc,
                                 'price_perc':price_perc, 'demand':demand, 'demand_rb':demand_rb})

def plot_hist_comf_viols(comf_viols,save_dir):
    comf_viols = np.concatenate(comf_viols)
    plt.figure()
    plt.hist(comf_viols,bins=50)
    plt.title('Histogram of comfort violations')
    plt.xlabel('comfort violations in degrees')
    plt.savefig(os.path.join(save_dir,'histogram_comf_viols.png'))
    plt.figure()
    plt.hist(comf_viols[comf_viols!=0.],bins=50)
    plt.title('Histogram of comfort violations (without 0)')
    plt.xlabel('comfort violations in degrees')
    _, _, ymin, ymax = plt.axis()
    plt.vlines(mean(comf_viols[comf_viols!=0.]),ymin,ymax,colors='r',linewidth=3)
    plt.vlines(np.percentile(comf_viols[comf_viols!=0.],90),ymin,ymax,colors='black',linewidth=3)
    plt.text(
        mean(comf_viols[comf_viols!=0.])+0.05,
        (ymax-ymin)*0.7,
        'Mean',color='red',
        bbox=dict(facecolor='none',edgecolor='red',boxstyle='round,pad=1')
    )
    plt.text(
        np.percentile(comf_viols[comf_viols!=0.],90)+0.05,
        (ymax-ymin)*0.7,
        '90% of samples',color='black',
        bbox=dict(facecolor='none',edgecolor='black',boxstyle='round,pad=1')
    )
    plt.savefig(os.path.join(save_dir,'histogram_comf_viols_wo_0.png'))
    return
def plot_improvements(stats,stats_rb,comfort_violations,comfort_violations_rb,save_dir):
    
    width = 0.7  # the width of the bars
    max_comf_viol = max([max(x) for x in comfort_violations])
    max_comf_viol_rb = max([max(x) for x in comfort_violations_rb])
    avg_comf_viol = stats['avg_comf_viol']
    avg_comf_viol_rb = stats_rb['avg_comf_viol']
    price_savings=(stats_rb['avg_price']-stats['avg_price'])/stats_rb['avg_price']*100
    fig = plt.figure(figsize=(7,5))
    plt.xticks([])
    plt.yticks([])
    scaling = abs(price_savings)/100. if abs(price_savings) > 5. else 0.05
    max_lim = max(max_comf_viol,max_comf_viol_rb)
    if price_savings > 0:
        min_lim = 0
    else:
        min_lim = max_lim*scaling
        min_lim = -min_lim-0.3*max_lim
    max_lim = max_lim + 0.2*max_lim
    
# =============================================================================
    # plot comfort violation
    ax = fig.add_subplot(111,label='2')
    x = np.arange(0,3,1.5)
    rects1 = ax.bar(x - width/2, [avg_comf_viol_rb,max_comf_viol_rb], width, label='baseline-RB agent')
    rects2 = ax.bar(x + width/2, [avg_comf_viol,max_comf_viol], width, label='DDPG agent')
    for rect in rects1+rects2:
        height = rect.get_height()    
        ax.text(
            rect.get_x() + rect.get_width() / 2.0, 
            height, f"{height:.3f}"+"$^\circ$C", ha='center', 
            va='bottom'
        )
    ax.legend()
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlim(-1,4)
    min_lim = np.nan_to_num(min_lim, nan=0)
    max_lim = np.nan_to_num(max_lim, nan=0)
    ax.set_ylim(min_lim,max_lim)
    
# =============================================================================
    price_plot_value = max(max_comf_viol,max_comf_viol_rb)*scaling
    price_plot_value = -price_plot_value if price_savings < 0 else price_plot_value
    # plot price savings
    ax = fig.add_subplot(111,frame_on=False,label='3',sharex=ax)
    x = np.arange(0,4,1.5)
    rects3 = ax.bar(x[-1], [price_plot_value], width+0.1, tick_label='price savings',color='purple')
    for rect in rects3:
        height = rect.get_height() 
        height = height - 0.07*max_lim if height < 0 else height
        ax.text(
            rect.get_x() + rect.get_width() / 2.0, 
            height, f'{price_savings:.2f}%', ha='center', 
            va='bottom'
        )
    ax.set_xticks(x)
    ax.set_xticklabels(['average comfort violation', 'maximum comfort violation','electricity price savings'])
    ax.set_yticks([])
    ax.set_xlim(-1,4)
    ax.set_ylim(min_lim,max_lim)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir,f'improvements_over_the_baseline.png'))
    
    
    