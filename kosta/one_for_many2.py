# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:55:51 2021

@author: kosta
"""
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from statistics import mean
import json
from models.util import load_data
from pfrl.pfrl.replay_buffer import batch_experiences
from agents.agents import RBAgent ,BangBangAgent,RBAgent_old_three, UnavoidableAgent
from kosta.reward import compute_reward
import kosta.hyperparams as hp
from kosta.environment import agent_kwargs,model_kwargs,data_kwargs
from kosta.checkpoint_utils import load_second_network,Alg1Logger,Alg2Logger,plot_test_data,plot_improvements,plot_hist_comf_viols
from kosta.model_utils import get_ddpg_agent_and_rb, get_network2
from kosta.network2_utils import (
    get_actual_delta_t,
    get_batches_from_experiences,
    update_network2,
    network2_train_step,
    network2_val_step,
    get_dataloaders,
    get_loss_fcn
)
import pandas as pd

def train_second_network(data_path):
    """
    Train the second network with algorithm2 from One for Many paper.
    """
    # define model, loss function and optimizer
    network2 = get_network2()
    network2 = network2.to(hp.device['name'])
    optim = hp.net2_optim(network2.parameters())
    loss_fcn = get_loss_fcn(hp.net2_loss_fcn) 
    # define data loaders
    data_split_path = os.path.join(data_path,f"room_{hp.rooms[0]}")
    train_loader,val_loader,test_loader=get_dataloaders(data_split_path,hp.batch_size)
    # train
    logger = Alg2Logger(hp.checkpoint_dir)
    print('Running initial validation...')
    print('Validating train set')
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    train_losses,_,_ = network2_val_step(
        val_loader=train_loader,
        network2=network2,
        loss_fcn=loss_fcn,
        device=hp.device['name']
    )
    print('Validating val set')
    val_losses,_,_ = network2_val_step(
        val_loader=val_loader,
        network2=network2,
        loss_fcn=loss_fcn,
        device=hp.device['name']
    )
    print('train loss:',mean(train_losses))
    print('val loss:',mean(val_losses))
    print('='*10)
    logger.update_logs(train_losses,val_losses)
    best_loss = 10000
    for epoch in range(hp.net2_epochs):
        print(f'EPOCH {epoch}/{hp.net2_epochs}')
        train_losses = network2_train_step(
            train_loader=train_loader,
            network2=network2,
            optim=optim,
            loss_fcn=loss_fcn,
            device=hp.device['name']
        )
        print('Validating')
        val_losses,_,_ = network2_val_step(
            val_loader=val_loader,
            network2=network2,
            loss_fcn=loss_fcn,
            device=hp.device['name']
        )
        print('train loss:',mean(train_losses))
        print('val loss:',mean(val_losses))
        print('='*10)
        logger.update_logs(train_losses,val_losses)
        if epoch % hp.chkpt_freq_alg2 == 0 and epoch != 0:
            logger.plot_logs()
            logger.save_logs()
        if mean(val_losses) < best_loss:
            best_loss = mean(val_losses)
            logger.save_weights(
                network2=network2,
                opt_net2=optim
            )
    # checkpoint at the end
    logger.plot_logs(epoch)
    logger.save_logs()
    return

def train_both_networks(env,network2=None):
    """
    Train both networks with algorithm1 from One for Many paper.
    
    :param env: Gym environment with step() and reset() methods
    :param network2: torch.nn.Module whose weights will be loaded from
            checkpoint_dir/alg2 in hyperparams.py
    """
    # define logger
    torch.set_default_tensor_type(torch.DoubleTensor)
    logger = Alg1Logger(hp.checkpoint_dir)
    # get network 2
    if network2 is None:
        network2 = load_second_network(os.path.join(hp.checkpoint_dir,'room_'+hp.rooms+'_env_data/alg2'))
    network2 = network2.to(hp.device['name'])
    net2_optim = hp.net2_optim(network2.parameters())
    net2_loss_fcn = get_loss_fcn(hp.net2_loss_fcn)
    # initialize replay buffer and DDPG agent
    ddpg_agent, replay_buffer = get_ddpg_agent_and_rb(env)
    # initialize RB agent
    rb_agent = UnavoidableAgent(data_kwargs,model_kwargs,agent_kwargs,compute_reward=compute_reward, room = hp.main_room)
    # continue training
    if hp.continue_training:
        logger.load_logs()
        network2,ddpg_agent.q_function,ddpg_agent.policy,net2_optim,\
            ddpg_agent.critic_optimizer,ddpg_agent.actor_optimizer=logger.load_weights(
            hp.load_episode,
            network2,
            ddpg_agent.q_function,
            ddpg_agent.policy,
            opt_net2=net2_optim,
            opt_q=ddpg_agent.critic_optimizer,
            opt_pol=ddpg_agent.actor_optimizer
        )
        replay_buffer = logger.load_replay_buffer(replay_buffer,hp.load_episode)
    # train
    sequences = logger.load_sequences_and_init_temps()[0] if hp.continue_training else None
    init_temps = logger.load_sequences_and_init_temps()[1] if hp.continue_training else None
    eval_stats = None
    eval_stats_rb = None
    start_episode = hp.load_episode+1 if hp.continue_training else 0
    iteration = hp.threshold_length*(hp.load_episode+1) if hp.continue_training else 0
    for episode in range(start_episode,hp.max_episodes):
        episode_reward = 0
        net2_losses = []
        print(f'Episode {episode}')
        s_curr = torch.tensor(env.reset())
        done = False
        ep_iter = 0
        while not done:
            delta_t = torch.tensor(ddpg_agent.batch_act([s_curr])).reshape((1,)) # select action in either train or eval mode (maybe i can give it only one example in batch)
            with torch.no_grad():
                delta_t = torch.tensor([0.9]) if delta_t.item() > 0.9 else delta_t
                delta_t = torch.tensor([0.1]) if delta_t.item() < 0.1 else delta_t
                action = delta_t
            s_next,reward,done,_,_,_ = env.step(action)
            s_next = torch.tensor(s_next)
            replay_buffer.append(
                state=s_curr,
                action=delta_t,
                reward=reward,
                next_state=s_next,
                is_state_terminal=done,
                real_action=action
            )
            #ddpg_agent.batch_observe() # Adds data to replay buffer
            if len(replay_buffer) > hp.rb_init_capacity:
                experiences = replay_buffer.sample(hp.batch_size)
                ddpg_agent.update(experiences) # calculates actor and critic loss and updates them  
                batch_state,batch_next_state,_,batch_action,_ = get_batches_from_experiences(experiences)
                batch_state = batch_state.to(hp.device['name'])
                batch_next_state = batch_next_state.to(hp.device['name'])
                batch_action = batch_action.to(hp.device['name'])
                if not hp.freeze_network2:
                    net2_loss = update_network2(
                        batch_state,
                        batch_next_state,
                        batch_action,
                        network2,
                        net2_optim,
                        net2_loss_fcn
                    )
                else:
                    batch_delta_t = get_actual_delta_t(batch_state, batch_next_state)
                    with torch.no_grad():
                        batch_pred = batch_delta_t
                    net2_loss = net2_loss_fcn(batch_pred, batch_action)
                if iteration % hp.target_update_freq == 0 and iteration != 0:
                    ddpg_agent.sync_target_network() # sync target network with original
                net2_losses.append(float(net2_loss))
            s_curr = s_next
            # save
            episode_reward += float(reward)      
            iteration += 1
            ep_iter += 1
            print(f'\riteration {iteration}',end='')
        print('')
        if episode % hp.chkpt_freq_alg1 == 0 and episode != 0:
            logger.plot_logs(episode)
            logger.save_logs()
            if hp.save_replay_buffer:
                logger.save_replay_buffer(replay_buffer,episode)
            logger.save_weights(
                episode=episode,
                network2=network2,
                q_func=ddpg_agent.q_function,
                policy=ddpg_agent.policy,
                opt_net2=net2_optim,
                opt_q=ddpg_agent.critic_optimizer,
                opt_pol=ddpg_agent.actor_optimizer
                )
        # if episode > 1:
        #     print("soghra")
        #     print(eval_stats["reward"])
        #     print(eval_stats["avg_reward"])
        logger.update_logs(episode_reward,net2_losses,eval_stats,eval_stats_rb)
        print('episode reward:',episode_reward)
        print('averaged network2 loss:',mean(net2_losses) if net2_losses else [])
        print('='*10)
        if episode % hp.alg1_eval_every == 0:
            if (episode == 0) or (len(replay_buffer) > hp.rb_init_capacity):
                print('Testing the agent..')
                sequences,init_temps, eval_stats, eval_stats_rb = test_agent(
                    env=env,
                    network2=network2,
                    ddpg_agent=ddpg_agent,
                    rb_agent=rb_agent,
                    test_dir=os.path.join(logger.save_dir,f'tests_alg1_ep_{episode}'),
                    num_episodes=hp.alg1_eval_episodes,
                    sequences=sequences,
                    init_temps=init_temps
                )

                if episode == 0:
                    logger.save_sequences_and_init_temps(sequences,init_temps)
    return


def test_agent(env,network2,ddpg_agent,rb_agent,test_dir,num_episodes=5,sequences=None,init_temps=None):
    """
    Test the agent by selecting actions greedy.

    Parameters
    ----------
    env : environment for the RL agent.
    network2 : second part of the policy network
    ddpg_agent : ddpg agent
    test_dir : directory where to save plots
    num_episodes : num of episodes to run the agent

    Returns
    -------
    to do

    """ 
    sequences = [None]*num_episodes if sequences is None else sequences
    init_temps = [None]*num_episodes if init_temps is None else init_temps
    os.makedirs(test_dir,exist_ok=True)
    comfort_violations = []
    prices = []
    episode_rewards = []
    episode_rewards_rb = []
    comfort_violations_rb = []
    episode_price_rewards = []
    episode_comfort_rewards = []
    episode_price_rewards_rb = []
    episode_comfort_rewards_rb = []
    prices_rb = []
    for episode in range(num_episodes):  
        print(f'episode {episode}/{num_episodes}')
        actions = []
        delta_ts = []
        print("111")
        print(episode)
        print(sequences)
        print("mina")
        print(sequences[episode])
        print(env.reset(sequence=sequences[episode],is_test=True, init_temp=init_temps[episode]))
        s_curr = torch.Tensor(env.reset(sequence=sequences[episode],is_test=True, init_temp=init_temps[episode]))
        init_temps[episode] = env.init_temp
        sequences[episode] = env.current_sequence
        print("222")
        print(sequences[episode])
        temperatures = [env.scale_back_temperatures(s_curr[2].cpu().numpy())]
        ####################### we need to consider #####################
        T_ambs = [scale_back_T_amb(env, s_curr[1].cpu().numpy())]
        Price = [scale_back_T_amb(env, s_curr[9].cpu().numpy())]
        irradiances = [scale_back_irradiance(env, s_curr[0].cpu().numpy())]
        # case = 'Heating' if env.current_data[0,-4]>0.45 else 'Cooling'
        case = 'Heating'
        done = False
        episode_reward = 0
        episode_price_reward = 0
        episode_comfort_reward = 0
        ep_iter = 0
        demand = []
        demand_rb = []
        
        result_array = np.array(list(range(env.current_sequence[0], env.current_sequence[1] + 1)))
        dates = env.umar_model.data.index[result_array[hp.n_autoregression:]]#.strftime("%H:%M:%S")
        while not done:
            with ddpg_agent.eval_mode():

                delta_t = torch.Tensor(ddpg_agent.batch_act(s_curr)) # select action in either train or eval mode (maybe i can give it only one example in batch)
               
            with torch.no_grad():
                delta_t = torch.tensor([0.9]) if delta_t.item() > 0.9 else delta_t
                delta_t = torch.tensor([0.1]) if delta_t.item() < 0.1 else delta_t
                action = delta_t
            # action = env.scale_action(s_curr[-4],action)
            print("action")
            print(action)
            s_next,reward,done,_,price_reward, comfort_reward = env.step(action)
            
            
            s_curr = torch.Tensor(s_next)
            episode_reward += reward
            episode_price_reward += price_reward
            episode_comfort_reward += comfort_reward
            ep_iter+=1
            actions.append(action)
            delta_ts.append(delta_t)
            temperatures.append(env.scale_back_temperatures(s_curr[2].cpu().numpy()))
            T_ambs.append(scale_back_T_amb(env, s_curr[1].cpu().numpy()))
            Price.append(scale_back_Price(env, s_curr[9].cpu().numpy()))
            irradiances.append(scale_back_irradiance(env, s_curr[0].cpu().numpy()))
        episode_rewards.append(episode_reward)
        episode_price_rewards.append(episode_price_reward)
        episode_comfort_rewards.append(episode_comfort_reward)
        comfort_violations.append(env.comfort_violations)
        prices.append(env.prices)
        a = sum(prices[episode])
        # RUN rb_agent
        reward,_,temperatures_rb,actions_rb, price_reward, comfort_reward, demand_rb = rb_agent.run(
            sequences[episode],
            init_temp=init_temps[episode],
            render=False
        )
        episode_rewards_rb.append(reward)
        episode_price_rewards_rb.append(price_reward)
        episode_comfort_rewards_rb.append(comfort_reward)
        comfort_violations_rb.append(rb_agent.env.comfort_violations)
        prices_rb.append(rb_agent.env.prices)
        b = sum(prices_rb[episode])
        actions_rb = [tensor.detach().numpy() for tensor in actions_rb]
        # Plot the data
                
        plot_test_data(
            test_dir,
            episode,
            actions,
            actions_rb,
            delta_ts,
            temperatures,
            temperatures_rb,
            hp.temp_bounds,
            T_ambs,
            Price,
            irradiances,
            case,
            dates,
            mean(env.comfort_violations),
            mean(env.prices),
            mean(rb_agent.env.comfort_violations),
            mean(rb_agent.env.prices),
            env.electricity_imports,
            demand_rb
        )
    avg_comf_viol = mean([mean(x) for x in comfort_violations])
    max_comf_viol = max([max(list(map(abs, x))) for x in comfort_violations])
    avg_price = mean([sum(x) for x in prices])

    max_price = max([max(list(map(abs, x))) for x in prices])
    avg_reward = mean(episode_rewards)
    avg_price_reward = mean(episode_price_rewards)
    avg_comfort_reward = mean(episode_comfort_rewards)
    stats = {
        # 'reward': episode_rewards,
        'avg_reward': avg_reward,
        'avg_comf_viol': avg_comf_viol,
        'max_comf_viol': max_comf_viol,
        'avg_price': avg_price,
        'max_price': max_price,
        'price_reward': avg_price_reward,
        'comfort_reward': avg_comfort_reward,
    }
    avg_comf_viol_rb = mean([mean(x) for x in comfort_violations_rb])
    max_comf_viol_rb = max([max(list(map(abs, x))) for x in comfort_violations_rb])
    avg_price_rb = mean([sum(x) for x in prices_rb])
    max_price_rb = max([max(list(map(abs, x))) for x in prices_rb])
    avg_reward_rb = mean(episode_rewards_rb)
    avg_price_reward_rb = mean(episode_price_rewards_rb)
    avg_comfort_reward_rb = mean(episode_comfort_rewards_rb)
    stats_rb = {
        # 'reward': episode_rewards_rb,
        'avg_reward': avg_reward_rb,
        'avg_comf_viol': avg_comf_viol_rb,
        'max_comf_viol': max_comf_viol_rb,
        'avg_price': avg_price_rb,
        'max_price': max_price_rb,
        'price_reward': avg_price_reward_rb,
        'comfort_reward': avg_comfort_reward_rb
    }
    with open(os.path.join(test_dir,'stats_DDPG.json'),'w') as fp:
        json.dump(stats,fp)
    with open(os.path.join(test_dir,'stats_RB.json'),'w') as fp:
        json.dump(stats_rb,fp)
    print('---'*5)
    print('DDPG STATS:')
    print('Average reward is: {:.4f}'.format(avg_reward))
    print('Average comfort violation is: {:.4f}'.format(avg_comf_viol))
    print('Maximal absolute comfort violation is: {:.4f}'.format(max_comf_viol))
    print('Average price is: {:.4f}'.format(avg_price))
    print('Maximal absolute price is: {:.4f}'.format(max_price))
    print('---'*5)
    print('RB STATS:')
    print('Average reward is: {:.4f}'.format(avg_reward_rb))
    print('Average comfort violation is: {:.4f}'.format(avg_comf_viol_rb))
    print('Maximal absolute comfort violation is: {:.4f}'.format(max_comf_viol_rb))
    print('Average price is: {:.4f}'.format(avg_price_rb))
    print('Maximal absolute price is: {:.4f}'.format(max_price_rb))
    print('---'*5)
    print('DDPG agent compared to RB agent:')
    print('Percentage of average comfort violation difference: {:.3f}%'.format(
          (stats_rb['avg_comf_viol']-stats['avg_comf_viol'])/stats_rb['avg_comf_viol']*100
    ))
    print('Percentage of average price difference: {:.3f}%'.format(
        (stats_rb['avg_price']-stats['avg_price'])/stats_rb['avg_price']*100
    ))
    print('Difference between maximal comfort violations: {:.3f} degrees'.format(
        stats_rb['max_comf_viol'] - stats['max_comf_viol']
    ))
    print('Difference between maximal prices: {:.3f}'.format(
         stats_rb['max_price'] -  stats['max_price']
    ))
    plot_improvements(stats,stats_rb,comfort_violations,comfort_violations_rb,test_dir)
    plot_hist_comf_viols(comfort_violations,test_dir)
    return sequences,init_temps,stats,stats_rb

def predict_agent(env,network2,pcnn,X_columns,ddpg_agent,test_dir, obs_path=None):
    # Read data
    observations = load_data('Data_Predict4',obs_path)

    num_episodes = observations.shape[0]
    # Initilization
    comfort_violations = []
    prices = []
    episode_rewards = []
    episode_price_rewards = []
    episode_comfort_rewards = []
    actions = []
    temperatures = []
    Price = []
    demand = []
    # Normalize
    observations_n = 0.8*(observations - env.min_[observations.columns])/(env.max_[observations.columns]- env.min_[observations.columns]) + 0.1
    # observation_columns
    for episode in range(num_episodes):  
        print(f'episode {episode}/{num_episodes}')
        delta_ts = []
        s_curr = torch.Tensor(np.append(observations_n.iloc[episode],[0.1,0.9,0]))
        # T_ambs = [scale_back_T_amb(env, s_curr[1].cpu().numpy())]
        # Price = [scale_back_T_amb(env, s_curr[9].cpu().numpy())]
        # irradiances = [scale_back_irradiance(env, s_curr[0].cpu().numpy())]
        case = 'Heating'
        done = False
        episode_reward = 0
        episode_price_reward = 0
        episode_comfort_reward = 0
        with ddpg_agent.eval_mode():
            delta_t = torch.Tensor(ddpg_agent.batch_act(s_curr))
        with torch.no_grad():
            delta_t = torch.tensor([0.9]) if delta_t.item() > 0.9 else delta_t
            delta_t = torch.tensor([0.1]) if delta_t.item() < 0.1 else delta_t
            action = delta_t
        
        # s_next,reward,done,_,price_reward, comfort_reward = env.step(action)
        dd = observations_n.iloc[episode].copy()
        dd['Valve 272'] = action.item()
        # dd['Temperature 273'] = dd['Temperature 272']
        dd['Temperature 273'] = 0.2
        
        n = 30
        d = torch.Tensor([dd[X_columns]])
        model_state = None
        p = [None]*n
        for j in range(n):
              pred, model_state = pcnn.model(d,model_state, warm_start=True)
              p[j] = pred.item()
              P = p[j]
        T_pred = env.scale_back_temperatures(P)    
        
        temperatures.append(T_pred)
    

        
        if episode < num_episodes - 1:
            observations_n.iloc[episode+1, 2] = P


        # temperatures.append(env.scale_back_temperatures(s_curr[2].cpu().numpy()))
        # Price.append(scale_back_Price(env, s_curr[9].cpu().numpy()))
    
        
        actions.append(action)
        
        
        demand.append(env.compute_electricity_from_grid(s_curr, action))

        # s_curr = torch.Tensor(s_next)
        # episode_reward += reward
        # episode_price_reward += price_reward
        # episode_comfort_reward += comfort_reward
        delta_ts.append(delta_t)
        # temperatures.append(env.scale_back_temperatures(s_curr[2].cpu().numpy()))
        # T_ambs.append(scale_back_T_amb(env, s_curr[1].cpu().numpy()))
        # Price.append(scale_back_Price(env, s_curr[9].cpu().numpy()))
        # irradiances.append(scale_back_irradiance(env, s_curr[0].cpu().numpy()))
        # episode_rewards.append(episode_reward)
        # episode_price_rewards.append(episode_price_reward)
        # episode_comfort_rewards.append(episode_comfort_reward)
        # comfort_violations.append(env.comfort_violations)
        # prices.append(env.prices)
        # a = sum(prices[episode])
        
        
        ## Plot the data
        # plot_test_data(
        #     test_dir,
        #     episode,
        #     actions,
        #     actions_rb,
        #     delta_ts,
        #     temperatures,
        #     temperatures_rb,
        #     hp.temp_bounds,
        #     T_ambs,
        #     Price,
        #     irradiances,
        #     case,
        #     dates,
        #     mean(env.comfort_violations),
        #     mean(env.prices),
        #     mean(rb_agent.env.comfort_violations),
        #     mean(rb_agent.env.prices),
        #     env.electricity_imports,
        #     demand_rb
        # )
    # avg_comf_viol = mean([mean(x) for x in comfort_violations])
    # max_comf_viol = max([max(list(map(abs, x))) for x in comfort_violations])
    # avg_price = mean([sum(x) for x in prices])

    # max_price = max([max(list(map(abs, x))) for x in prices])
    # avg_reward = mean(episode_rewards)
    # avg_price_reward = mean(episode_price_rewards)
    # avg_comfort_reward = mean(episode_comfort_rewards)
    # stats = {
    #     # 'reward': episode_rewards,
    #     'avg_reward': avg_reward,
    #     'avg_comf_viol': avg_comf_viol,
    #     'max_comf_viol': max_comf_viol,
    #     'avg_price': avg_price,
    #     'max_price': max_price,
    #     'price_reward': avg_price_reward,
    #     'comfort_reward': avg_comfort_reward,
    # }
    # avg_comf_viol_rb = mean([mean(x) for x in comfort_violations_rb])
    # max_comf_viol_rb = max([max(list(map(abs, x))) for x in comfort_violations_rb])
    # avg_price_rb = mean([sum(x) for x in prices_rb])
    # max_price_rb = max([max(list(map(abs, x))) for x in prices_rb])
    # avg_reward_rb = mean(episode_rewards_rb)
    # avg_price_reward_rb = mean(episode_price_rewards_rb)
    # avg_comfort_reward_rb = mean(episode_comfort_rewards_rb)
    # stats_rb = {
    #     # 'reward': episode_rewards_rb,
    #     'avg_reward': avg_reward_rb,
    #     'avg_comf_viol': avg_comf_viol_rb,
    #     'max_comf_viol': max_comf_viol_rb,
    #     'avg_price': avg_price_rb,
    #     'max_price': max_price_rb,
    #     'price_reward': avg_price_reward_rb,
    #     'comfort_reward': avg_comfort_reward_rb
    # }
    # with open(os.path.join(test_dir,'stats_DDPG.json'),'w') as fp:
    #     json.dump(stats,fp)
    # with open(os.path.join(test_dir,'stats_RB.json'),'w') as fp:
    #     json.dump(stats_rb,fp)
    # print('---'*5)
    # print('DDPG STATS:')
    # print('Average reward is: {:.4f}'.format(avg_reward))
    # print('Average comfort violation is: {:.4f}'.format(avg_comf_viol))
    # print('Maximal absolute comfort violation is: {:.4f}'.format(max_comf_viol))
    # print('Average price is: {:.4f}'.format(avg_price))
    # print('Maximal absolute price is: {:.4f}'.format(max_price))
    # print('---'*5)
    # print('RB STATS:')
    # print('Average reward is: {:.4f}'.format(avg_reward_rb))
    # print('Average comfort violation is: {:.4f}'.format(avg_comf_viol_rb))
    # print('Maximal absolute comfort violation is: {:.4f}'.format(max_comf_viol_rb))
    # print('Average price is: {:.4f}'.format(avg_price_rb))
    # print('Maximal absolute price is: {:.4f}'.format(max_price_rb))
    # print('---'*5)
    # print('DDPG agent compared to RB agent:')
    # print('Percentage of average comfort violation difference: {:.3f}%'.format(
    #       (stats_rb['avg_comf_viol']-stats['avg_comf_viol'])/stats_rb['avg_comf_viol']*100
    # ))
    # print('Percentage of average price difference: {:.3f}%'.format(
    #     (stats_rb['avg_price']-stats['avg_price'])/stats_rb['avg_price']*100
    # ))
    # print('Difference between maximal comfort violations: {:.3f} degrees'.format(
    #     stats_rb['max_comf_viol'] - stats['max_comf_viol']
    # ))
    # print('Difference between maximal prices: {:.3f}'.format(
    #      stats_rb['max_price'] -  stats['max_price']
    # ))
    # plot_improvements(stats,stats_rb,comfort_violations,comfort_violations_rb,test_dir)
    # plot_hist_comf_viols(comfort_violations,test_dir)
    return actions, temperatures, demand

        

    
def scale_back_T_amb(env,T_amb):
    return (T_amb-0.1)*(env.max_["Outside temperature"]-env.min_["Outside temperature"])/0.8 + env.min_["Outside temperature"]

def scale_back_Price(env,Price):
    return (Price-0.1)*(env.max_["Price"]-env.min_["Price"])/0.8 + env.min_["Price"]

def scale_back_irradiance(env,irradiances):
    return (irradiances- 0.1) * (env.max_["Solar irradiation"] - env.min_["Solar irradiation"]) / 0.8 + env.min_[
        "Solar irradiation"]
    # max_ir = np.max(irradiances)
    # max_id = np.argmax(irradiances)
    # ir = f"s{max_id+1}"
    # return (max_ir-0.1)*(env.max_[ir]-env.min_[ir])/0.8 + env.min_[ir]
    
    
