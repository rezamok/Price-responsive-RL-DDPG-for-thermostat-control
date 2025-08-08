# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:32:21 2021

@author: kosta
"""
import os
import numpy as np
import kosta.hyperparams as hp

# +
def compute_reward_loris(environment, observation, action):
    """
    Small helper function to compute temperature rewards
    """

    reward = 0
    energy_scale = hp.energy_scale
    # Get the current temperatures
    temperatures = environment.scale_back_temperatures(
        environment.current_data[
            environment.n_autoregression + environment.current_step, environment.predictions_columns[:len(environment.rooms)]])
    #print(environment.scale_back_temperatures(observation[10]))                                                     
    bounds = (environment.current_data[environment.n_autoregression + environment.current_step, -2:] - 0.1) / 0.8 * (
                environment.temp_bounds[-1] - environment.temp_bounds[0]) + environment.temp_bounds[0]
    
    too_low = np.where(temperatures < bounds[0])[0]
    too_high = np.where(temperatures > bounds[1])[0]
    right = np.where((temperatures > bounds[0]) & (temperatures < bounds[1]))[0]
    # print("mina")
    # print(too_low)
    electricity_from_grid = environment.compute_electricity_from_grid(observation, action)
    energy = electricity_from_grid * energy_scale
    
    reward -= energy
                
    if len(too_low) > 0:
        reward += np.sum(temperatures[too_low] - bounds[0])
        if observation[-4] > 0.8999:
            reward -= np.sum((1 - (action[:-1][too_low] - 0.1) / 0.8) * (temperatures[too_low] - bounds[0]) ** 2)
        elif observation[-4] < 0.1001:
            reward -= np.sum((action[:-1][too_low] - 0.1) / 0.8 * (temperatures[too_low] - bounds[0]) ** 2)
        else:
            raise ValueError(f"Case has to be 0.1 or 0.9")

    if len(too_high) > 0:
        reward += np.sum(bounds[1] - temperatures[too_high])
        if observation[-4] > 0.8999:
            reward -= np.sum((action[:-1][too_high] - 0.1) / 0.8 * (bounds[1] - temperatures[too_high]) ** 2)
        elif observation[-4] < 0.1001:
            reward -= np.sum((1 - (action[:-1][too_high] - 0.1) / 0.8) * (bounds[1] - temperatures[too_high]) ** 2)
        else:
            raise ValueError(f"Case has to be 0.1 or 0.9")
    return reward/1000, temperatures, bounds

def compute_reward_mine(environment, observation, action):
    """
    Small helper function to compute temperature rewards
    """

    reward = 0
    energy_scale = hp.energy_scale
    # Get the current temperatures
    temperatures = environment.scale_back_temperatures(
        environment.current_data[
            environment.n_autoregression + environment.current_step, environment.predictions_columns[:len(environment.rooms)]])
    #print(environment.scale_back_temperatures(observation[10]))                                                     
    bounds = (environment.current_data[environment.n_autoregression + environment.current_step, -2:] - 0.1) / 0.8 * (
                environment.temp_bounds[-1] - environment.temp_bounds[0]) + environment.temp_bounds[0]
    
    too_low = np.where(temperatures < bounds[0])[0]
    too_high = np.where(temperatures > bounds[1])[0]
    right = np.where((temperatures > bounds[0]) & (temperatures < bounds[1]))[0]

    electricity_from_grid = environment.compute_electricity_from_grid(observation, action)
    energy = electricity_from_grid * energy_scale
    
    reward -= energy
                
    if len(too_low) > 0:
        reward += np.sum(temperatures[too_low] - bounds[0])
    if len(too_high) > 0:
        reward += np.sum(bounds[1] - temperatures[too_high])
    return reward/hp.reward_scale, temperatures, bounds

def compute_reward_comf(environment, observation, action):
    """
    Small helper function to compute temperature rewards
    """

    reward = 0
#    energy_scale = 9
    # Get the current temperatures
    temperatures = environment.scale_back_temperatures(
        environment.current_data[
            environment.n_autoregression + environment.current_step, environment.predictions_columns[:len(environment.rooms)]])
    #print(environment.scale_back_temperatures(observation[10]))                                                     
    bounds = (environment.current_data[environment.n_autoregression + environment.current_step, -2:] - 0.1) / 0.8 * (
                environment.temp_bounds[-1] - environment.temp_bounds[0]) + environment.temp_bounds[0]
    
    too_low = np.where(temperatures < bounds[0])[0]
    too_high = np.where(temperatures > bounds[1])[0]
    right = np.where((temperatures > bounds[0]) & (temperatures < bounds[1]))[0]

#    electricity_from_grid = environment.compute_electricity_from_grid(observation, action)
#    energy = electricity_from_grid * energy_scale
    
#    reward -= energy
                
    if len(too_low) > 0:
        reward += np.sum(temperatures[too_low] - bounds[0])
    if len(too_high) > 0:
        reward += np.sum(bounds[1] - temperatures[too_high])
    return reward/10, temperatures, bounds
#this is the main one
def compute_reward1(environment, observation, action):
    """
    Small helper function to compute temperature rewards
    """

    reward = 0
    energy_scale = hp.energy_scale
    # Get the current temperatures
    temperatures = environment.scale_back_temperatures(
        environment.current_data[
            environment.n_autoregression + environment.current_step, environment.predictions_columns[:len(environment.rooms)]])
    # print("alain")
    # print( environment.current_data[
    #         environment.n_autoregression + environment.current_step, :])
    bounds = (environment.current_data[environment.n_autoregression + environment.current_step, -2:] - 0.1) / 0.8 * (
                environment.temp_bounds[-1] - environment.temp_bounds[0]) + environment.temp_bounds[0]

    
    too_low = np.where(temperatures < bounds[0])[0]
    too_high = np.where(temperatures > bounds[1])[0]
    right = np.where((temperatures > bounds[0]) & (temperatures < bounds[1]))[0]
    #
    # electricity_from_grid = environment.compute_electricity_from_grid(observation, action)
    # energy = electricity_from_grid * energy_scale
    # # print("energy part of reward")
    # # print(- energy)
    # reward -= energy
    # print("minaaa")
    # print(temperatures)
    # print( bounds)
    if observation[-5] > 0.5:
        # heating
#        if temperatures[0] < (bounds[0]-0.5):
#            reward += -5
#         print("tem")
#         print(temperatures[0])
#         print("bound")
#         print(bounds[0])
         #if temperatures[0] < (bounds[0] + 0.5):
         if temperatures[0] < (bounds[0] ):
            reward += 5*(temperatures[0] - (bounds[0]) )
            # print("comfort part of reward")
            # print(temperatures[0] - (bounds[0]+0.5))
         elif temperatures[0] > (bounds[1]):
         # elif temperatures[0] > (bounds[0] + 0.5):
            reward += 5*((bounds[0] )-temperatures[0])
            # print("comfort part of reward")
            # print( (bounds[0]+0.5)-temperatures[0])
         elif temperatures[0] > (bounds[0]):
        # elif temperatures[0] > (bounds[0] + 0.5):
            reward += (bounds[0]) - temperatures[0]
    else:
        # cooling
        if temperatures[0] > (bounds[1] - 0.5):
            reward += (bounds[1]-0.5) - temperatures[0]

        elif temperatures[0] < (bounds[0] - 0.5):
            reward += temperatures[0] - (bounds[1]-0.5)
    # print("hey hey")
    # print(temperatures)
    return reward/hp.reward_scale, temperatures, bounds
    return reward/15, temperatures, bounds


def compute_reward(environment, observation, action):
    """
    Small helper function to compute temperature rewards
    """

    reward = 0
    comfort_reward = 0
    energy_reward = 0
    energy_scale = hp.energy_scale
    # Get the current temperatures
    temperatures = environment.scale_back_temperatures(
        environment.current_data[
            environment.n_autoregression + environment.current_step, environment.predictions_columns[
                                                                     :len(environment.rooms)]])
    # print("alain")
    # print( environment.current_data[
    #         environment.n_autoregression + environment.current_step, :])
    bounds = (environment.current_data[environment.n_autoregression + environment.current_step, -2:] - 0.1) / 0.8 * (
            environment.temp_bounds[-1] - environment.temp_bounds[0]) + environment.temp_bounds[0]
    price = (observation[9] - 0.1) / 0.8 * (environment.max_['Price'] - environment.min_['Price']) + environment.min_['Price']
    

    too_low = np.where(temperatures < bounds[0])[0]
    too_high = np.where(temperatures > bounds[1])[0]
    right = np.where((temperatures > bounds[0]) & (temperatures < bounds[1]))[0]
    #
    electricity_from_grid = environment.compute_electricity_from_grid(observation, action)
    # energy = electricity_from_grid * energy_scale 
    price_reward = - electricity_from_grid * energy_scale * price
    # price_reward = - electricity_from_grid * energy_scale
    
    # if observation[-5] > 0.5:
    #     # heating
    #     if temperatures[0] < (bounds[0]):
    #         comfort_reward = (temperatures[0] - (bounds[0]))
    #     elif temperatures[0] > (bounds[1]):
    #         comfort_reward = ((bounds[1]) - temperatures[0])
            # Only heating in here
        # heating
    if temperatures[0] < (bounds[0]):
        comfort_reward = (temperatures[0] - (bounds[0]))
    elif temperatures[0] > (bounds[1]):
        comfort_reward = ((bounds[1]) - temperatures[0])
    # else:
    #     # cooling
    #     if temperatures[0] > (bounds[1]):
    #         comfort_reward = (bounds[1]) - temperatures[0]
    #     elif temperatures[0] < (bounds[0]):
    #         comfort_reward = temperatures[0] - (bounds[1])

    reward = price_reward + comfort_reward

    return reward / hp.reward_scale, temperatures, bounds, price_reward, comfort_reward




