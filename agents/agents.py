import os
import numpy as np
import pickle
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from agents.helpers import check_env

from agents.a2c import A2C
from agents.ppo2 import PPO2
from agents.td3 import TD3
from agents.her import HER
import torch
#if sys.platform == "win32":
from stable_baselines import ACER, ACKTR, DQN, SAC
#else:
    #from stable_baselines import ACER, ACKTR, DDPG, DQN, GAIL, PPO1, SAC, TRPO
from stable_baselines.common.vec_env import VecNormalize, VecCheckNan, SubprocVecEnv
from stable_baselines.common.cmd_util import make_vec_env

from agents.callbacks import CallbackList, SaveVecNormalizeCallback, CheckpointCallback, \
    EvalCallback, CompareAgentCallback

from agents.policies import CustomLSTMPolicy, CustomMlpPolicy
from agents.environments import UMAREnv, ToyEnv, compute_reward

from agents.helpers import prepare_models, prepare_toy_models
from agents.helpers import prepare_model_mina
from util.util import FIGURES_SAVE_PATH
from models.util import build_physics_based_inputs_outputs_indices


class RBAgent:
    """
    Very basic Rule-Based agent heating or cooling only as soon as the bounds are reached.
    """

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, umar_model=None, battery_model=None,
                 compute_reward=compute_reward, room = '272'):

        self.simple_env = agent_kwargs["simple_env"]

        print("\nPreparing the rule-based agent")
        #all_inputs, all_outputs, base_indices, effect_indices = build_physics_based_inputs_outputs_indices()

        # all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
        #               'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Power 272', 'Valve 272', 'Case']
        #room = 272
        if room == '274':
            base_indices = [1, 4, 7, 8, 9, 10, 11]
            effect_indices = [2, 4, 5, 6]
            all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 274', 'Temperature 273', 'Month sin',
                          'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 274', 'Case']
            all_outputs = ['Temperature 274']
        elif room == '272':
            base_indices = [1, 3, 7, 8, 9, 10, 11]
            effect_indices = [2, 3, 5, 6]
            all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
                          'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 272', 'Case']
            all_outputs = ['Temperature 272']
        if (umar_model is None) | (battery_model is None):
            # Prepare the battery and UMAR models
            if self.simple_env:
                umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                              Y_columns=all_outputs,
                                                              X_columns=all_inputs,
                                                              base_indices=base_indices,
                                                              effect_indices=effect_indices,
                                                              room=room)
            else:
                umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                              Y_columns=all_outputs,
                                                              X_columns=all_inputs,
                                                              base_indices=base_indices,
                                                              effect_indices=effect_indices,
                                                              room=room)

        # Prepare the environment
        if self.simple_env:
            env = ToyEnv(umar_model=umar_model,
                         battery_model= None,
                         agent_kwargs=agent_kwargs,
                         compute_reward=compute_reward)
        else:
            env = UMAREnv(umar_model=umar_model,
                          battery_model=battery_model,
                          agent_kwargs=agent_kwargs,
                          compute_reward=compute_reward)

        self.small_model = agent_kwargs["small_model"]
        self.compute_reward = compute_reward

        self.her = agent_kwargs["her"]

        # Save the environment
        self.env = env
        self.half_degree_scaled = (self.env.scaled_temp_bounds[1, :] - self.env.scaled_temp_bounds[0, :]) / 2 / (
                self.env.temp_bounds[1] - self.env.temp_bounds[0])
        # (np.mean(self.env.scaled_temp_bounds, axis=0) - self.env.scaled_temp_bounds[0, :]) / 2

    def take_decision(self, observation, goal=None):
        """
        Decision making function: heat when the temperature drops below the lower bound and cool when it
        goes above the upper bound.

        Args:
            observation: the current observation of the environment from which to take the decision
        """
        if goal is None:
            if len(self.env.temp_bounds) == 2:
                goal = self.env.scaled_temp_bounds[0, :] + self.half_degree_scaled

                if observation[2] < 0.10001:
                    goal = self.env.scaled_temp_bounds[0, :] + self.half_degree_scaled
                elif observation[2] < 0.49:
                    goal = self.env.scaled_temp_bounds[1, :] + self.half_degree_scaled
                else:
                    raise NotImplementedError(f"What? Should be lower bound, lower than 0.5: {observation[-3]}")


        if self.small_model:
            index = 8 if self.env.simple else 9
        else:
            index = 8 if self.env.simple else 10

        action = np.array((goal >
                           observation[-(index + 2 * len(self.env.rooms)):
                                       -(index + len(self.env.rooms))]) * 1 * 0.8 + 0.1)
       

        # Decide what to do of the battery
        if self.env.battery:
            electricity_consumption = self.env.umar_model.min_["Electricity total consumption"] + (
                    observation[self.env.elec_column] - 0.1) * (
                                              self.env.umar_model.max_["Electricity total consumption"] -
                                              self.env.umar_model.min_[
                                                  "Electricity total consumption"]) / 0.8
            energies = observation[-(3 + len(self.env.rooms)):-3]
            energy = sum([self.env.umar_model.min_[f"Energy room {room}"] + (energies[i] - 0.1) * (
                    self.env.umar_model.max_[f"Energy room {room}"] - self.env.umar_model.min_[
                f"Energy room {room}"]) / 0.8 for i, room in enumerate(self.env.rooms)])

            consumption = energy / self.env.COP + electricity_consumption

            if consumption > 0:
                # Transform in kWh, then in kW over 15 min
                margin = 0.96 / self.env.battery_size * (
                        observation[-1] - self.env.battery_margins[0] - 0.25) * 60 / self.env.umar_model.interval
                if margin > consumption:
                    action = np.append(action, max(-consumption, -self.env.battery_max_power))
                else:
                    action = np.append(action,
                                       min(max(-margin, -self.env.battery_max_power), self.env.battery_max_power))
            else:
                margin = 0.96 / self.env.battery_size * (
                        self.env.battery_margins[1] - observation[-1] - 0.25) * 60 / self.env.umar_model.interval
                if margin > -consumption:
                    action = np.append(action, min(-consumption, self.env.battery_max_power))
                else:
                    action = np.append(action,
                                       min(max(margin, -self.env.battery_max_power), self.env.battery_max_power))

        # Return the action
        return action

    def run(self, sequence, goal_number=None, init_temp=None, render=True):
        """
        Run the agent over a sequence of inputs

        Args:
            sequence: a sequence of indices to get the data. If None, a random sequence is used
            render:   whether to plot the result or not
        """

        # Reset the environment to get the first observation
        observation = self.env.reset(sequence, goal_number, init_temp)
        temperatures = [self.env.scale_back_temperatures(observation[2])]
        actions = []
        if self.her:
            observation = observation['observation']
            goal = self.env.desired_goals[self.env.goal_number]
        else:
            goal = None

        done = False
        cumul_reward = 0
        length = 0
        k = 0
        # Iterate until the episode is over (i.e. the sequence is finished)
        while not done:
            # Choose an action
            action = self.take_decision(observation, goal)
            k = k+1
            # print("while")
            # print(action.dtype)
            action = torch.Tensor(action)
            # print(action.dtype)
            # print(observation.dtype)
            # Take a step
            observation, reward, done,_ = self.env.step(action, rule_based=True)
            # observation, reward, done, info = self.env.step(action, rule_based=True)
            temperatures.append(self.env.scale_back_temperatures(observation[2]))
            actions.append(action)
            if self.her:
                observation = observation['observation']
            # Recall the rewards
            cumul_reward += reward
            length += 1

        # Some modifications needed to be compatible with the 'render' function
        self.env.current_step -= 1
        if self.env.battery:
            self.env.battery_powers.pop()
            self.env.battery_soc.pop()
        self.env.electricity_imports.pop()

        # If wanted, plot the result
        if render:
            self.env.render()

        return cumul_reward, length, temperatures, actions


class BangBangAgent:
    """
    Very basic Rule-Based agent heating or cooling only as soon as the bounds are reached.
    """

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, umar_model=None, battery_model=None,
                 compute_reward=compute_reward, room='272'):
        self.simple_env = agent_kwargs["simple_env"]
        print("\nPreparing the bang-bang agent")
        ########################################
        if room == '274':
            base_indices = [1, 4, 7, 8, 9, 10, 11]
            effect_indices = [2, 4, 5, 6]
            all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 274', 'Temperature 273', 'Month sin',
                          'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 274', 'Case']
            all_outputs = ['Temperature 274']
        elif room == '272':
            base_indices = [1, 3, 7, 8, 9, 10, 11]
            effect_indices = [2, 3, 5, 6]
            all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
                          'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 272', 'Case']
            all_outputs = ['Temperature 272']
        ########################################

        if (umar_model is None) | (battery_model is None):
            # Prepare the battery and UMAR models
            if self.simple_env:
                # umar_model, battery_model = prepare_toy_models(data_kwargs, model_kwargs, agent_kwargs)
                umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                              Y_columns=all_outputs,
                                                              X_columns=all_inputs,
                                                              base_indices=base_indices,
                                                              effect_indices=effect_indices,
                                                              room=room)
            else:
                # umar_model, battery_model = prepare_models(data_kwargs, model_kwargs)
                umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                              Y_columns=all_outputs,
                                                              X_columns=all_inputs,
                                                              base_indices=base_indices,
                                                              effect_indices=effect_indices,
                                                              room=room)

        # Prepare the environment
        if self.simple_env:
            env = ToyEnv(umar_model=umar_model,
                         battery_model=battery_model,
                         agent_kwargs=agent_kwargs,
                         compute_reward=compute_reward)
        else:
            env = UMAREnv(umar_model=umar_model,
                          battery_model=battery_model,
                          agent_kwargs=agent_kwargs,
                          compute_reward=compute_reward)

        self.small_model = agent_kwargs["small_model"]
        self.compute_reward = compute_reward

        self.her = agent_kwargs["her"]
        self.on = np.array([False] * len(env.rooms))

        # Save the environment
        self.env = env
        self.degree_scaled = (self.env.scaled_temp_bounds[1, :] - self.env.scaled_temp_bounds[0, :]) / (
                self.env.temp_bounds[1] - self.env.temp_bounds[0])

    def take_decision(self, observation, goal=None):
        """
        Decision making function: heat when the temperature drops below the lower bound and cool when it
        goes above the upper bound.

        Args:
            observation: the current observation of the environment from which to take the decision
        """
        if goal is None:
            if len(self.env.temp_bounds) == 2:
                    goal = self.env.scaled_temp_bounds[0, :]

            if len(self.env.temp_bounds) == 4:
                    if observation[2] < 0.10001:
                        goal = self.env.scaled_temp_bounds[0, :]
                    elif observation[2] < 0.49:
                        goal = self.env.scaled_temp_bounds[1, :]
                    else:
                        raise NotImplementedError(f"What? Should be lower bound, lower than 0.5: {observation[-3]}")

        if self.small_model:
            index = 8 if self.env.simple else 9
        else:
            index = 8 if self.env.simple else 10

        temperatures = observation[-(index + 2 * len(self.env.rooms)): -(index + len(self.env.rooms))]

        # Check in which case we are

        for room in range(len(self.on)):
            action = np.array((goal >
                               observation[-(index + 2 * len(self.env.rooms)):
                                           -(index + len(self.env.rooms))]) * 1 * 0.8 + 0.1)

        # Decide what to do of the battery
        if self.env.battery:
            electricity_consumption = self.env.umar_model.min_["Electricity total consumption"] + (
                    observation[self.env.elec_column] - 0.1) * (
                                              self.env.umar_model.max_["Electricity total consumption"] -
                                              self.env.umar_model.min_[
                                                  "Electricity total consumption"]) / 0.8
            energies = observation[-(3 + len(self.env.rooms)):-3]
            energy = sum([self.env.umar_model.min_[f"Energy room {room}"] + (energies[i] - 0.1) * (
                    self.env.umar_model.max_[f"Energy room {room}"] - self.env.umar_model.min_[
                f"Energy room {room}"]) / 0.8 for i, room in enumerate(self.env.rooms)])

            consumption = energy / self.env.COP + electricity_consumption

            if consumption > 0:
                # Transform in kWh, then in kW over 15 min
                margin = 0.96 / self.env.battery_size * (
                        observation[-1] - self.env.battery_margins[0] - 0.25) * 60 / self.env.umar_model.interval
                if margin > consumption:
                    action = np.append(action, max(-consumption, -self.env.battery_max_power))
                else:
                    action = np.append(action,
                                       min(max(-margin, -self.env.battery_max_power), self.env.battery_max_power))
            else:
                margin = 0.96 / self.env.battery_size * (
                        self.env.battery_margins[1] - observation[-1] - 0.25) * 60 / self.env.umar_model.interval
                if margin > -consumption:
                    action = np.append(action, min(-consumption, self.env.battery_max_power))
                else:
                    action = np.append(action,
                                       min(max(margin, -self.env.battery_max_power), self.env.battery_max_power))

        return action
        # return self.on * 1 * 0.8 + 0.1

    def run(self, sequence, goal_number=None, init_temp=None, render=True):
        """
        Run the agent over a sequence of inputs

        Args:
            sequence: a sequence of indices to get the data. If None, a random sequence is used
            render:   whether to plot the result or not
        """

        # Reset the environment to get the first observation
        observation = self.env.reset(sequence, goal_number, init_temp)
        temperatures = [self.env.scale_back_temperatures(observation[2])]
        actions = []
        if self.her:
            observation = observation['observation']
            goal = self.env.desired_goals[self.env.goal_number]
        else:
            goal = None

        done = False
        cumul_reward = 0
        length = 0
        k = 0
        # Iterate until the episode is over (i.e. the sequence is finished)
        while not done:
            # Choose an action
            action = self.take_decision(observation, goal)
            k = k + 1
            action = torch.Tensor(action)
            # Take a step
            observation, reward, done, info = self.env.step(action, rule_based=True)
            temperatures.append(self.env.scale_back_temperatures(observation[2]))
            actions.append(action)
            if self.her:
                observation = observation['observation']
            # Recall the rewards
            cumul_reward += reward
            length += 1

        # Some modifications needed to be compatible with the 'render' function
        self.env.current_step -= 1
        if self.env.battery:
            self.env.battery_powers.pop()
            self.env.battery_soc.pop()
        self.env.electricity_imports.pop()

        # If wanted, plot the result
        if render:
            self.env.render()

        # return cumul_reward, length
        return cumul_reward, length, temperatures, actions

class UnavoidableAgent:
    """
    Agent to capture unavoidable energy usage and comfort violations (i.e. energy - and corresponding
    comfort violations required to at least reach the bounds since the agent might start out of them).
    """

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, umar_model=None, battery_model=None,
                 compute_reward=compute_reward, room = '272'):

        self.simple_env = agent_kwargs["simple_env"]
        # room = room[0]
        print("\nPreparing the unavoidable agent")
        if room == '274':
            base_indices = [1, 4, 7, 8, 9, 10, 11]
            effect_indices = [2, 4, 5, 6]
            all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 274', 'Temperature 273', 'Month sin',
                          'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 274', 'Case', 'Price', 'PriceChange']
            all_outputs = ['Temperature 274']
        elif room == '272':
            base_indices = [1, 3, 7, 8, 9, 10, 11]
            effect_indices = [2, 3, 5, 6]
            all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
                          'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 272', 'Case', 'Price', 'PriceChange']
            all_outputs = ['Temperature 272']
        if (umar_model is None) | (battery_model is None):
            # Prepare the battery and UMAR models
            if self.simple_env:
                umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                              Y_columns=all_outputs,
                                                              X_columns=all_inputs,
                                                              base_indices=base_indices,
                                                              effect_indices=effect_indices,
                                                              room=room)
            else:
                umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                              Y_columns=all_outputs,
                                                              X_columns=all_inputs,
                                                              base_indices=base_indices,
                                                              effect_indices=effect_indices,
                                                              room=room)

            # Prepare the environment
        if self.simple_env:
            env = ToyEnv(umar_model=umar_model,
                         battery_model=None,
                         agent_kwargs=agent_kwargs,
                         compute_reward=compute_reward)
        else:
            env = UMAREnv(umar_model=umar_model,
                          battery_model=battery_model,
                          agent_kwargs=agent_kwargs,
                          compute_reward=compute_reward)

        self.small_model = agent_kwargs["small_model"]
        self.compute_reward = compute_reward

        self.her = agent_kwargs["her"]
        self.rooms = env.rooms
        self.goals = np.array([0.] * len([self.rooms]))
        self.dones = np.array([False] * len([self.rooms]))
        self.flag = True # true in the bound false out of bound
        self.case = True  # true for heating and false for cooling
        # Save the environment
        self.env = env

    def take_decision(self, observation):
        """
        Decision making function: Aim for the set point in the middle, heat if it is below, cool if above
        and look at how much energy was used to at least enter the bound

        Args:
            observation: the current observation of the environment from which to take the decision
        """
        if self.small_model:
            index = 8 if self.env.simple else 9
        else:
            index = 8 if self.env.simple else 10

        temperatures = (self.env.scale_back_temperatures(observation[2]))
        # print("mina")
        # print(temperatures)

        # temperatures = self.env.scale_back_temperatures(observation[-(index + 2 * len(self.env.rooms)):
        #                                                             -(index + len(self.env.rooms))])
        bounds = (observation[-3: -1] - 0.1) / 0.8 * (self.env.temp_bounds[-1] - self.env.temp_bounds[0]) \
                 + self.env.temp_bounds[0]
        # print(bounds)
        for room in range(len([self.rooms])):
            if temperatures < bounds[0]:
                self.goals[room] = bounds[0]
                self.case = True
                self.flag = False
            elif temperatures > bounds[1]:
                self.goals[room] = bounds[1]
                self.case = False
                self.flag = False
            else:
                self.dones[room] = True
                self.flag = True
        action = [0.1]

        if self.flag == False and self.case == True:
            action = np.array(((self.goals > temperatures) * 1)  * 0.8 + 0.1)
        if self.flag == True and self.case == True:
            action = [0.9]




        done = np.all(self.dones)

        # Decide what to do of the battery
        if self.env.battery:
            electricity_consumption = self.env.umar_model.min_["Electricity total consumption"] + (
                    observation[self.env.elec_column] - 0.1) * (
                                              self.env.umar_model.max_["Electricity total consumption"] -
                                              self.env.umar_model.min_[
                                                  "Electricity total consumption"]) / 0.8
            energies = observation[-(3 + len(self.env.rooms)):-3]
            energy = sum([self.env.umar_model.min_[f"Energy room {room}"] + (energies[i] - 0.1) * (
                    self.env.umar_model.max_[f"Energy room {room}"] - self.env.umar_model.min_[
                f"Energy room {room}"]) / 0.8 for i, room in enumerate(self.env.rooms)])

            consumption = energy / self.env.COP + electricity_consumption

            if consumption > 0:
                # Transform in kWh, then in kW over 15 min
                margin = 0.96 / self.env.battery_size * (
                        observation[-1] - self.env.battery_margins[0] - 0.25) * 60 / self.env.umar_model.interval
                if margin > consumption:
                    action = np.append(action, max(-consumption, -self.env.battery_max_power))
                else:
                    action = np.append(action,
                                       min(max(-margin, -self.env.battery_max_power), self.env.battery_max_power))
            else:
                margin = 0.96 / self.env.battery_size * (
                        self.env.battery_margins[1] - observation[-1] - 0.25) * 60 / self.env.umar_model.interval
                if margin > -consumption:
                    action = np.append(action, min(-consumption, self.env.battery_max_power))
                else:
                    action = np.append(action,
                                       min(max(margin, -self.env.battery_max_power), self.env.battery_max_power))

        # Return the action
        return action

    def run(self, sequence, goal_number=None, init_temp=None, render=True):
        """
        Run the agent over a sequence of inputs

        Args:
            sequence: a sequence of indices to get the data. If None, a random sequence is used
            render:   whether to plot the result or not
        """

        # Reset the environment to get the first observation
        observation = self.env.reset(sequence, goal_number, init_temp)
        temperatures = [self.env.scale_back_temperatures(observation[2])]
        # temperatures = [20]
        actions = []
        if self.her:
            observation = observation['observation']
            goal = self.env.desired_goals[self.env.goal_number]
        else:
            goal = None

        # Reinitialize the parameters
        self.dones = np.array([False] * len([self.rooms]))

        done = False
        cumul_reward = 0
        cumul_price_reward = 0
        cumul_comfort_reward = 0
        length = 0
        k = 0

        # Iterate until the episode is over (i.e. the sequence is finished)
        while not done:
            # Choose an action
            action = self.take_decision(observation)
            k = k + 1
            action = torch.Tensor(action)
            # Take a step
            observation, reward, done, info, price_reward, comfort_reward = self.env.step(action, rule_based=True)
            temperatures.append(self.env.scale_back_temperatures(observation[2]))
            actions.append(action)
            if self.her:
                observation = observation['observation']
            # Recall the rewards
            cumul_reward += reward
            cumul_price_reward += price_reward
            cumul_comfort_reward += comfort_reward
            length += 1
            
        # Some modifications needed to be compatible with the 'render' function
        self.env.current_step -= 1
        if self.env.battery:
            self.env.battery_powers.pop()
            self.env.battery_soc.pop()
        self.env.electricity_imports.pop()

        # If wanted, plot the result
        if render:
            self.env.render()
        demand = self.env.electricity_imports
        return cumul_reward, length, temperatures, actions, cumul_price_reward, cumul_comfort_reward, demand


class Agent:

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, compute_reward,
                 sub: bool = False, load_best: bool = False, load_best_comfort: bool = False,
                 load_best_price: bool = False):

        self.save_path = agent_kwargs["save_path"]
        self.name = agent_kwargs["name"]
        self.algorithm = agent_kwargs["algorithm"]
        self.save_name = os.path.join(self.save_path, self.algorithm, self.name)
        self.n_envs = agent_kwargs["n_envs"]
        self.normalizing = agent_kwargs["normalizing"]
        self.gamma = agent_kwargs["gamma"]
        self.vf_loss_coef = agent_kwargs["vf_loss_coef"]
        self.ent_coef = agent_kwargs["ent_coef"]
        self.learning_rate = agent_kwargs["learning_rate"]
        self.threshold_length = model_kwargs["threshold_length"]
        self.her = agent_kwargs["her"]
        self.goal_selection_strategy = agent_kwargs["goal_selection_strategy"]
        self.n_sampled_goal = agent_kwargs["n_sampled_goal"]
        self.simple_env = agent_kwargs["simple_env"]
        self.n_eval_episodes = agent_kwargs["n_eval_episodes"]
        self.n_autoregression = model_kwargs["n_autoregression"]

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.isdir(os.path.join(self.save_path, self.algorithm)):
            os.mkdir(os.path.join(self.save_path, self.algorithm))
        if not os.path.isdir(self.save_name):
            os.mkdir(self.save_name)

        print("\nPreparing the agent")
        if self.simple_env:
            umar_model, battery_model = prepare_toy_models(data_kwargs, model_kwargs, agent_kwargs)

            # Prepare the environments using multiprocessing
            env = ToyEnv(umar_model=umar_model,
                         battery_model=battery_model,
                         agent_kwargs=agent_kwargs,
                         compute_reward=compute_reward)

        else:
            # Prepare the battery and UMAR models
            umar_model, battery_model = prepare_models(data_kwargs, model_kwargs, agent_kwargs)

            # Prepare the environments using multiprocessing
            env = UMAREnv(umar_model=umar_model,
                          battery_model=battery_model,
                          agent_kwargs=agent_kwargs,
                          compute_reward=compute_reward)

        # Sanity check that the environment works
        print("Checking the environment...")
        check_env(env)

        # Create subprocesses
        if not self.her:
            print("Creating the subprocesses...")
            if sub:
                env = make_vec_env(lambda: env, n_envs=self.n_envs, vec_env_cls=SubprocVecEnv)
            else:
                env = make_vec_env(lambda: env, n_envs=self.n_envs)
            print("Ready!")
            # Wrap it to check for NaNs and to normalize observations and rewards
            env = VecCheckNan(env, raise_exception=True)

        # Save it
        self.env = env

        self.rewards = []
        self.comfort_violations = []
        self.prices = []

        self.test_sequences = None
        self.init_temps = None

        try:
            with open(os.path.join(self.save_name, "best_stats.txt"), "rb") as fp:
                self.best_mean_reward = pickle.load(fp)
                self.best_mean_comfort = pickle.load(fp)
                self.best_mean_prices = pickle.load(fp)
        except:
            self.best_mean_reward = None
            self.best_mean_comfort = None
            self.best_mean_prices = None

        print("Creating the evaluation environment...")
        # Separate evaluation environment, this time only one is needed
        if self.simple_env:
            eval_env = ToyEnv(umar_model=umar_model,
                              battery_model=battery_model,
                              agent_kwargs=agent_kwargs,
                              compute_reward=compute_reward)
        else:
            eval_env = UMAREnv(umar_model=umar_model,
                               battery_model=battery_model,
                               agent_kwargs=agent_kwargs,
                               compute_reward=compute_reward)

        if not self.her:
            eval_env = make_vec_env(lambda: eval_env, n_envs=1)
            # Again using wrappers
            eval_env = VecCheckNan(eval_env, raise_exception=True)
            if self.normalizing:
                eval_env = VecNormalize(eval_env)

        self.eval_env = eval_env

        self.rbagent = RBAgent(data_kwargs=data_kwargs,
                               model_kwargs=model_kwargs,
                               agent_kwargs=agent_kwargs,
                               compute_reward=compute_reward,
                               umar_model=umar_model,
                               battery_model=battery_model)

        self.bangbang = BangBangAgent(data_kwargs=data_kwargs,
                                      model_kwargs=model_kwargs,
                                      agent_kwargs=agent_kwargs,
                                      compute_reward=compute_reward,
                                      umar_model=umar_model,
                                      battery_model=battery_model)

        self.unavoidable = UnavoidableAgent(data_kwargs=data_kwargs,
                                            model_kwargs=model_kwargs,
                                            agent_kwargs=agent_kwargs,
                                            compute_reward=compute_reward,
                                            umar_model=umar_model,
                                            battery_model=battery_model)

        try:
            with open(os.path.join(self.save_name, "rb_statistics.txt"), "rb") as fp:
                self.rb_rewards = pickle.load(fp)
                self.rb_comfort_violations = pickle.load(fp)
                self.rb_prices = pickle.load(fp)
                self.test_sequences = pickle.load(fp)
                self.init_temps = pickle.load(fp)
            print(f"Rule-based performance on the testing set:")
            print(f"Rewards:              {np.mean(self.rb_rewards):.2f} +/- {np.std(self.rb_rewards):.2f}.")
            print(
                f"Comfort violations:   {np.mean(self.rb_comfort_violations):.2f} +/- {np.std(self.rb_comfort_violations):.2f}.")
            print(f"Prices:               {np.mean(self.rb_prices):.2f} +/- {np.std(self.rb_prices):.2f}.\n")

            with open(os.path.join(self.save_name, "bangbang_statistics.txt"), "rb") as fp:
                self.bangbang_rewards = pickle.load(fp)
                self.bangbang_comfort_violations = pickle.load(fp)
                self.bangbang_prices = pickle.load(fp)
            print(f"Bang-bang performance on the testing set:")
            print(
                f"Rewards:              {np.mean(self.bangbang_rewards):.2f} +/- {np.std(self.bangbang_rewards):.2f}.")
            print(
                f"Comfort violations:   {np.mean(self.bangbang_comfort_violations):.2f} +/- {np.std(self.bangbang_comfort_violations):.2f}.")
            print(
                f"Prices:               {np.mean(self.bangbang_prices):.2f} +/- {np.std(self.bangbang_prices):.2f}.\n")

            with open(os.path.join(self.save_name, "unavoidable_statistics.txt"), "rb") as fp:
                self.unavoidable_rewards = pickle.load(fp)
                self.unavoidable_comfort_violations = pickle.load(fp)
                self.unavoidable_prices = pickle.load(fp)
            print(f"Unavoidable performance on the testing set:")
            print(
                f"Rewards:              {np.mean(self.unavoidable_rewards):.2f} +/- {np.std(self.unavoidable_rewards):.2f}.")
            print(
                f"Comfort violations:   {np.mean(self.unavoidable_comfort_violations):.2f} +/- {np.std(self.unavoidable_comfort_violations):.2f}.")
            print(
                f"Prices:               {np.mean(self.unavoidable_prices):.2f} +/- {np.std(self.unavoidable_prices):.2f}.\n")

            evaluate_rb = False

        except:
            evaluate_rb = True

            if self.simple_env:
                indices = np.array(self.eval_env.venv.envs[0].umar_model.test_indices)
                jump = len(indices) // self.n_eval_episodes
                indices = indices[[self.n_autoregression + i * jump for i in range(self.n_eval_episodes)]]
                self.test_sequences = [np.arange(index - self.n_autoregression, index + self.threshold_length) for index
                                       in indices]
                self.init_temps = [np.random.rand(3) * 3 + 21 for _ in range(self.n_eval_episodes)]

        checkpoint_callback = CheckpointCallback(save_freq=agent_kwargs["save_freq"],
                                                 save_path=os.path.join(self.save_name, f"checkpoints"))

        eval_callback = EvalCallback(rbagent=self.rbagent,
                                     bangbang=self.bangbang,
                                     unavoidable=self.unavoidable,
                                     eval_env=self.eval_env,
                                     best_mean_reward=self.best_mean_reward,
                                     best_mean_comfort=self.best_mean_comfort,
                                     best_mean_prices=self.best_mean_prices,
                                     fixed_sequences=True,
                                     sequences=self.test_sequences,
                                     init_temps=self.init_temps,
                                     deterministic=True,
                                     render=False,
                                     n_eval_episodes=self.n_eval_episodes,
                                     all_goals=True,
                                     evaluate_rb=evaluate_rb,
                                     best_model_save_path=os.path.join(self.save_name, f"logs"),
                                     log_path=os.path.join(self.save_name, f"logs"),
                                     eval_freq=int(agent_kwargs["eval_freq"] / agent_kwargs["n_envs"]),
                                     normalizing=self.normalizing)
        if evaluate_rb:
            with open(os.path.join(self.save_name, "rb_statistics.txt"), "wb") as fp:
                pickle.dump(eval_callback.rb_rewards, fp)
                pickle.dump(eval_callback.rb_comfort_violations, fp)
                pickle.dump(eval_callback.rb_prices, fp)
                pickle.dump(eval_callback.sequences, fp)
                pickle.dump(eval_callback.init_temps, fp)
            self.rb_rewards = eval_callback.rb_rewards
            self.rb_comfort_violations = eval_callback.rb_comfort_violations
            self.rb_prices = eval_callback.rb_prices
            self.test_sequences = eval_callback.sequences

            with open(os.path.join(self.save_name, "bangbang_statistics.txt"), "wb") as fp:
                pickle.dump(eval_callback.bangbang_rewards, fp)
                pickle.dump(eval_callback.bangbang_comfort_violations, fp)
                pickle.dump(eval_callback.bangbang_prices, fp)
            self.bangbang_rewards = eval_callback.bangbang_rewards
            self.bangbang_comfort_violations = eval_callback.bangbang_comfort_violations
            self.bangbang_prices = eval_callback.bangbang_prices

            with open(os.path.join(self.save_name, "unavoidable_statistics.txt"), "wb") as fp:
                pickle.dump(eval_callback.unavoidable_rewards, fp)
                pickle.dump(eval_callback.unavoidable_comfort_violations, fp)
                pickle.dump(eval_callback.unavoidable_prices, fp)
            self.unavoidable_rewards = eval_callback.unavoidable_rewards
            self.unavoidable_comfort_violations = eval_callback.unavoidable_comfort_violations
            self.unavoidable_prices = eval_callback.unavoidable_prices

        if self.normalizing:
            save_vecnormalize = SaveVecNormalizeCallback(save_freq=agent_kwargs["save_freq"],
                                                         save_path=os.path.join(self.save_name, f"vecnormalize"))

        compare_rbagent = CompareAgentCallback(eval_env=self.eval_env,
                                               rbagent=self.rbagent,
                                               bangbang=self.bangbang,
                                               unavoidable=self.unavoidable,
                                               n_eval_episodes=1,
                                               sequences=self.test_sequences if self.simple_env else None,
                                               log_path=os.path.join(self.save_name, f"logs"),
                                               eval_freq=int(agent_kwargs["eval_freq"] / agent_kwargs["n_envs"]) * 2,
                                               normalizing=self.normalizing)

        # Create the callback list
        if self.normalizing:
            self.callbacks = CallbackList([checkpoint_callback, eval_callback,
                                           compare_rbagent, save_vecnormalize])
        else:
            self.callbacks = CallbackList([checkpoint_callback, eval_callback,
                                           compare_rbagent])

        # Define the model
        self.model = None
        self.load_model(agent_kwargs=agent_kwargs, load_best=load_best, load_best_comfort=load_best_comfort,
                        load_best_price=load_best_price)

    def load_model(self, agent_kwargs, load_best: bool = False, load_best_comfort: bool = False,
                   load_best_price: bool = False):
        """
        Function to load an agent. If None is found, a new one is created
        """

        # First try to load the environment statistics of normalization (they are running stats, you
        # should not reset them when loading a trained model)
        if not self.her:
            if self.normalizing:
                try:
                    self.env = VecNormalize.load(os.path.join(self.save_name, "vecnormalize", "vecnormalize.pkl"),
                                                 self.env.venv)
                    print("Environment normalizing stats loaded")
                    print("\n Warning: if you want to load the best model, you have to check that the"
                          " right normalization statistics are also loaded\n")
                except:
                    if not isinstance(self.env, VecNormalize):
                        self.env = VecNormalize(self.env)
                    else:
                        pass
                    print("Environment created")

        # Define the algorithm used
        minibatches = False
        if self.algorithm == "A2C":
            algo = A2C
        elif self.algorithm == "ACER":
            algo = ACER
        elif self.algorithm == "ACKTR":
            algo = ACKTR
        elif self.algorithm == "DDPG":
            algo = DDPG
        elif self.algorithm == "DQN":
            algo = DQN
        elif self.algorithm == "HER":
            algo = HER
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "GAIL":
            algo = GAIL
        elif self.algorithm == "PPO1":
            algo = PPO1
            minibatches = True
        elif self.algorithm == "PPO2":
            algo = PPO2
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "SAC":
            algo = SAC
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "TD3":
            algo = TD3
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "TRPO":
            algo = TRPO
        else:
            raise ValueError(f"The algorithm {self.algorithm} is not implemented in stable-baselines")

        # Try to load an agent
        print("Trying to load an agent...")
        try:
            if self.her:
                policy_kwargs = dict(layers=agent_kwargs["pi_layers"])
                self.model = HER.load(os.path.join(self.save_name, self.name), model_class=algo,
                                      policy_kwargs=policy_kwargs)
            else:
                policy_kwargs = dict(n_lstm=agent_kwargs["lstm_size"],
                                     extraction_size=agent_kwargs["extraction_size"],
                                     vf_layers=agent_kwargs["vf_layers"],
                                     pi_layers=agent_kwargs["pi_layers"])

                if load_best:
                    self.model = algo.load(os.path.join(self.save_name, "logs", "best_model"),
                                           policy=CustomLSTMPolicy, policy_kwargs=policy_kwargs)
                elif load_best_comfort:
                    self.model = algo.load(os.path.join(self.save_name, "logs", "best_model_comfort"),
                                           policy=CustomLSTMPolicy, policy_kwargs=policy_kwargs)
                elif load_best_price:
                    self.model = algo.load(os.path.join(self.save_name, "logs", "best_model_price"),
                                           policy=CustomLSTMPolicy, policy_kwargs=policy_kwargs)
                else:
                    self.model = algo.load(os.path.join(self.save_name, self.name),
                                           policy=CustomLSTMPolicy, policy_kwargs=policy_kwargs)

            # When loaded, we need to set the right environment
            self.model.set_env(self.env)
            
            with open(os.path.join(self.save_name, "statistics.txt"), "rb") as fp:
                self.rewards = pickle.load(fp)
                self.comfort_violations = pickle.load(fp)
                self.prices = pickle.load(fp)

            success = True

        # When no model was found
        except:
            # Two cases, because PPO2 requires a special argument: the number of environments and the number
            # must be a multiple of the number of minibatches
            if minibatches:
                if self.her:
                    policy_kwargs = dict(layers=agent_kwargs["pi_layers"])
                    self.model = HER(CustomMlpPolicy, self.env, algo, n_sampled_goal=self.n_sampled_goal,
                                     goal_selection_strategy=self.goal_selection_strategy,
                                     verbose=0, policy_kwargs=policy_kwargs)
                else:
                    policy_kwargs = dict(n_lstm=agent_kwargs["lstm_size"],
                                         extraction_size=agent_kwargs["extraction_size"],
                                         vf_layers=agent_kwargs["vf_layers"],
                                         pi_layers=agent_kwargs["pi_layers"])
                    self.model = algo(CustomLSTMPolicy, self.env, nminibatches=self.n_envs, policy_kwargs=policy_kwargs,
                                      gamma=self.gamma, learning_rate=self.learning_rate, n_steps=self.threshold_length,
                                      vf_coef=self.vf_loss_coef, ent_coef=self.ent_coef,
                                      tensorboard_log=os.path.join(self.save_path, "tensorboard", self.name))
                success = False

            # Otherwise, initialize an agent with the wanted algorithm
            else:
                policy_kwargs = dict(n_lstm=agent_kwargs["lstm_size"],
                                     extraction_size=agent_kwargs["extraction_size"],
                                     vf_layers=agent_kwargs["vf_layers"],
                                     pi_layers=agent_kwargs["pi_layers"])
                self.model = algo(CustomLSTMPolicy, self.env, policy_kwargs=policy_kwargs,
                                  tensorboard_log=os.path.join(self.save_path, "tensorboard", self.name))
                success = False

        # Informative print
        if success:
            print("Found!")
        else:
            print("Nothing found!")

    def train(self, total_timesteps):
        """
        Function to train the agent for a certain number of steps

        Args:
            total_timesteps: number of time steps to train
        """

        # Learn the wanted time steps
        print("\nTraining starts!")

        # Clean the caches
        self.callbacks.callbacks[1].rewards = []
        self.callbacks.callbacks[1].comfort_violations = []
        self.callbacks.callbacks[1].prices = []

        self.model.learn(total_timesteps=total_timesteps,
                         callback=self.callbacks)

        # Save the progress
        print("Saving...")

        self.rewards += self.callbacks.callbacks[1].rewards
        self.comfort_violations += self.callbacks.callbacks[1].comfort_violations
        self.prices += self.callbacks.callbacks[1].prices
        self.best_mean_reward = self.callbacks.callbacks[1].best_mean_reward
        self.best_mean_prices = self.callbacks.callbacks[1].best_mean_prices
        self.best_mean_comfort = self.callbacks.callbacks[1].best_mean_comfort

        with open(os.path.join(self.save_name, "statistics.txt"), "wb") as fp:
            pickle.dump(self.rewards, fp)
            pickle.dump(self.comfort_violations, fp)
            pickle.dump(self.prices, fp)

        with open(os.path.join(self.save_name, "best_stats.txt"), "wb") as fp:
            pickle.dump(self.best_mean_reward, fp)
            pickle.dump(self.best_mean_comfort, fp)
            pickle.dump(self.best_mean_prices, fp)

        self.model.save(os.path.join(self.save_name, self.name))
        if not self.her and self.normalizing:
            self.env.save(os.path.join(self.save_name, "vecnormalize", "vecnormalize.pkl"))

    def plot_training(self, plot_std: bool = True, save_name=None, rewards_lims=None,
                      comfort_lims=None, text_scale=1.25, print_: bool = True):
        """
        Function to plot the performance of the agent on the testing set during training.
        Plots the mean and optionally one std of the rewards obtained, the comfort violations
        and the energy costs along training.
        On the second, a zoom on the interesting part is provided, where the limits
        can be tuned by hand

        Args:
            plot_std:       Flag to plot one standard deviation from the mean as well
            save_name:      Name to save the figure
            rewards_lims:   Limits to zoom the reward plot
            comfort_lims:   Limits to zoom the comfort
            text_scale:     Scale the texts
            print_:         Flag to print out some important numbers of the graph
        """

        fig, ax = plt.subplots(3, 2, figsize=(25, 25), sharex=True)

        # Compute all needed information for the plots
        mean_rewards = np.array([np.mean(x) for x in self.rewards])
        std_rewards = np.array([np.std(x) for x in self.rewards])
        mean_comfort_violations = np.array([np.mean(x) for x in self.comfort_violations])
        std_comfort_violations = np.array([np.std(x) for x in self.comfort_violations])
        mean_prices = np.array([np.mean(x) for x in self.prices])
        std_prices = np.array([np.std(x) for x in self.prices])
        xrange = np.arange(len(mean_rewards))

        if print_:
            print(f"Best Performances:\n")
            print("Unavoidable:")
            print(f"\nComfort violations:   {np.mean(self.unavoidable_comfort_violations):.2f}")
            print(f"Total benefits/costs: {np.mean(self.unavoidable_prices):.2f}")
            print(f"__________________________\n\nBang-bang:")
            print(f"\nReward:             {np.mean(self.bangbang_rewards):.2f}")
            print(f"Comfort violations:   {np.mean(self.bangbang_comfort_violations):.2f}")
            print(f"Total benefits/costs: {np.mean(self.bangbang_prices):.2f}")
            print(f"__________________________\n\nRule-based:")
            print(f"\nReward:             {np.mean(self.rb_rewards):.2f}")
            print(f"Comfort violations:   {np.mean(self.rb_comfort_violations):.2f}")
            print(f"Total benefits/costs: {np.mean(self.rb_prices):.2f}")
            print(f"__________________________\n\nBest agent along training:")
            print(f"\nReward:             {np.max(mean_rewards):.2f}")
            print(f"Comfort violations:   {np.min(mean_comfort_violations):.2f}")
            try:
                print(f"Total benefits/costs: {np.min(mean_prices[200:]):.2f}\n")
            except ValueError:
                pass

        # Loop over the columns (the same thing is plotted, the zoom and the std are the only differences)
        for j in range(2):
            if plot_std & (j == 0):
                ax[0, j].fill_between(xrange, np.array([np.mean(self.rb_rewards)] * len(xrange)) -
                                      np.array([np.std(self.rb_rewards)] * len(xrange)),
                                      np.array([np.mean(self.rb_rewards)] * len(xrange)) +
                                      np.array([np.std(self.rb_rewards)] * len(xrange)), color="orange", alpha=0.1)
                ax[0, j].fill_between(xrange, mean_rewards - std_rewards, mean_rewards + std_rewards, color="blue",
                                      alpha=0.1)
            ax[0, j].plot(xrange, [np.mean(self.rb_rewards)] * len(xrange), label="Rule-based", color="orange")
            ax[0, j].plot(xrange, [np.mean(self.bangbang_rewards)] * len(xrange), label="Bang-bang", color="red")
            ax[0, j].plot(xrange, mean_rewards, color="blue")

            if plot_std & (j == 0):
                ax[1, j].fill_between(xrange, np.array([np.mean(self.rb_comfort_violations)] * len(xrange)) -
                                      np.array([np.std(self.rb_comfort_violations)] * len(xrange)),
                                      np.array([np.mean(self.rb_comfort_violations)] * len(xrange)) +
                                      np.array([np.std(self.rb_comfort_violations)] * len(xrange)), color="orange",
                                      alpha=0.1)
                ax[1, j].fill_between(xrange, mean_comfort_violations - std_comfort_violations,
                                      mean_comfort_violations + std_comfort_violations, color="blue", alpha=0.1)
            ax[1, j].plot(xrange, [np.mean(self.rb_comfort_violations)] * len(xrange), label="Rule-based",
                          color="orange")
            ax[1, j].plot(xrange, [np.mean(self.bangbang_comfort_violations)] * len(xrange), label="Bang-bang",
                          color="red")
            ax[1, j].plot(xrange, [np.mean(self.unavoidable_comfort_violations)] * len(xrange), label="Unavoidable",
                          color="black")
            ax[1, j].plot(xrange, mean_comfort_violations, color="blue")

            if plot_std & (j == 0):
                ax[2, j].fill_between(xrange, np.array([np.mean(self.rb_prices)] * len(xrange)) -
                                      np.array([np.std(self.rb_prices)] * len(xrange)),
                                      np.array([np.mean(self.rb_prices)] * len(xrange)) +
                                      np.array([np.std(self.rb_prices)] * len(xrange)), color="orange", alpha=0.1)
                ax[2, j].fill_between(xrange, mean_prices - std_prices, mean_prices + std_prices, color="blue",
                                      alpha=0.1)
            ax[2, j].plot(xrange, [np.mean(self.rb_prices)] * len(xrange), label="Rule-based", color="orange")
            ax[2, j].plot(xrange, [np.mean(self.bangbang_prices)] * len(xrange), label="Bang-bang", color="red")
            ax[2, j].plot(xrange, [np.mean(self.unavoidable_prices)] * len(xrange), label="Unavoidable", color="black")
            ax[2, j].plot(xrange, mean_prices, color="blue")

        # All relevant labels
        ax[0, 0].set_ylabel("Reward", size=20 * text_scale)
        ax[1, 0].set_ylabel("Violation ($^\circ$C*15min)", size=20 * text_scale)
        ax[2, 0].set_ylabel("Cost", size=20 * text_scale)
        ax[2, 0].set_xlabel("Evaluation number", size=20 * text_scale)
        ax[2, 1].set_xlabel("Evaluation number", size=20 * text_scale)

        # Custom limits, can be tuned with parameters
        if rewards_lims is None:
            ax[0, 1].set_ylim(np.max(mean_rewards) - 0.2 * np.std(self.rb_rewards),
                              max(np.max(mean_rewards), np.mean(self.rb_rewards) + 5))
        else:
            ax[0, 1].set_ylim(rewards_lims[0], rewards_lims[1])

        if comfort_lims is None:
            ax[1, 1].set_ylim(min(np.min(mean_comfort_violations), np.mean(self.rb_comfort_violations)) - 1,
                              max(np.mean(self.rb_comfort_violations) + 3, np.min(mean_comfort_violations) + 1))
        else:
            ax[1, 1].set_ylim(comfort_lims[0], comfort_lims[1])

        # Put the right titles, the legends and make the labels bigger
        titles = ["Rewards", "Comfort violations", "Energy (costs)"]
        for i in range(3):
            for j in range(2):
                ax[i, j].legend(prop={'size': 15 * text_scale})
                ax[i, j].set_title(titles[i], size=25 * text_scale)
                ax[i, j].tick_params(axis="x", which="major", labelsize=15 * text_scale)
                ax[i, j].tick_params(axis="y", which="major", labelsize=15 * text_scale)

        # Wrap everything up
        plt.tight_layout()

        # Define the name using the agent's name if None is given
        if save_name is None:
            save_name = self.name + "_Training"
        plt.savefig(os.path.join(FIGURES_SAVE_PATH, save_name + ".svg"), format='svg')

        plt.show()


class RBAgent_old_three:
    """
    Very basic Rule-Based agent heating or cooling only as soon as the bounds are reached.
    """

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, umar_model=None, battery_model=None,
                 compute_reward=compute_reward, room = '272'):

        self.simple_env = agent_kwargs["simple_env"]
        print("\nPreparing the rule-based agent")

        if room == '274':
            base_indices = [1, 4, 7, 8, 9, 10, 11]
            effect_indices = [2, 4, 5, 6]
            all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 274', 'Temperature 273', 'Month sin',
                          'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 274', 'Case']
            all_outputs = ['Temperature 274']
        elif room == '272':
            base_indices = [1, 3, 7, 8, 9, 10, 11]
            effect_indices = [2, 3, 5, 6]
            all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
                          'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 272', 'Case']
            all_outputs = ['Temperature 272']

        if (umar_model is None) | (battery_model is None):
            # Prepare the battery and UMAR models
            # umar_model, battery_model = prepare_models(data_kwargs, model_kwargs)
            if self.simple_env:
                umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                              Y_columns=all_outputs,
                                                              X_columns=all_inputs,
                                                              base_indices=base_indices,
                                                              effect_indices=effect_indices,
                                                              room=room)
            else:
                umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                              Y_columns=all_outputs,
                                                              X_columns=all_inputs,
                                                              base_indices=base_indices,
                                                              effect_indices=effect_indices,
                                                              room=room)
        # Prepare the environment
        # env = UMAREnv(umar_model=umar_model,
        #               battery_model=battery_model,
        #               agent_kwargs=agent_kwargs,
        #               compute_reward=compute_reward)
        if self.simple_env:
            env = ToyEnv(umar_model=umar_model,
                         battery_model=None,
                         agent_kwargs=agent_kwargs,
                         compute_reward=compute_reward)
        else:
            env = UMAREnv(umar_model=umar_model,
                          battery_model=battery_model,
                          agent_kwargs=agent_kwargs,
                          compute_reward=compute_reward)

        self.small_model = agent_kwargs["small_model"]
        self.compute_reward = compute_reward

        self.her = agent_kwargs["her"]

        # Save the environment
        self.env = env
        self.half_degree_scaled = (np.mean(self.env.scaled_temp_bounds, axis=0) - self.env.scaled_temp_bounds[0, :]) / 2

    def take_decision(self, observation, goal=None):
        """
        Decision making function: heat when the temperature drops below the lower bound and cool when it
        goes above the upper bound.

        Args:
            observation: the current observation of the environment from which to take the decision
        """
        if goal is None:
            goal = (self.env.scaled_temp_bounds[0, :] + self.env.scaled_temp_bounds[1, :]) / 2

        if self.small_model:
            index = 8
        else:
            index = 8


        action = np.array((goal - self.half_degree_scaled >
                           observation[-(index + 2 * len(self.env.rooms)):
                                       -(index + len(self.env.rooms))]) * 1 * 0.8 + 0.1)
        

        # Decide what to do of the battery
        if self.env.battery:
            electricity_consumption = self.env.umar_model.min_["Electricity total consumption"] + (
                    observation[self.env.elec_column] - 0.1) * (
                                                  self.env.umar_model.max_["Electricity total consumption"] -
                                                  self.env.umar_model.min_[
                                                      "Electricity total consumption"]) / 0.8
            energies = observation[-(3 + len(self.env.rooms)):-3]
            energy = sum([self.env.umar_model.min_[f"Energy room {room}"] + (energies[i] - 0.1) * (
                    self.env.umar_model.max_[f"Energy room {room}"] - self.env.umar_model.min_[
                f"Energy room {room}"]) / 0.8 for i, room in enumerate(self.env.rooms)])

            consumption = energy / self.env.COP + electricity_consumption

            if consumption > 0:
                # Transform in kWh, then in kW over 15 min
                margin = 0.96 / self.env.battery_size * (
                            observation[-1] - self.env.battery_margins[0] - 0.25) * 60 / self.env.umar_model.interval
                if margin > consumption:
                    action = np.append(action, max(-consumption, -self.env.battery_max_power))
                else:
                    action = np.append(action,
                                       min(max(-margin, -self.env.battery_max_power), self.env.battery_max_power))
            else:
                margin = 0.96 / self.env.battery_size * (
                            self.env.battery_margins[1] - observation[-1] - 0.25) * 60 / self.env.umar_model.interval
                if margin > -consumption:
                    action = np.append(action, min(-consumption, self.env.battery_max_power))
                else:
                    action = np.append(action,
                                       min(max(margin, -self.env.battery_max_power), self.env.battery_max_power))

        # Return the action
        return action

    def run(self, sequence, goal_number=None, init_temp=None, render=True):
        """
        Run the agent over a sequence of inputs

        Args:
            sequence: a sequence of indices to get the data. If None, a random sequence is used
            render:   whether to plot the result or not
        """

        # Reset the environment to get the first observation
        # observation = self.env.reset(sequence, goal_number)
        observation = self.env.reset(sequence, goal_number, init_temp)
        temperatures = [self.env.scale_back_temperatures(observation[2])]
        # temperatures = [[20]]
        actions = []
        if self.her:
            observation = observation['observation']
            goal = self.env.desired_goals[self.env.goal_number]
        else:
            goal = None

        done = False
        cumul_reward = 0
        length = 0
        k = 0
        # Iterate until the episode is over (i.e. the sequence is finished)
        while not done:
            # Choose an action
            action = self.take_decision(observation, goal)
            k = k + 1
            action = torch.Tensor(action)
            # Take a step
            observation, reward, done, info = self.env.step(action, rule_based=True)
            temperatures.append(self.env.scale_back_temperatures(observation[2]))
            actions.append(action)
            if self.her:
                observation = observation['observation']
            # Recall the rewards
            cumul_reward += reward
            length += 1

        # Some modifications needed to be compatible with the 'render' function
        self.env.current_step -= 1
        if self.env.battery:
            self.env.battery_powers.pop()
            self.env.battery_soc.pop()
        self.env.electricity_imports.pop()

        # If wanted, plot the result
        if render:
            self.env.render()

        return cumul_reward, length, temperatures, actions


class Agent_old_three:

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, compute_reward=compute_reward, sub=False):

        self.save_path = agent_kwargs["save_path"]
        self.name = agent_kwargs["name"]
        self.algorithm = agent_kwargs["algorithm"]
        self.save_name = os.path.join(self.save_path, self.algorithm, self.name)
        self.n_envs = agent_kwargs["n_envs"]
        self.normalizing = agent_kwargs["normalizing"]
        self.gamma = agent_kwargs["gamma"]
        self.her = agent_kwargs["her"]
        self.goal_selection_strategy = agent_kwargs["goal_selection_strategy"]
        self.n_sampled_goal = agent_kwargs["n_sampled_goal"]

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.isdir(os.path.join(self.save_path, self.algorithm)):
            os.mkdir(os.path.join(self.save_path, self.algorithm))
        if not os.path.isdir(self.save_name):
            os.mkdir(self.save_name)

        if not tf.test.is_gpu_available():
            self.n_envs = 1

        print("\nPreparing the agent")
        # Prepare the battery and UMAR models
        umar_model, battery_model = prepare_models(data_kwargs, model_kwargs, agent_kwargs)

        # Prepare the environments using multiprocessing
        env = UMAREnv(umar_model=umar_model,
                      battery_model=battery_model,
                      agent_kwargs=agent_kwargs,
                      compute_reward=compute_reward)

        # Sanity check that the environment works
        print("Checking the environment...")
        check_env(env)

        # Create subprocesses
        if not self.her:
            print("Creating the subprocesses...")
            if sub:
                env = make_vec_env(lambda: env, n_envs=self.n_envs, vec_env_cls=SubprocVecEnv)
            else:
                env = make_vec_env(lambda: env, n_envs=self.n_envs)
            print("Ready!")
            # Wrap it to check for NaNs and to normalize observations and rewards
            env = VecCheckNan(env, raise_exception=True)

        # Save it
        self.env = env

        self.mean_rewards = []
        self.std_rewards = []
        self.test_sequences = None

        print("Creating the evaluation environment...")
        # Separate evaluation environment, this time only one is needed
        eval_env = UMAREnv(umar_model=umar_model,
                           battery_model=battery_model,
                           agent_kwargs=agent_kwargs,
                           compute_reward=compute_reward)

        if not self.her:
            eval_env = make_vec_env(lambda: eval_env, n_envs=1)
            # Again using wrappers
            eval_env = VecCheckNan(eval_env, raise_exception=True)
            if self.normalizing:
                eval_env = VecNormalize(eval_env)

        self.eval_env = eval_env

        try:
            with open(os.path.join(self.save_name, "rb_rewards.txt"), "rb") as fp:
                self.mean_rb_rewards = pickle.load(fp)
                self.std_rb_rewards = pickle.load(fp)
                self.test_sequences = pickle.load(fp)
            print(
                f"Rule-based performance on the testing set: {self.mean_rb_rewards:.2f} +/- {self.std_rb_rewards:.2f}.")
            evaluate_rb = False

        except:
            evaluate_rb = True

        checkpoint_callback = CheckpointCallback(save_freq=agent_kwargs["save_freq"],
                                                 save_path=os.path.join(self.save_name, f"checkpoints"))

        self.rbagent = RBAgent(data_kwargs=data_kwargs,
                               model_kwargs=model_kwargs,
                               agent_kwargs=agent_kwargs,
                               compute_reward=compute_reward,
                               umar_model=umar_model,
                               battery_model=battery_model)

        eval_callback = EvalCallback(rbagent=self.rbagent,
                                     eval_env=self.eval_env,
                                     fixed_sequences=True,
                                     sequences=self.test_sequences,
                                     deterministic=True,
                                     render=False,
                                     n_eval_episodes=100,
                                     all_goals=True,
                                     evaluate_rb=evaluate_rb,
                                     best_model_save_path=os.path.join(self.save_name, f"logs"),
                                     log_path=os.path.join(self.save_name, f"logs"),
                                     eval_freq=int(agent_kwargs["eval_freq"] / agent_kwargs["n_envs"]),
                                     normalizing=self.normalizing)
        if evaluate_rb:
            with open(os.path.join(self.save_name, "rb_rewards.txt"), "wb") as fp:
                pickle.dump(eval_callback.mean_rb_rewards, fp)
                pickle.dump(eval_callback.std_rb_rewards, fp)
                pickle.dump(eval_callback.sequences, fp)
            self.mean_rb_rewards = eval_callback.mean_rb_rewards
            self.std_rb_rewards = eval_callback.std_rb_rewards
            self.test_sequences = eval_callback.sequences

        if self.normalizing:
            save_vecnormalize = SaveVecNormalizeCallback(save_freq=agent_kwargs["save_freq"],
                                                         save_path=os.path.join(self.save_name, f"vecnormalize"))

        compare_rbagent = CompareAgentCallback(self.rbagent,
                                               self.eval_env,
                                               n_eval_episodes=1,
                                               log_path=os.path.join(self.save_name, f"logs"),
                                               eval_freq=int(agent_kwargs["eval_freq"] / agent_kwargs["n_envs"])*5,
                                               normalizing=self.normalizing)

        # Create the callback list
        if self.normalizing:
            self.callbacks = CallbackList([checkpoint_callback, eval_callback,
                                           compare_rbagent, save_vecnormalize])
        else:
            self.callbacks = CallbackList([checkpoint_callback, eval_callback,
                                           compare_rbagent])

        # Define the model
        self.model = None
        self.load_model(agent_kwargs=agent_kwargs)

    def load_model(self, agent_kwargs):
        """
        Function to load an agent. If None is found, a new one is created
        """

        # First try to load the environment statistics of normalization (they are running stats, you
        # should not reset them when loading a trained model)
        if not self.her:
            if self.normalizing:
                try:
                    self.env = VecNormalize.load(os.path.join(self.save_name, "vecnormalize", "vecnormalize.pkl"),
                                                 self.env.venv)
                    print("Environment normalizing stats loaded")
                except:
                    if not isinstance(self.env, VecNormalize):
                        self.env = VecNormalize(self.env)
                    else:
                        pass
                    print("Environment created")

        # Define the algorithm used
        minibatches = False
        if self.algorithm == "A2C":
            algo = A2C
        elif self.algorithm == "ACER":
            algo = ACER
        elif self.algorithm == "ACKTR":
            algo = ACKTR
        elif self.algorithm == "DDPG":
            algo = DDPG
        elif self.algorithm == "DQN":
            algo = DQN
        elif self.algorithm == "HER":
            algo = HER
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "GAIL":
            algo = GAIL
        elif self.algorithm == "PPO1":
            algo = PPO1
        elif self.algorithm == "PPO2":
            algo = PPO2
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "SAC":
            algo = SAC
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "TD3":
            algo = TD3
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "TRPO":
            algo = TRPO
        else:
            raise ValueError(f"The algorithm {self.algorithm} is not implemented in stable-baselines")

        # Try to load an agent
        print("Trying to load an agent...")
        try:
            if self.her:
                policy_kwargs = dict(layers=agent_kwargs["pi_layers"])
                self.model = HER.load(os.path.join(self.save_name, self.name), model_class=algo,
                                      policy_kwargs=policy_kwargs)
            else:
                policy_kwargs = dict(n_lstm=agent_kwargs["lstm_size"],
                                     extraction_size=agent_kwargs["extraction_size"],
                                     vf_layers=agent_kwargs["vf_layers"],
                                     pi_layers=agent_kwargs["pi_layers"])
                self.model = algo.load(os.path.join(self.save_name, self.name),
                                       policy=CustomLSTMPolicy, policy_kwargs=policy_kwargs)
            # When loaded, we need to set the right environment
            self.model.set_env(self.env)

            with open(os.path.join(self.save_name, "rewards.txt"), "rb") as fp:
                self.mean_rewards = pickle.load(fp)
                self.std_rewards = pickle.load(fp)

            success = True

        # When no model was found
        except:
            # Two cases, because PPO2 requires a special argument: the number of environments and the number
            # must be a multiple of the number of minibatches
            if minibatches:
                if self.her:
                    policy_kwargs = dict(layers=agent_kwargs["pi_layers"])
                    self.model = HER(CustomMlpPolicy, self.env, algo, n_sampled_goal=self.n_sampled_goal,
                                     goal_selection_strategy=self.goal_selection_strategy,
                                     verbose=0, policy_kwargs=policy_kwargs)
                else:
                    policy_kwargs = dict(n_lstm=agent_kwargs["lstm_size"],
                                     extraction_size=agent_kwargs["extraction_size"],
                                     vf_layers=agent_kwargs["vf_layers"],
                                     pi_layers=agent_kwargs["pi_layers"])
                    self.model = algo(CustomLSTMPolicy, self.env, nminibatches=self.n_envs, policy_kwargs=policy_kwargs,
                                      tensorboard_log=os.path.join(self.save_path, "tensorboard", self.name))
                success = False

            # Otherwise, initialize an agent with the wanted algorithm
            else:
                policy_kwargs = dict(n_lstm=agent_kwargs["lstm_size"],
                                     extraction_size=agent_kwargs["extraction_size"],
                                     vf_layers=agent_kwargs["vf_layers"],
                                     pi_layers=agent_kwargs["pi_layers"])
                self.model = algo(CustomLSTMPolicy, self.env, policy_kwargs=policy_kwargs,
                                  tensorboard_log=os.path.join(self.save_path, "tensorboard", self.name))
                success = False

        # Informative print
        if success:
            print("Found!")
        else:
            print("Nothing found!")

    def train(self, total_timesteps):
        """
        Function to train the agent for a certain number of steps

        Args:
            total_timesteps: number of time steps to train
        """

        # Learn the wanted time steps
        print("\nTraining starts!")

        # Clean the caches
        self.callbacks.callbacks[1].mean_rewards = []
        self.callbacks.callbacks[1].std_rewards = []

        self.model.learn(total_timesteps=total_timesteps,
                         callback=self.callbacks)

        # Save the progress
        print("Saving...")

        self.mean_rewards += self.callbacks.callbacks[1].mean_rewards
        self.std_rewards += self.callbacks.callbacks[1].std_rewards

        with open(os.path.join(self.save_name, "rewards.txt"), "wb") as fp:
            pickle.dump(self.mean_rewards, fp)
            pickle.dump(self.std_rewards, fp)

        self.model.save(os.path.join(self.save_name, self.name))
        if not self.her and self.normalizing:
            self.env.save(os.path.join(self.save_name, "vecnormalize", "vecnormalize.pkl"))


class RBAgent_old_bis:
    """
    Very basic Rule-Based agent heating or cooling only as soon as the bounds are reached.
    """

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, umar_model=None, battery_model=None,
                 compute_reward=compute_reward):

        print("\nPreparing the rule-based agent")
        if (umar_model is None) | (battery_model is None):
            # Prepare the battery and UMAR models
            umar_model, battery_model = prepare_models(data_kwargs, model_kwargs)

        # Prepare the environment
        env = UMAREnv(umar_model=umar_model,
                      battery_model=battery_model,
                      agent_kwargs=agent_kwargs,
                      compute_reward=compute_reward)

        self.small_model = agent_kwargs["small_model"]
        self.compute_reward = compute_reward

        # Save the environment
        self.env = env

    def take_decision(self, observation):
        """
        Decision making function: heat when the temperature drops below the lower bound and cool when it
        goes above the upper bound.

        Args:
            observation: the current observation of the environment from which to take the decision
        """

        if self.small_model:
            index = 7
        else:
            index = 8

        # Check in which case we are
        if observation[-3] == 0.9:
            # Heating case: Heat if we are below the lower bound
            action = np.array(((self.env.scaled_temp_bounds[0, :] + self.env.scaled_temp_bounds[1, :]) / 2 >
                               observation[-(index + 2*len(self.env.rooms)):-(index + len(self.env.rooms))]) * 1 * 0.8 + 0.1)
        else:
            # Cooling: cool if we are above the upper bound
            action = np.array(((self.env.scaled_temp_bounds[0, :] + self.env.scaled_temp_bounds[1, :]) / 2 <
                               observation[-(index + 2*len(self.env.rooms)):-(index + len(self.env.rooms))]) * 1 * 0.8 + 0.1)

        # Decide what to do of the battery
        if self.env.battery:
            electricity_consumption = self.env.umar_model.min_["Electricity total consumption"] + (
                        observation[self.env.elec_column] - 0.1) * (self.env.umar_model.max_["Electricity total consumption"] -
                                                          self.env.umar_model.min_[
                                                              "Electricity total consumption"]) / 0.8
            energies = observation[-(3 + len(self.env.rooms)):-3]
            energy = sum([self.env.umar_model.min_[f"Energy room {room}"] + (energies[i] - 0.1) * (
                        self.env.umar_model.max_[f"Energy room {room}"] - self.env.umar_model.min_[
                    f"Energy room {room}"]) / 0.8 for i, room in enumerate(self.env.rooms)])

            consumption = energy/self.env.COP + electricity_consumption

            if consumption > 0:
                # Transform in kWh, then in kW over 15 min
                margin = 0.96 / self.env.battery_size * (observation[-1] - self.env.battery_margins[0] - 0.25) * 60 / self.env.umar_model.interval
                if margin > consumption:
                    action = np.append(action, max(-consumption, -self.env.battery_max_power))
                else:
                    action = np.append(action, min(max(-margin, -self.env.battery_max_power), self.env.battery_max_power))
            else:
                margin = 0.96 / self.env.battery_size * (self.env.battery_margins[1] - observation[-1] - 0.25) * 60 / self.env.umar_model.interval
                if margin > -consumption:
                    action = np.append(action, min(-consumption, self.env.battery_max_power))
                else:
                    action = np.append(action, min(max(margin, -self.env.battery_max_power), self.env.battery_max_power))

        # Return the action
        return action

    def run(self, sequence=None, render=True):
        """
        Run the agent over a sequence of inputs

        Args:
            sequence: a sequence of indices to get the data. If None, a random sequence is used
            render:   whether to plot the result or not
        """

        # Reset the environment to get the first observation
        observation = self.env.reset(sequence)
        done = False
        cumul_reward = 0
        length = 0

        # Iterate until the episode is over (i.e. the sequence is finished)
        while not done:
            # Choose an action
            action = self.take_decision(observation)

            # Take a step
            observation, reward, done, info = self.env.step(action, rule_based=True)

            # Recall the rewards
            cumul_reward += reward
            length += 1

        # Some modifications needed to be compatible with the 'render' function
        self.env.current_step -= 1
        if self.env.battery:
            self.env.battery_powers.pop()
            self.env.battery_soc.pop()
        self.env.electricity_imports.pop()

        # If wanted, plot the result
        if render:
            self.env.render()

        return cumul_reward, length


class Agent_old_bis:

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, compute_reward=compute_reward, sub=False):

        self.save_path = agent_kwargs["save_path"]
        self.name = agent_kwargs["name"]
        self.algorithm = agent_kwargs["algorithm"]
        self.save_name = os.path.join(self.save_path, self.algorithm, self.name)
        self.n_envs = agent_kwargs["n_envs"]
        self.normalizing = agent_kwargs["normalizing"]
        self.gamma = agent_kwargs["gamma"]

        print("\nPreparing the agent")
        # Prepare the battery and UMAR models
        umar_model, battery_model = prepare_models(data_kwargs, model_kwargs, agent_kwargs)

        # Prepare the environments using multiprocessing
        env = UMAREnv(umar_model=umar_model,
                      battery_model=battery_model,
                      agent_kwargs=agent_kwargs,
                      compute_reward=compute_reward)

        # Sanity check that the environment works
        print("Checking the environment...")
        check_env(env)

        # Create subprocesses
        print("Creating the subprocesses...")
        if sub:
            env = make_vec_env(lambda: env, n_envs=self.n_envs, vec_env_cls=SubprocVecEnv)
        else:
            env = make_vec_env(lambda: env, n_envs=self.n_envs)
        print("Ready!")
        # Wrap it to check for NaNs and to normalize observations and rewards
        env = VecCheckNan(env, raise_exception=True)

        # Save it
        self.env = env

        print("Creating the evaluation environment...")
        # Separate evaluation environment, this time only one is needed
        eval_env = UMAREnv(umar_model=umar_model,
                           battery_model=battery_model,
                           agent_kwargs=agent_kwargs,
                           compute_reward=compute_reward)

        eval_env = make_vec_env(lambda: eval_env, n_envs=1)
        # Again using wrappers
        eval_env = VecCheckNan(eval_env, raise_exception=True)
        if self.normalizing:
            eval_env = VecNormalize(eval_env)

        self.eval_env = eval_env

        # Use deterministic actions for evaluation
        checkpoint_callback = CheckpointCallback(save_freq=agent_kwargs["save_freq"],
                                                 save_path=os.path.join(self.save_name, f"checkpoints"))

        eval_callback = EvalCallback(self.eval_env,
                                     fixed_sequences=True,
                                     deterministic=True,
                                     render=False,
                                     n_eval_episodes=50,
                                     best_model_save_path=os.path.join(self.save_name, f"logs"),
                                     log_path=os.path.join(self.save_name, f"logs"),
                                     eval_freq=agent_kwargs["eval_freq"],
                                     normalizing=self.normalizing)

        eval_callback_render = EvalCallback(self.eval_env,
                                            fixed_sequences=False,
                                            deterministic=True,
                                            render=True,
                                            n_eval_episodes=1,
                                            best_model_save_path=os.path.join(self.save_name, f"logs"),
                                            log_path=os.path.join(self.save_name, f"logs"),
                                            eval_freq=agent_kwargs["eval_freq"] * 5,
                                            normalizing=self.normalizing)

        if self.normalizing:
            save_vecnormalize = SaveVecNormalizeCallback(save_freq=agent_kwargs["save_freq"],
                                                     save_path=os.path.join(self.save_name, f"vecnormalize"))

        self.rbagent = RBAgent(data_kwargs=data_kwargs,
                               model_kwargs=model_kwargs,
                               agent_kwargs=agent_kwargs,
                               compute_reward=compute_reward,
                               umar_model=umar_model,
                               battery_model=battery_model)

        compare_rbagent = CompareAgentCallback(self.rbagent,
                                               self.eval_env,
                                               render=True,
                                               n_eval_episodes=1,
                                               log_path=os.path.join(self.save_name, f"logs"),
                                               eval_freq=agent_kwargs["eval_freq"] * 5,
                                               normalizing=self.normalizing)

        # Create the callback list
        if self.normalizing:
            self.callbacks = CallbackList([checkpoint_callback, eval_callback, eval_callback_render,
                                           compare_rbagent, save_vecnormalize])
        else:
            self.callbacks = CallbackList([checkpoint_callback, eval_callback, eval_callback_render, compare_rbagent])

        # Define the model
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Function to load an agent. If None is found, a new one is created
        """

        # First try to load the environment statistics of normalization (they are running stats, you
        # should not reset them when loading a trained model)
        if self.normalizing:
            try:
                self.env = VecNormalize.load(os.path.join(self.save_name, "vecnormalize", "vecnormalize.pkl"),
                                             self.env.venv)
                print("Environment normalizing stats loaded")
            except:
                if not isinstance(self.env, VecNormalize):
                    self.env = VecNormalize(self.env)
                else:
                    pass
                print("Environment created")

        # Define the algorithm used
        minibatches = False
        if self.algorithm == "A2C":
            algo = A2C
        elif self.algorithm == "ACER":
            algo = ACER
        elif self.algorithm == "ACKTR":
            algo = ACKTR
        elif self.algorithm == "DDPG":
            algo = DDPG
        elif self.algorithm == "DQN":
            algo = DQN
        elif self.algorithm == "HER":
            algo = HER
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "GAIL":
            algo = GAIL
        elif self.algorithm == "PPO1":
            algo = PPO1
        elif self.algorithm == "PPO2":
            algo = PPO2
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "SAC":
            algo = SAC
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "TD3":
            algo = TD3
            self.n_envs = 1
            minibatches = True
        elif self.algorithm == "TRPO":
            algo = TRPO
        else:
            raise ValueError(f"The algorithm {self.algorithm} is not implemented in stable-baselines")

        # Try to load an agent
        print("Trying to load an agent...")
        try:
            self.model = algo.load(os.path.join(self.save_name, self.name),
                                   policy=CustomLSTMPolicy)
            # When loaded, we need to set the right environment
            self.model.set_env(self.env)
            success = True

        # When no model was found
        except:
            # Two cases, because PPO2 requires a special argument: the number of environments and the number
            # must be a multiple of the number of minibatches
            if minibatches:
                self.model = algo(CustomLSTMPolicy, self.env, nminibatches=self.n_envs,
                                  tensorboard_log=os.path.join(self.save_path, "tensorboard", self.name))
                success = False

            # Otherwise, initialize an agent with the wanted algorithm
            else:
                self.model = algo(CustomLSTMPolicy, self.env,
                                  tensorboard_log=os.path.join(self.save_path, "tensorboard", self.name))
                success = False

        # Informative print
        if success:
            print("Found!")
        else:
            print("Nothing found!")

    def train(self, total_timesteps):
        """
        Function to train the agent for a certain number of steps

        Args:
            total_timesteps: number of time steps to train
        """

        # Learn the wanted time steps
        print("\nTraining starts!")
        self.model.learn(total_timesteps=total_timesteps,
                         callback=self.callbacks)

        # Save the progress
        print("Saving...")
        self.model.save(os.path.join(self.save_name, self.name))
        if self.normalizing:
            self.env.save(os.path.join(self.save_name, "vecnormalize", "vecnormalize.pkl"))


class RBAgent_old:
    """
    Very basic Rule-Based agent heating or cooling only as soon as the bounds are reached.
    """

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, umar_model=None, battery_model=None,
                 compute_reward=compute_reward):

        print("\nPreparing the rule-based agent")
        if (umar_model is None) | (battery_model is None):
            # Prepare the battery and UMAR models
            umar_model, battery_model = prepare_models(data_kwargs, model_kwargs)

        # Prepare the environment
        env = UMAREnv(umar_model=umar_model,
                      battery_model=battery_model,
                      agent_kwargs=agent_kwargs,
                      compute_reward=compute_reward)

        self.compute_reward = compute_reward

        # Save the environment
        self.env = env

    def take_decision(self, observation):
        """
        Decision making function: heat when the temperature drops below the lower bound and cool when it
        goes above the upper bound.

        Args:
            observation: the current observation of the environment from which to take the decision
        """

        # Check in which case we are
        if observation[-3] == 0.9:
            # Heating case: Heat if we are below the lower bound
            action = np.array((self.env.scaled_temp_bounds[0, :] >
                               observation[-(8 + 2*len(self.env.rooms)):-(8 + len(self.env.rooms))]) * 1 * 0.8 + 0.1)
        else:
            # Cooling: cool if we are above the upper bound
            action = np.array((self.env.scaled_temp_bounds[1, :] <
                               observation[-(8 + 2*len(self.env.rooms)):-(8 + len(self.env.rooms))]) * 1 * 0.8 + 0.1)

        # Decide what to do of the battery
        if self.env.battery:
            electricity_consumption = self.env.umar_model.dataset.min_["Electricity total consumption"] + (
                        observation[self.env.elec_column] - 0.1) * (self.env.umar_model.dataset.max_["Electricity total consumption"] -
                                                          self.env.umar_model.dataset.min_[
                                                              "Electricity total consumption"]) / 0.8
            energies = observation[-(3 + len(self.env.rooms)):-3]
            energy = sum([self.env.umar_model.dataset.min_[f"Energy room {room}"] + (energies[i] - 0.1) * (
                        self.env.umar_model.dataset.max_[f"Energy room {room}"] - self.env.umar_model.dataset.min_[
                    f"Energy room {room}"]) / 0.8 for i, room in enumerate(self.env.rooms)])

            consumption = energy/self.env.COP + electricity_consumption

            if consumption > 0:
                # Transform in kWh, then in kW over 15 min
                margin = 0.96 / self.env.battery_size * (observation[-1] - self.env.battery_margins[0] - 0.25) * 60 / self.env.umar_model.dataset.interval
                if margin > consumption:
                    action = np.append(action, max(-consumption, -self.env.battery_max_power))
                else:
                    action = np.append(action, min(max(-margin, -self.env.battery_max_power), self.env.battery_max_power))
            else:
                margin = 0.96 / self.env.battery_size * (self.env.battery_margins[1] - observation[-1] - 0.25) * 60 / self.env.umar_model.dataset.interval
                if margin > -consumption:
                    action = np.append(action, min(-consumption, self.env.battery_max_power))
                else:
                    action = np.append(action, min(max(margin, -self.env.battery_max_power), self.env.battery_max_power))

            # Return the action
            return action

    def run(self, sequence=None, render=True):
        """
        Run the agent over a sequence of inputs

        Args:
            sequence: a sequence of indices to get the data. If None, a random sequence is used
            render:   whether to plot the result or not
        """

        # Reset the environment to get the first observation
        observation = self.env.reset(sequence)
        done = False
        cumul_reward = 0
        length = 0

        # Iterate until the episode is over (i.e. the sequence is finished)
        while not done:
            # Choose an action
            action = self.take_decision(observation)

            # Take a step
            observation, reward, done, info = self.env.step(action, rule_based=True)

            # Recall the rewards
            cumul_reward += reward
            length += 1

        # Some modifications needed to be compatible with the 'render' function
        self.env.current_step -= 1
        if self.env.battery:
            self.env.battery_powers.pop()
            self.env.battery_soc.pop()
        self.env.electricity_imports.pop()

        # If wanted, plot the result
        if render:
            self.env.render()

        return cumul_reward, length


class Agent_old:

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, compute_reward=compute_reward, sub=False):

        self.save_path = agent_kwargs["save_path"]
        self.name = agent_kwargs["name"]
        self.algorithm = agent_kwargs["algorithm"]
        self.save_name = os.path.join(self.save_path, self.algorithm, self.name)
        self.n_envs = agent_kwargs["n_envs"]
        self.normalizing = agent_kwargs["normalizing"]

        print("\nPreparing the agent")
        # Prepare the battery and UMAR models
        umar_model, battery_model = prepare_models(data_kwargs, model_kwargs)

        # Prepare the environments using multiprocessing
        env = UMAREnv(umar_model=umar_model,
                      battery_model=battery_model,
                      agent_kwargs=agent_kwargs,
                      compute_reward=compute_reward)

        # Sanity check that the environment works
        print("Checking the environment...")
        check_env(env)

        # Create subprocesses
        print("Creating the subprocesses...")
        if sub:
            env = make_vec_env(lambda: env, n_envs=self.n_envs, vec_env_cls=SubprocVecEnv)
        else:
            env = make_vec_env(lambda: env, n_envs=self.n_envs)
        print("Ready!")
        # Wrap it to check for NaNs and to normalize observations and rewards
        env = VecCheckNan(env, raise_exception=True)

        # Save it
        self.env = env

        print("Creating the evaluation environment...")
        # Separate evaluation environment, this time only one is needed
        eval_env = UMAREnv(umar_model=umar_model,
                           battery_model=battery_model,
                           agent_kwargs=agent_kwargs,
                           compute_reward=compute_reward)

        eval_env = make_vec_env(lambda: eval_env, n_envs=1)
        # Again using wrappers
        eval_env = VecCheckNan(eval_env, raise_exception=True)
        if self.normalizing:
            eval_env = VecNormalize(eval_env)

        self.eval_env = eval_env

        # Use deterministic actions for evaluation
        checkpoint_callback = CheckpointCallback(save_freq=agent_kwargs["save_freq"],
                                                 save_path=os.path.join(self.save_name, f"checkpoints"))

        eval_callback = EvalCallback(self.eval_env,
                                     fixed_sequences=True,
                                     deterministic=True,
                                     render=False,
                                     n_eval_episodes=100,
                                     best_model_save_path=os.path.join(self.save_name, f"logs"),
                                     log_path=os.path.join(self.save_name, f"logs"),
                                     eval_freq=agent_kwargs["eval_freq"],
                                     normalizing=self.normalizing)

        eval_callback_render = EvalCallback(self.eval_env,
                                            fixed_sequences=False,
                                            deterministic=True,
                                            render=True,
                                            n_eval_episodes=1,
                                            best_model_save_path=os.path.join(self.save_name, f"logs"),
                                            log_path=os.path.join(self.save_name, f"logs"),
                                            eval_freq=agent_kwargs["eval_freq"] * 5,
                                            normalizing=self.normalizing)

        if self.normalizing:
            save_vecnormalize = SaveVecNormalizeCallback(save_freq=agent_kwargs["save_freq"],
                                                     save_path=os.path.join(self.save_name, f"vecnormalize"))

        self.rbagent = RBAgent(data_kwargs=data_kwargs,
                               model_kwargs=model_kwargs,
                               agent_kwargs=agent_kwargs,
                               compute_reward=compute_reward,
                               umar_model=umar_model,
                               battery_model=battery_model)

        compare_rbagent = CompareAgentCallback(self.rbagent,
                                               self.eval_env,
                                               render=True,
                                               n_eval_episodes=1,
                                               log_path=os.path.join(self.save_name, f"logs"),
                                               eval_freq=agent_kwargs["eval_freq"] * 5,
                                               normalizing=self.normalizing)

        # Create the callback list
        if self.normalizing:
            self.callbacks = [checkpoint_callback, eval_callback, eval_callback_render, compare_rbagent, save_vecnormalize]
        else:
            self.callbacks = [checkpoint_callback, eval_callback, eval_callback_render, compare_rbagent]

        # Define the model
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Function to load an agent. If None is found, a new one is created
        """

        # First try to load the environment statistics of normalization (they are running stats, you
        # should not reset them when loading a trained model)
        if self.normalizing:
            try:
                self.env = VecNormalize.load(os.path.join(self.save_name, "vecnormalize", "vecnormalize.pkl"),
                                             self.env.venv)
                print("Environment normalizing stats loaded")
            except:
                if not isinstance(self.env, VecNormalize):
                    self.env = VecNormalize(self.env)
                else:
                    pass
                print("Environment created")

        # Define the algorithm used
        minibatches = False
        if self.algorithm == "A2C":
            algo = A2C
        elif self.algorithm == "ACER":
            algo = ACER
        elif self.algorithm == "ACKTR":
            algo = ACKTR
        elif self.algorithm == "DDPG":
            algo = DDPG
        elif self.algorithm == "DQN":
            algo = DQN
        elif self.algorithm == "HER":
            algo = HER
            self.n_envs = 1
        elif self.algorithm == "GAIL":
            algo = GAIL
        elif self.algorithm == "PPO1":
            algo = PPO1
        elif self.algorithm == "PPO2":
            algo = PPO2
            minibatches = True
        elif self.algorithm == "SAC":
            algo = SAC
            self.n_envs = 1
        elif self.algorithm == "TD3":
            algo = TD3
            self.n_envs = 1
        elif self.algorithm == "TRPO":
            algo = TRPO
        else:
            raise ValueError(f"The algorithm {self.algorithm} is not implemented in stable-baselines")

        # Try to load an agent
        print("Trying to load an agent...")
        try:
            self.model = algo.load(os.path.join(self.save_name, self.name),
                                   policy=CustomLSTMPolicy)
            # When loaded, we need to set the right environment
            self.model.set_env(self.env)
            print("Found!")

        # When no model was found
        except:
            # Two cases, because PPO2 requires a special argument: the number of environments and the number
            # must be a multiple of the number of minibatches
            if minibatches:
                self.model = algo(CustomLSTMPolicy, self.env, nminibatches=self.n_envs, gamma=self.gamma,
                                  tensorboard_log=os.path.join(self.save_path, "tensorboard", self.name))

            # Otherwise, initialize an agent with the wanted algorithm
            else:
                self.model = algo(CustomLSTMPolicy, self.env, gamma=self.gamma,
                                  tensorboard_log=os.path.join(self.save_path, "tensorboard", self.name))
            print("Nothing found!")


    def train(self, total_timesteps):
        """
        Function to train the agent for a certain number of steps

        Args:
            total_timesteps: number of time steps to train
        """

        # Learn the wanted time steps
        print("\nTraining starts!")
        self.model.learn(total_timesteps=total_timesteps,
                         callback=self.callbacks)

        # Save the progress
        print("Saving...")
        self.model.save(os.path.join(self.save_name, self.name))
        if self.normalizing:
            self.env.save(os.path.join(self.save_name, "vecnormalize", "vecnormalize.pkl"))
