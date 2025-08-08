import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

import gym
from gym import spaces
import kosta.hyperparams as hp
from stable_baselines.common.vec_env import VecEnv

from agents.helpers import prepare_performance_plot, analyze_agent, add_bounds, evaluate_lstm_policy


def compute_temperature_reward(environment, observation, action):
    """
    Small helper function to compute temperature rewards
    """

    reward = 0

    # Get the current temperatures
    temperatures = environment.scale_back_temperatures(
        environment.current_data[environment.n_autoregression + environment.current_step, environment.predictions_columns[:len(environment.rooms)]])

    too_low = np.where(temperatures < environment.temp_bounds[0, :])[0]
    too_high = np.where(temperatures > environment.temp_bounds[1, :])[0]
    right = np.where((temperatures >= environment.temp_bounds[0, :]) & (temperatures <= environment.temp_bounds[1, :]))[
        0]

    reward += np.sum(1 - (
        np.power(temperatures[right] - (environment.temp_bounds[0, right] + environment.temp_bounds[1, right]) / 2, 2)))
    if len(too_low) > 0:
        if observation[-3] > 0.8999:
            reward += np.sum((1 - (action[:-1][too_low] - 0.1) / 0.8) * (
                temperatures[too_low] - environment.temp_bounds[0, too_low])) * environment.temperature_penalty_factor
        elif observation[-3] < 0.1001:
            reward += np.sum(
                (action[:-1][too_low] - 0.1) / 0.8 * (temperatures[too_low] - environment.temp_bounds[0, too_low]))
        else:
            raise ValueError(f"Case has to be 0.1 or 0.9")

    if len(too_high) > 0:
        if observation[-3] > 0.8999:
            reward += np.sum(
                (action[:-1][too_high] - 0.1) / 0.8 * (environment.temp_bounds[1, too_high] - temperatures[too_high]))
        elif observation[-3] < 0.1001:
            reward += np.sum((1 - (action[:-1][too_high] - 0.1) / 0.8) * (
                    environment.temp_bounds[1, too_high] - temperatures[too_high])) * environment.temperature_penalty_factor
        else:
            raise ValueError(f"Case has to be 0.1 or 0.9")

    return reward


def compute_temperature_reward_bis(environment, observation, action):
    """
    Small helper function to compute temperature rewards
    """

    reward = 0

    # Get the current temperatures
    temperatures = environment.scale_back_temperatures(
        environment.current_data[environment.n_autoregression + environment.current_step, environment.predictions_columns[:len(environment.rooms)]])

    too_low = np.where(temperatures < environment.temp_bounds[0, :])[0]
    too_high = np.where(temperatures > environment.temp_bounds[1, :])[0]
    right = np.where((temperatures >= environment.temp_bounds[0, :]) & (temperatures <= environment.temp_bounds[1, :]))[
        0]

    reward += np.sum(1 - (
        np.power(temperatures[right] - (environment.temp_bounds[0, right] + environment.temp_bounds[1, right]) / 2, 2)))
    if len(too_low) > 0:
        if observation[-3] > 0.8999:
            reward += np.sum((1 - (action[:-1][too_low] - 0.1) / 0.8) * (
                temperatures[too_low] - environment.temp_bounds[0, too_low]))
        elif observation[-3] < 0.1001:
            reward += np.sum(
                (action[:-1][too_low] - 0.1) / 0.8 * (temperatures[too_low] - environment.temp_bounds[0, too_low]))
        else:
            raise ValueError(f"Case has to be 0.1 or 0.9")
        reward += np.sum(temperatures[too_low] - environment.temp_bounds[0, too_low])

    if len(too_high) > 0:
        if observation[-3] > 0.8999:
            reward += np.sum(
                (action[:-1][too_high] - 0.1) / 0.8 * (environment.temp_bounds[1, too_high] - temperatures[too_high]))
        elif observation[-3] < 0.1001:
            reward += np.sum((1 - (action[:-1][too_high] - 0.1) / 0.8) * (
                    environment.temp_bounds[1, too_high] - temperatures[too_high]))
        else:
            raise ValueError(f"Case has to be 0.1 or 0.9")
        reward += np.sum(environment.temp_bounds[1, too_high] - temperatures[too_high])

    return reward


def compute_temperature_reward_old(environment, observation, action):
    """
    Small helper function to compute temperature rewards
    """

    reward = 0

    # Get the current temperatures
    temperatures = environment.scale_back_temperatures(
        environment.current_data[environment.n_autoregression + environment.current_step, environment.predictions_columns[:len(environment.rooms)]])

    too_low = np.where(temperatures < environment.temp_bounds[0, :])[0]
    too_high = np.where(temperatures > environment.temp_bounds[1, :])[0]
    right = np.where((temperatures >= environment.temp_bounds[0, :]) & (temperatures <= environment.temp_bounds[1, :]))[
        0]

    reward += np.sum(1 - (
        np.power(temperatures[right] - (environment.temp_bounds[0, right] + environment.temp_bounds[1, right]) / 2, 2)))
    if observation[-3] > 0.8999:
        if len(too_low) > 0:
            reward += np.sum((1 - (action[:-1][too_low] - 0.1) / 0.8) * (
                        temperatures[too_low] - environment.temp_bounds[0, too_low]))
        if len(too_high) > 0:
            reward += np.sum(
                (action[:-1][too_high] - 0.1) / 0.8 * (environment.temp_bounds[1, too_high] - temperatures[too_high]))
    elif observation[-3] < 0.1001:
        if len(too_low) > 0:
            reward += np.sum(
                (action[:-1][too_low] - 0.1) / 0.8 * (temperatures[too_low] - environment.temp_bounds[0, too_low]))
        if len(too_high) > 0:
            reward += np.sum((1 - (action[:-1][too_high] - 0.1) / 0.8) * (
                        environment.temp_bounds[1, too_high] - temperatures[too_high]))
    else:
        raise ValueError(f"Case has to be 0.1 or 0.9")

    return reward


def compute_battery_reward(environment, observation):
    # if len(self.battery_powers) > 1:
    #   battery_ramping = np.power((self.battery_powers[-2] - battery_power) / self.battery_max_power, 2)
    # else:
    #   battery_ramping = np.abs(battery_power)

    if (observation[-1] > environment.battery_margins[0]) & (observation[-1] < environment.battery_margins[1]):
        return 1 - (
            np.power(len(environment.rooms) * (50. - observation[-1]) / (50. - environment.battery_margins[0]), 2))

    else:
        return - np.abs(environment.battery_barrier_penalty)


def compute_price_reward(environment, observation, electricity_from_grid):
    price = observation[-2]
    price_scale = len(environment.rooms) / (
                environment.battery_max_power + environment.umar_model.max_["Electricity total consumption"])

    if electricity_from_grid > 0:
        return - price_scale * electricity_from_grid * price
    else:
        return - price_scale * electricity_from_grid * 0.1


def compute_reward_old(self, observation, action):
    temperature_reward = compute_temperature_reward(self, observation, action)

    electricity_from_grid = self.compute_electricity_from_grid(observation, action)
    price_reward = compute_price_reward(self, observation, electricity_from_grid)

    if self.battery:
        battery_reward = compute_battery_reward(self, observation)
        return temperature_reward + battery_reward + price_reward
    else:
        return temperature_reward + price_reward


def compute_reward(environment, observation, action):
    """
    Small helper function to compute temperature rewards
    """

    reward = 0

    # Get the current temperatures
    temperatures = environment.scale_back_temperatures(
        environment.current_data[
            environment.n_autoregression + environment.current_step, environment.predictions_columns[
                                                                     :len(environment.rooms)]])

    bounds = (environment.current_data[environment.n_autoregression + environment.current_step, -2:] - 0.1) / 0.8 * (
                environment.temp_bounds[-1] - environment.temp_bounds[0]) + environment.temp_bounds[0]

    too_low = np.where(temperatures < bounds[0])[0]
    too_high = np.where(temperatures > bounds[1])[0]

    if observation[-5] > 0.8999:
        right = np.where((temperatures >= bounds[0]) & (temperatures <= bounds[0] + 1))[0]
        reward += np.sum(1 - (np.power(2 * (temperatures[right] - bounds[0] - 0.5), 2)))

    elif observation[-5] < 0.1001:
        right = np.where((temperatures <= bounds[1]) & (temperatures >= bounds[1] - 1))[0]
        reward += np.sum(1 - (np.power(2 * (bounds[1] - 0.5 - temperatures[right]), 2)))
    else:
        raise ValueError(f"Case has to be 0.1 or 0.9")

    if len(too_low) > 0:
        reward += np.sum(temperatures[too_low] - bounds[0])

    if len(too_high) > 0:
        reward += np.sum(bounds[1] - temperatures[too_high])

    return reward, temperatures, bounds


class UMAREnv(gym.GoalEnv):
    """Custom Environment of UMAR that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, umar_model, battery_model, agent_kwargs, compute_reward):
        """
        Initialize a custom environment, using the learned models of the battery and of the
        UMAR dynamics.

        Args:
            umar_model:     Model of the room temperature and energy consumption
            battery_model:  Model of the battery
            agent_kwargs:   See 'parameters.py', all arguments needed for an agent
            compute_reward: Function to use to compute rewards if HER is not used
        """
        super(UMAREnv, self).__init__()

        # Define the models
        self.umar_model = umar_model
        self.battery_model = battery_model

        # Recall constants
        self.n_autoregression = umar_model.n_autoregression
        self.threshold_length = umar_model.threshold_length
        self.temp_bounds = agent_kwargs["temp_bounds"]
        self.price_levels = agent_kwargs["price_levels"]
        self.discrete = agent_kwargs["discrete"]
        self.ddpg = agent_kwargs["ddpg"]
        self.rooms = agent_kwargs["rooms"]
        self.battery = agent_kwargs["battery"]
        self.discrete_actions = np.append(agent_kwargs["discrete_actions"][:len(self.rooms)],
                                          agent_kwargs["discrete_actions"][-1])
        self.battery_max_power = agent_kwargs["battery_max_power"]
        self.COP = agent_kwargs["COP"]
        self.save_path = agent_kwargs["save_path"]
        self.battery_margins = agent_kwargs["battery_margins"]
        self.battery_size = agent_kwargs["battery_size"]
        self.battery_barrier_penalty = agent_kwargs["battery_barrier_penalty"]
        self.temperature_penalty_factor = agent_kwargs["temperature_penalty_factor"]
        self.backup = agent_kwargs["backup"]
        self.small_model = agent_kwargs["small_model"]
        self.small_obs = agent_kwargs["small_obs"]
        self.her = agent_kwargs["her"]
        self.autoregressive_terms = agent_kwargs["autoregressive_terms"]
        self.simple = agent_kwargs["simple_env"]

        self.compute_reward = compute_reward
        self.inverse_normalize  = self.umar_model.room_models[self.rooms[0]].dataset.inverse_normalize

        self.components = [component for room in self.rooms for component
                           in self.umar_model.components if room in component]

        # The data will be handled as a numpy array, so we recall which column is what
        self.observation_columns, self.autoregressive_columns, self.control_columns, self.predictions_columns = self.get_columns()
        self.elec_column = \
            np.where(self.umar_model.data.columns[self.observation_columns] == f"Electricity total consumption")[
                0].item()

        # Define the observation space (Recall that the UMAR model needs normalized inputs)
        if self.her:
            self.obs_space = spaces.Box(low=np.array([0.099] * (
                    len(self.observation_columns) + len(self.autoregressive_columns) * len(
                self.autoregressive_terms)) + [0.]),
                                        high=np.array([0.901] * (len(self.observation_columns) + len(
                                            self.autoregressive_columns) * len(self.autoregressive_terms)) + [100.]),
                                        dtype=np.float64)

            self.observation_space = spaces.Dict({
                'observation': spaces.Box(low=np.array([0.099] * (
                        len(self.observation_columns) + len(self.autoregressive_columns) * len(
                    self.autoregressive_terms)) + [0.]),
                                          high=np.array([0.901] * (len(self.observation_columns) + len(
                                              self.autoregressive_columns) * len(self.autoregressive_terms)) + [100.]),
                                          dtype=np.float64),
                'achieved_goal': spaces.Box(low=np.array([0.099] * len(self.rooms)),
                                            high=np.array([0.901] * len(self.rooms)),
                                            dtype=np.float64),
                'desired_goal': spaces.Box(low=np.array([0.099] * len(self.rooms)),
                                           high=np.array([0.901] * len(self.rooms)),
                                           dtype=np.float64)
            })
        else:
            self.observation_space = spaces.Box(low=np.array([0.099] * len(self.observation_columns) + [0.]),
                                                high=np.array([0.901] * len(self.observation_columns) + [100.]),
                                                dtype=np.float64)

        # Define different action spaces
        if self.discrete:
            if self.battery:
                self.action_space = spaces.MultiDiscrete(self.discrete_actions)
            else:
                self.action_space = spaces.MultiDiscrete(self.discrete_actions[:-1])

        elif self.ddpg:
            if self.battery:
                self.action_space = spaces.Box(low=np.array([0.1] * len(self.rooms) + [-self.battery_max_power]),
                                               high=np.array([0.9] * len(self.rooms) + [self.battery_max_power]),
                                               dtype=np.float64)
            else:
                self.action_space = spaces.Box(low=np.array([0.1] * len(self.rooms)),
                                               high=np.array([0.9] * len(self.rooms)),
                                               dtype=np.float64)
        else:
            if self.battery:
                self.action_space = spaces.Box(low=np.array([-1.] * (len(self.rooms) + 1)),
                                               high=np.array([1.] * (len(self.rooms) + 1)),
                                               dtype=np.float64)
            else:
                self.action_space = spaces.Box(low=np.array([-1.] * len(self.rooms)),
                                               high=np.array([1.] * len(self.rooms)),
                                               dtype=np.float64)

        # Initialize various needed variables
        self.current_sequence = None
        self.last_sequence = None
        self.last_init_temp = None
        self.last_goal_number = 1
        self.goal_number = 1
        self.current_data = None
        self.current_step = 0
        self.h = {}
        self.c = {}
        self.battery_soc = []
        self.battery_powers = []
        self.base = {}
        self.past_heating = {}
        self.past_cooling = {}

        # Define the scaled values needed for the reward computation (as the data for the model of
        # UMAR is normalized)
        # self.zero_energy, self.scaled_temp_bounds = self.compute_scaled_limits(
        self.scaled_temp_bounds = self.compute_scaled_limits()
        self.min_maxes = self.get_min_maxes()

        self.desired_goal = None
        self.desired_goals = [self.scaled_temp_bounds[0, :],
                              self.scaled_temp_bounds.mean(axis=0),
                              self.scaled_temp_bounds[1, :]]

        # Set up the training environment, detecting the presence of GPU
        self.device = self.training_setup()
        self.rewards = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.comfort_violations = []
        self.prices = []

        self.good_sequences = []
        self.bad_sequences = []

    def reset(self, sequence=None, goal_number=None, init_temp=None):
        """
        Function called at the end of an episode to reset and start a new one. Here we pick a random
        sequence of data from NEST and consider it as an episode
        """
        if sequence is not None:
            self.current_sequence = sequence
        else:
            lengths = np.array([len(seq) for seq in self.umar_model.train_sequences])
            sequence = np.random.choice(self.umar_model.train_sequences, p=lengths / sum(lengths))
            if len(sequence) > self.threshold_length + self.n_autoregression:
                start = np.random.randint(self.n_autoregression, len(sequence) - self.threshold_length + 1)
                self.current_sequence = sequence[start - self.n_autoregression: start + self.threshold_length]
            else:
                self.current_sequence = sequence

        if self.her:
            if goal_number is not None:
                self.goal_number = goal_number
                self.desired_goal = self.desired_goals[self.goal_number]
            else:
                # Chose a goal (keep the number for the plot)
                self.goal_number = np.random.randint(0, len(self.desired_goals))
                self.desired_goal = self.desired_goals[self.goal_number]

        # Put the current step to 0
        self.current_step = 0
        self.electricity_imports = []
        self.rewards = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.comfort_violations = []
        self.prices = []

        self.current_data = self.umar_model.data.iloc[self.current_sequence, :].copy().values

        # Put the control actions (i.e. the valves) and tee predictions (i.e. the room temperatures, as
        # they will e defined by the model) to nans
        self.current_data[self.n_autoregression + 1:, self.predictions_columns] = np.nan
        self.current_data[self.n_autoregression:, self.control_columns] = np.nan

        # Define the current observation of the agent from UMAR data
        output = {}
        self.input_tensor = {}
        for room in self.rooms:

            self.input_tensor[room], _ = self.umar_model.room_models[room].build_input_output_from_sequences(
                sequences=[self.current_sequence])
            (base, heating, cooling), (self.h[room], self.c[room]) = self.umar_model.room_models[room].model(
                self.input_tensor[room][:, :self.n_autoregression, :])
            output[f"Energy room {room}"] = torch.ones(self.input_tensor[room].shape[0], 1).to(self.device)
            output[f"Energy room {room}"] *= self.umar_model.room_models[room].zero_energy

            output[f"Thermal temperature measurement {room}"] = base
            self.base[room] = base

            if heating is not None:
                self.past_heating[room] = heating["Temperature"]
                output[f"Thermal temperature measurement {room}"] += self.past_heating[room]
                output[f"Energy room {room}"] += heating["Energy"]
            if cooling is not None:
                self.past_cooling[room] = cooling["Temperature"]
                output[f"Thermal temperature measurement {room}"] -= self.past_cooling[room]
                output[f"Energy room {room}"] -= cooling["Energy"]

        prediction = np.array([output[x].squeeze().detach().numpy()
                               for x in self.umar_model.data.columns
                               if (x in self.components) & (x[-3:] in self.rooms)])
        prediction = prediction.clip(0.1, 0.9)

        # Input the predictions made for this new step
        self.current_data[self.n_autoregression, self.predictions_columns] = prediction

        # Get the new UMAR observation
        observation = np.array([])
        if self.her:
            for x in self.autoregressive_terms:
                observation = np.append(observation,
                                        self.current_data[self.n_autoregression - x, self.autoregressive_columns])
        observation = np.append(observation, self.current_data[self.n_autoregression, self.observation_columns])

        # Reinitialize battery information and get the first SoC measurement (i.e. first observation)
        if self.battery:
            self.battery_soc = []
            self.battery_powers = []
            self.battery_soc.append(
                self.battery_model.data.loc[self.umar_model.data.index[self.current_sequence[self.n_autoregression]]][
                    0])

            # Append the battery observation
            observation = np.append(observation, self.battery_soc[0])
        else:
            observation = np.append(observation, 0.)

        observation = self._get_observation(observation)

        # Return the first observation to start the episode, as per gym requirements
        return observation

    def _get_observation(self, observation):
        """
        Function to wrap the observation in therequired format for HER if needed
        """
        if self.her:
            index = 9 if self.small_model else 10
            return OrderedDict([
                ('observation', observation.copy()),
                ('achieved_goal', observation[-index - 2 * len(self.rooms): -index - len(self.rooms)].copy()),
                ('desired_goal', self.desired_goal.copy())])
        else:
            return observation

    def step(self, action, rule_based: bool = False, force_done: bool = False):
        """
        Function used to make a step according to the chosen action by the algorithm
        """
        observation = self.current_data[self.n_autoregression + self.current_step, self.observation_columns]
        assert (observation[-5] == 0.9) | (observation[-5] == 0.1), "Impossible!"

        if not self.battery:
            action = np.append(action, 0.)

        # We firstly need to rescale actions to 0.1-0.9 because this is what our UMAR model requires as input
        # And we also rescale battery actions to lie within bounds
        if not rule_based:

            action = self.scale_action(observation[-5], action)

            if self.backup:
                action, _ = self.check_backup(observation, action)

        # Record the valves openings in the data
        self.current_data[self.n_autoregression + self.current_step, self.control_columns] = action[:-1]

        if self.her:
            observation = np.array([])
            for x in self.autoregressive_terms:
                observation = np.append(observation, self.current_data[
                    self.n_autoregression + self.current_step - x, self.autoregressive_columns])
            observation = np.append(observation,
                                    self.current_data[
                                        self.n_autoregression + self.current_step, self.observation_columns])

        # Recall the battery power input and compute its impact on the SoC
        if self.battery:
            self.battery_powers.append(action[-1])
            # self.battery_soc.append((self.battery_soc[-1] + self.battery_model.predict(action[-1])).clip(0, 100))
            self.battery_soc.append(self.battery_soc[-1] + action[-1] / 60 * self.umar_model.interval)

            # Append the battery observation to the other one
            observation = np.append(observation, self.battery_soc[-1])

        else:
            observation = np.append(observation, 0.)

        observation = self._get_observation(observation)

        # Compute the reward using the custom function
        if self.her:
            reward = self._compute_reward(observation['achieved_goal'], observation['desired_goal'], None)
            _, temperatures, bounds = self.compute_reward(self, observation['observation'],
                                                           np.array([0.] * self.action_space.shape[0]))
        else:
            reward, temperatures, bounds = self.compute_reward(self, observation, action)

        self.rewards.append(reward)

        self.lower_bounds.append(bounds[0])
        self.upper_bounds.append(bounds[1])

        comfort_violation = []
        for i in range(len(temperatures)):
            if temperatures[i] < bounds[0]:
                comfort_violation.append(bounds[0] - temperatures[i])
            elif temperatures[i] > bounds[1]:
                comfort_violation.append(temperatures[i] - bounds[1])
        self.comfort_violations.append(np.sum(comfort_violation))

        self.electricity_imports.append(self.compute_electricity_from_grid(observation, action))
        price = (observation[9] - 0.1) / 0.8 * (850 - 100) + 100

        if len(self.price_levels) == 1:
            self.prices.append((self.price_levels[0] * self.electricity_imports[-1])*price)
        else:
            self.prices.append((((self.current_data[self.n_autoregression + self.current_step, -3] - 0.1) / 0.8 * \
                                (self.price_levels[-1] - self.price_levels[0]) + self.price_levels[0]) * \
                               self.electricity_imports[-1])*price)

        output = {}
        for room in self.rooms:

            self.input_tensor[room][:, self.n_autoregression + self.current_step,
            self.umar_model.room_models[room].pred_column] = self.base[room].squeeze()
            self.input_tensor[room][:, self.n_autoregression + self.current_step,
            self.umar_model.room_models[room].ctrl_column] = action[np.where(str(room) in self.rooms)[0]].item()
            (base, heating, cooling), (self.h[room], self.c[room]) = self.umar_model.room_models[room].model(
                self.input_tensor[room][:, self.n_autoregression + self.current_step, :].view(
                    self.input_tensor[room].shape[0], 1, -1), self.h[room], self.c[room])

            output[f"Energy room {room}"] = torch.ones(self.input_tensor[room].shape[0], 1).to(self.device)
            output[f"Energy room {room}"] *= self.umar_model.room_models[room].zero_energy

            output[f"Thermal temperature measurement {room}"] = base
            self.base[room] = base

            if heating is not None:
                self.past_heating[room] += heating["Temperature"]
                output[f"Thermal temperature measurement {room}"] += self.past_heating[room]
                output[f"Energy room {room}"] += heating["Energy"]
            if cooling is not None:
                self.past_cooling[room] += cooling["Temperature"]
                output[f"Thermal temperature measurement {room}"] -= self.past_cooling[room]
                output[f"Energy room {room}"] -= cooling["Energy"]
        prediction = np.array([output[x].squeeze().detach().numpy()
                               for x in self.umar_model.data.columns
                               if (x in self.components) & (x[-3:] in self.rooms)])
        prediction = prediction.clip(0.1, 0.9)

        # Advance of one step
        self.current_step += 1

        # Input the predictions made for this new step
        self.current_data[self.n_autoregression + self.current_step, self.predictions_columns] = prediction

        observation = np.array([])
        if self.her:
            for x in self.autoregressive_terms:
                observation = np.append(observation, self.current_data[
                    self.n_autoregression + self.current_step - x, self.autoregressive_columns])
        observation = np.append(observation,
                                self.current_data[self.n_autoregression + self.current_step, self.observation_columns])

        # Recall the battery power input and compute its impact on the SoC
        if self.battery:
            observation = np.append(observation, self.battery_soc[-1])
        else:
            observation = np.append(observation, 0.)

        observation = self._get_observation(observation)

        # Check if we are at the end of an episode, i.e. at the end of the sequence being analyzed
        done = (self.current_step == len(self.current_sequence) - self.n_autoregression - 1) | force_done

        # Can use this to recall information
        info = {}

        if done:
            if self.battery:
                _, temperatures, bounds = self.compute_reward(self, observation,
                                                              np.array([0.] * self.action_space.shape[0]))
            else:
                _, temperatures, bounds = self.compute_reward(self, observation,
                                                              np.array([0.] * (self.action_space.shape[0] + 1)))

            self.lower_bounds.append(bounds[0])
            self.upper_bounds.append(bounds[1])

            comfort_violation = []
            for i in range(len(temperatures)):
                if temperatures[i] < bounds[0]:
                    comfort_violation.append(bounds[0] - temperatures[i])
                elif temperatures[i] > bounds[1]:
                    comfort_violation.append(temperatures[i] - bounds[1])
            self.comfort_violations.append(np.sum(comfort_violation))

            self.last_sequence = self.current_sequence.copy()
            self.last_goal_number = self.goal_number
            self.last_rewards = self.rewards.copy()
            self.last_data = self.current_data.copy()
            self.last_electricity_imports = self.electricity_imports.copy()
            self.last_battery_soc = self.battery_soc.copy()
            self.last_battery_powers = self.battery_powers.copy()
            self.last_lower_bounds = self.lower_bounds.copy()
            self.last_upper_bounds = self.upper_bounds.copy()
            self.last_prices = self.prices.copy()
            self.last_comfort_violations = self.comfort_violations.copy()

        # Return everything needed for gym
        return observation, reward, done, info

    def _compute_reward(self, achieved_goal, desired_goal, info):
        return - np.sum(np.abs(achieved_goal - desired_goal))

    def get_columns(self):
        """
        Helper function to get the columns of the data corresponding to different needed groups
        Returns the columns corresponding to observations, controls and predictions
        """

        columns = []
        autoregressive = []
        for component in self.components:
            for sensor in self.umar_model.components_inputs_dict[component]:
                if self.small_obs:
                    if (sensor not in columns) & ("humidity" not in sensor) & ("window" not in sensor) &\
                            ("wind" not in sensor) & ("relative" not in sensor) & ("brightness" not in sensor):
                        columns.append(sensor)
                        if sensor in self.umar_model.autoregressive:
                            autoregressive.append(sensor)
                else:
                    if sensor not in columns:
                        columns.append(sensor)
                        if sensor in self.umar_model.autoregressive:
                            autoregressive.append(sensor)

        columns.append("Electricity price")
        columns.append("Lower bound")
        columns.append("Upper bound")
        rooms = [f"Thermal valve {room}" for room in self.rooms]

        observation_columns = [i for i in range(len(self.umar_model.data.columns))
                               if ("valve" not in self.umar_model.data.columns[i]) &
                               (self.umar_model.data.columns[i] in columns)]
        observation_columns = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13]

        autoregressive_columns = [i for i in range(len(self.umar_model.data.columns))
                                  if ("valve" not in self.umar_model.data.columns[i]) &
                                  (self.umar_model.data.columns[i] in autoregressive)]
        control_columns = [i for i in range(len(self.umar_model.data.columns))
                           if self.umar_model.data.columns[i] in rooms]
        predictions_columns = [i for i in range(len(self.umar_model.data.columns))
                               if self.umar_model.data.columns[i] in self.components]

        return observation_columns, autoregressive_columns, control_columns, predictions_columns

    def compute_scaled_limits(self):
        """
        Function to compute the scaled version of 0 for the energy consumption.
        This also transforms temperature bounds from human-readable form (e.g. 21, 23 degrees) to
        the scaled version between 0.1 and 0.9.
        Note that this is different for each room, since their min and max temperatures differ.
        """
        # Build an array that will contain the lower and higher bound values for each room
        scaled_temp_bounds = np.zeros((len(self.temp_bounds), len(self.rooms)))

        # Loop over the rooms
        i = 0
        for room in self.components:
            # Compute the scaled bounds and store them
            if "temperature" in room:
                scaled_temp_bounds[:, i] = (self.temp_bounds - self.umar_model.min_[room]) / \
                                           (self.umar_model.max_[room] - self.umar_model.min_[room]) \
                                           * 0.8 + .1
                i += 1

        # temp = [self.temp_bounds[0]] * len(self.rooms)
        # for i in range(len(self.temp_bounds) - 1):
        #    temp += [self.temp_bounds[i+1]] * len(self.rooms)
        # self.temp_bounds = np.array(temp).reshape(len(self.temp_bounds), len(self.rooms))

        # Return everything
        return scaled_temp_bounds  # zero_energy, scaled_temp_bounds

    def get_min_maxes(self):
        min_maxes = np.zeros((2, len(self.rooms)))
        i = 0
        for room in self.components:
            # Compute the scaled bounds and store them
            if "temperature" in room:
                min_maxes[:, i] = [self.umar_model.min_[room], self.umar_model.max_[room]]
                i += 1
        return min_maxes

    def scale_action(self, case, action):

        if self.discrete:
            # Discrete action are recorded as integers and need to be put to floats for the next manipulations to work
            action = action.astype(np.float64)
            # Rescale the valves openings to 0.1-0.9
            if self.umar_model.heating & self.umar_model.cooling:
                if case > 0.8999:
                    action[:-1] = 0.8 * action[:-1] / (self.discrete_actions[:-1] - 1) + 0.1
                else:
                    action[:-1] = 0.8 * (-(action[:-1] / (self.discrete_actions[:-1] - 1) + 1)) + 0.1
            else:
                action[:-1] = 0.8 * action[:-1] / (self.discrete_actions[:-1] - 1) + 0.1
            # Rescale the battery power to [-max, max]
            if self.battery:
                action[-1] = self.battery_max_power * 2 * (action[-1] / (self.discrete_actions[-1] - 1) - 0.5)

        elif not self.ddpg:
            # Rescale the valves openings to 0.1-0.9
            if self.umar_model.heating & self.umar_model.cooling:
                if case > 0.8999:
                    action[:-1] = 0.4 * (action[:-1] + 1) + 0.1
                else:
                    action[:-1] = 0.4 * (-action[:-1] + 1) + 0.1
            else:
                action[:-1] = (action[:-1] + 1) * 0.4 + 0.1
            # Rescale the battery power to [-max, max]
            if self.battery:
                action[-1] = self.battery_max_power * action[-1]

        else:
            # Nothing to do in the DDPG case as the actions already lie in the right intervals
            pass

        return action

    def check_backup(self, observation, action):

        if self.small_model:
            index = 8
        else:
            index = 9

        temperatures = self.scale_back_temperatures(observation[-(index + 2 * len(self.rooms)):-(index + len(self.rooms))])

        bounds = (observation[-2:] - 0.1) / 0.8 * (self.temp_bounds[-1] - self.temp_bounds[0]) + self.temp_bounds[0]

        no_heating = []
        full_heating = []
        no_cooling = []
        full_cooling = []

        if observation[-5] > 0.8999:
            no_heating = np.where(bounds[1] + 1 < temperatures)[0]
            full_heating = np.where(bounds[0] - 1 > temperatures)[0]
            action[:-1][no_heating] = 0.1
            action[:-1][full_heating] = 0.9
        elif observation[-5] < 0.10001:
            no_cooling = np.where(bounds[0] - 1 > temperatures)[0]
            full_cooling = np.where(bounds[1] + 1 < temperatures)[0]
            action[:-1][no_cooling] = 0.1
            action[:-1][full_cooling] = 0.9
        else:
            raise ValueError(f"Case can only be 0.1 or 0.9, not {observation[-5]}")

        if self.battery:
            if action[-1] > 0:
                margin = self.battery_margins[1] - self.battery_soc[-1]
                if action[-1] > margin:
                    action[-1] = margin
            elif action[-1] < 0:
                margin = self.battery_soc[-1] - self.battery_margins[0]
                if action[-1] < - margin:
                    action[-1] = - margin

        flags = np.sum(no_heating) + np.sum(full_heating) + np.sum(no_cooling) + np.sum(full_cooling)

        return action, flags

    def scale_back_temperatures(self, data):
        return self.min_maxes[0, :] + (data - 0.1) * (self.min_maxes[1, :] - self.min_maxes[0, :]) / 0.8

    def compute_electricity_from_grid(self, observation, action):
        energies = observation[-(5 + len(self.rooms)):-5]

        electricity_consumption = observation[self.elec_column]

        electricity_consumption = self.umar_model.min_["Electricity total consumption"] + (
                electricity_consumption - 0.1) * (self.umar_model.max_["Electricity total consumption"] -
                                                  self.umar_model.min_["Electricity total consumption"]) / 0.8
        energy = abs(sum([self.umar_model.min_[f"Energy room {room}"] + (energies[i] - 0.1) * (
                self.umar_model.max_[f"Energy room {room}"] - self.umar_model.min_[f"Energy room {room}"]) / 0.8 for
                          i, room in enumerate(self.rooms)])) / int(60. / self.umar_model.interval)

        if observation[-5] > 0.8999:
            energy /= self.COP[1]
        elif observation[-5] < 0.10001:
            energy /= self.COP[0]
        else:
            raise NotImplementedError(f"Case can only be 0.1 or 0.9, here: {observation[-5]}")

        if self.battery:
            return energy + action[-1]  # + electricity_consumption
        else:
            return energy  # + electricity_consumption

    def training_setup(self):
        """
        Small helper function to detect GPU and make it work on Colab
        """

        # Detect GPU and define te device accordingly
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("GPU acceleration on!")

        # Local case
        else:
            device = "cpu"

        # Return the computed stuff
        return device

    def render(self, mode='human', force: bool = False):
        """
        Custom render function: here it plots what the agent did during the episode, i.e. the temperature
        evolution of the rooms with the corresponding valves pattern, the energy consumption, as well
        as the battery power and SoC
        """
        if (self.current_step == len(self.current_sequence) - self.n_autoregression - 2) | force:
            _, data = prepare_performance_plot(env=self,
                                               sequence=self.current_sequence[:-1],
                                               data=self.current_data[:-1, :],
                                               rewards=self.rewards[:-1],
                                               electricity_imports=self.electricity_imports,
                                               lower_bounds=self.lower_bounds[:-1],
                                               upper_bounds=self.upper_bounds[:-1],
                                               prices=self.prices[:-1],
                                               comfort_violations=self.comfort_violations[:-1],
                                               battery_soc=self.battery_soc,
                                               battery_powers=self.battery_powers[:-1],
                                               show_=False)

            analyze_agent(env=self,
                          name="RL agent",
                          data=data,
                          rewards=self.rewards,
                          comfort_violations=self.comfort_violations,
                          prices=self.prices[:-1],
                          electricity_imports=self.electricity_imports[:-1],
                          lower_bounds=self.lower_bounds,
                          upper_bounds=self.upper_bounds,
                          battery_soc=self.battery_soc,
                          battery_powers=self.battery_powers[:-1])

            plt.tight_layout()
            plt.show()
            plt.close()

    def close(self):
        pass


class ToyEnv(gym.GoalEnv):
    """
    Custom simpler Environment of UMAR that follows gym interface

    TODO: there is a lot of redundancy with UmarEnv above. this works like that but it might
    be merged somehow
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, umar_model, battery_model, agent_kwargs, compute_reward):
        """
        Initialize a custom environment, using the learned models of the battery and a toy
        model of UMAR dynamics (ARX).

        Args:
            umar_model:     Model of the room temperature and energy consumption
            battery_model:  Model of the battery
            agent_kwargs:   See 'parameters.py', all arguments needed for an agent
            compute_reward: Function to use to compute rewards if HER is not used
        """
        super(ToyEnv, self).__init__()

        # Define the models
        self.umar_model = umar_model
        self.battery_model = battery_model

        # Recall constants
        # self.n_autoregression = 14
        self.n_autoregression = 13
 #       self.threshold_length = umar_model.threshold_length
        self.rooms = agent_kwargs["rooms"]
        self.min_ = umar_model.min_
        self.max_ = umar_model.max_
        self.temp_bounds = agent_kwargs["temp_bounds"]
        self.price_levels = agent_kwargs["price_levels"]
        self.discrete = agent_kwargs["discrete"]
        self.ddpg = agent_kwargs["ddpg"]
        self.battery = agent_kwargs["battery"]
        self.discrete_actions = np.append(agent_kwargs["discrete_actions"][:len(self.rooms)],
                                          agent_kwargs["discrete_actions"][-1])
        self.battery_max_power = agent_kwargs["battery_max_power"]
        self.COP = agent_kwargs["COP"]
        self.save_path = agent_kwargs["save_path"]
        self.battery_margins = agent_kwargs["battery_margins"]
        self.battery_size = agent_kwargs["battery_size"]
        self.battery_barrier_penalty = agent_kwargs["battery_barrier_penalty"]
        self.temperature_penalty_factor = agent_kwargs["temperature_penalty_factor"]
        self.backup = agent_kwargs["backup"]
        self.small_model = agent_kwargs["small_model"]
        self.small_obs = agent_kwargs["small_obs"]
        self.her = agent_kwargs["her"]
        self.autoregressive_terms = agent_kwargs["autoregressive_terms"]
        self.temp_bounds = agent_kwargs["temp_bounds"]
        self.interval = hp.interval
        self.simple = agent_kwargs["simple_env"]
        self.num_test_envs = agent_kwargs['num_test_envs']

        # Define constants
        print("hihi")
        print(self.rooms)
        # while isinstance(self.rooms, list) and len(self.rooms) == 1:
        #     self.rooms = self.rooms[0]
        
        self.temp_min = self.umar_model.min_["Temperature 272"]
        self.temp_max = self.umar_model.max_["Temperature 272"]
        # self.temp_min = self.umar_model.min_["Temperature "+self.rooms]
        # self.temp_max = self.umar_model.max_["Temperature "+self.rooms]
        print("mina")
        # print(self.temp_min)
        # print(self.temp_max)
        self.cooling_power = -2
        self.heating_power = 2.5
        if not isinstance(self.rooms, list):
            self.rooms = [self.rooms]

        self.cooling_powers = {room: self.cooling_power * 217 / (5 * 217 + 2 * 70) if room in ["272", "274"]
        else self.cooling_power * 217 * 3 / (5 * 217 + 2 * 70) for room in self.rooms}
        self.heating_powers = {room: self.heating_power * 217 / (5 * 217 + 2 * 70) if room in ["272", "274"]
        else self.heating_power * 217 * 3 / (5 * 217 + 2 * 70) for room in self.rooms}


        self.compute_reward = compute_reward

        self.observation_columns, self.control_columns, self.predictions_columns = self.get_columns_mina()

        # Define the observation space (Recall that the UMAR model needs normalized inputs)

        if self.her:
            self.obs_space = spaces.Box(
                low=np.array([0.099] * (len(self.observation_columns) * len(self.autoregressive_terms)) + [0.]),
                high=np.array([0.901] * (len(self.observation_columns) * len(self.autoregressive_terms)) + [100.]),
                dtype=np.float64)

            self.observation_space = spaces.Dict({
                'observation': spaces.Box(
                    low=np.array([0.099] * (len(self.observation_columns) * len(self.autoregressive_terms)) + [0.]),
                    high=np.array([0.901] * (len(self.observation_columns) * len(self.autoregressive_terms)) + [100.]),
                    dtype=np.float64),
                'achieved_goal': spaces.Box(low=np.array([0.099] * len(self.rooms)),
                                            high=np.array([0.901] * len(self.rooms)),
                                            dtype=np.float64),
                'desired_goal': spaces.Box(low=np.array([0.099] * len(self.rooms)),
                                           high=np.array([0.901] * len(self.rooms)),
                                           dtype=np.float64)
            })
        else:
            self.observation_space = spaces.Box(low=np.array([0.099] * len(self.observation_columns) + [0.]),
                                                high=np.array([0.901] * len(self.observation_columns) + [100.]),
                                                dtype=np.float64)

        # Define different action spaces
        if self.discrete:
            if self.battery:
                self.action_space = spaces.MultiDiscrete(self.discrete_actions)
            else:
                self.action_space = spaces.MultiDiscrete(self.discrete_actions[:-1])

        elif self.ddpg:
            if self.battery:
                self.action_space = spaces.Box(low=np.array([0.1] * len(self.rooms) + [-self.battery_max_power]),
                                               high=np.array([0.9] * len(self.rooms) + [self.battery_max_power]),
                                               dtype=np.float64)
            else:
                self.action_space = spaces.Box(low=np.array([0.1] * len(self.rooms)),
                                               high=np.array([0.9] * len(self.rooms)),
                                               dtype=np.float64)

        else:
            if self.battery:
                self.action_space = spaces.Box(low=np.array([-1.] * (len(self.rooms) + 1)),
                                               high=np.array([1.] * (len(self.rooms) + 1)),
                                               dtype=np.float64)
            else:
                self.action_space = spaces.Box(low=np.array([-1.] * len(self.rooms)),
                                               high=np.array([1.] * len(self.rooms)),
                                               dtype=np.float64)




        # Initialize various needed variables


        self.current_sequence = None
        self.last_sequence = None
        self.last_goal_number = 1
        self.goal_number = 1
        self.current_data = None
        self.current_step = 0
        self.battery_soc = []
        self.battery_powers = []

        # Define the scaled values needed for the reward computation (as the data for the model of
        # UMAR is normalized)
        # self.zero_energy, self.scaled_temp_bounds = self.compute_scaled_limits(
        self.scaled_temp_bounds = self.compute_scaled_limits()

        self.desired_goal = None
        self.desired_goals = [self.scaled_temp_bounds[0, :],
                              self.scaled_temp_bounds.mean(axis=0),
                              self.scaled_temp_bounds[1, :]]

        # Set up the training environment, detecting the presence of GPU
        self.device = self.training_setup()
        self.rewards = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.comfort_violations = []
        self.prices = []
        self.test_i = 0
        self.autoregression = agent_kwargs['autoregression']
        self.case_column = self.umar_model.case_column
    def compute_scaled_limits(self):
        """
        Function to compute the scaled version of 0 for the energy consumption.
        This also transforms temperature bounds from human-readable form (e.g. 21, 23 degrees) to
        the scaled version between 0.1 and 0.9.
        Note that this is different for each room, since their min and max temperatures differ.
        """
        # Build an array that will contain the lower and higher bound values for each room
        scaled_temp_bounds = np.zeros((len(self.temp_bounds), len(self.rooms)))
        self.temp_bounds = np.array(self.temp_bounds)
        # Loop over the rooms
        for i in range(len(self.rooms)):
            scaled_temp_bounds[:, i] = (self.temp_bounds - self.temp_min) / (self.temp_max - self.temp_min) * 0.8 + .1

        # Return everything
        return scaled_temp_bounds

    def get_columns(self):
        """
        Helper function to get the columns of the data corresponding to different needed groups
        Returns the columns corresponding to observations, controls and predictions
        """
        forbidden = [x for x in self.umar_model.data.columns if (x[-3:-1] == "27") & (x[-3:] not in self.rooms)]
        observation_columns = [i for i in range(len(self.umar_model.data.columns))
                               if ("u_" not in self.umar_model.data.columns[i]) &
                               (self.umar_model.data.columns[i] not in forbidden)]
        # observation_columns = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13]
        observation_columns = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]


        control_columns = [i for i in range(len(self.umar_model.data.columns))
                           if ("u_" in self.umar_model.data.columns[i]) &
                           np.any([room in self.umar_model.data.columns[i] for room in self.rooms])]
        predictions_columns = [i for i in range(len(self.umar_model.data.columns))
                               if ("u_" not in self.umar_model.data.columns[i]) &
                               ("_27" in self.umar_model.data.columns[i])]


        return observation_columns, control_columns, predictions_columns
    def get_columns_mina(self):
        """
        Helper function to get the columns of the data corresponding to different needed groups
        Returns the columns corresponding to observations, controls and predictions
        """

        # observation_columns = [ 0 , 1 ,2, 4,5 , 6,7,8,10,11, 12]
        observation_columns = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]
        control_columns = [9]
        # predictions_columns = [2,3]
        predictions_columns = [2]
        return observation_columns, control_columns, predictions_columns

    def reset(self, sequence=None, goal_number=None, init_temp=None, is_test=False):
        """
        Function called at the end of an episode to reset and start a new one. Here we pick a random
        sequence of data from NEST and consider it as an episode if nothing is given.
        Parameters can be passed for duplicates, i.e. to compare agents with the same conditions.
        Args:
            sequence:       Sequence of data to use
            goal_number:    For HER, goal to attain for that episode
            init_temp:      Initial temperatures of the rooms
        Returns:
            The first observation of the environment
        """
        if is_test:
            self.sequences = np.array(self.umar_model.validation_sequences)[
                [(100 * i) % len(self.umar_model.validation_sequences) for i in range(self.num_test_envs)]]
            if isinstance(self.sequences, tuple):
                self.sequences = [self.sequences]
            # In the train case, we just keep all the data
        else:
            self.sequences = self.umar_model.train_sequences


        ########################################################
        # Define the sequence of data to use
        if sequence is not None:
            self.current_sequence = sequence
        else:
            if is_test:
                # Cycle around the test sequences
                self.current_sequence = self.sequences[self.test_i % len(self.sequences)]
                self.test_i += 1
            else:
                self.current_sequence = self.sequences[0]
        if self.her:
            if goal_number is not None:
                self.goal_number = goal_number
                self.desired_goal = self.desired_goals[self.goal_number]
            else:
                self.goal_number = np.random.randint(0, len(self.desired_goals))
                self.desired_goal = self.desired_goals[self.goal_number]

        # Put the current step to 0
        self.current_step = 0
        self.electricity_imports = []
        self.rewards = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.comfort_violations = []
        self.prices = []

        # Recall the current data
        #self.current_data = self.umar_model.data.iloc[self.current_sequence, :].copy().values
        self.current_data = self.umar_model.data.iloc[self.current_sequence[0]: self.current_sequence[1] + 1].copy().values

        if init_temp is None:
            # self.init_temp = np.random.rand(1) * 3 + 23
            self.init_temp = np.random.rand(1) * 3 + 21
        else:
            self.init_temp = init_temp

            # Store the (normalized version of the) initial room temperatures. If HER is used with autoregression,
            # Initialize the temperature "n_autoregression" steps in the past and run the model until the present
            # time to get the actual room temperatures at the start of the experiments
        if self.her:
            self.current_data[0, self.predictions_columns] = self.init_temp
            for i in range(1, self.n_autoregression):
                self.current_data[i, self.predictions_columns] = self.normalize(
                    self.umar_model.model(self.inverse_normalize(self.current_data[i - 1, :])))
        else:
            # self.current_data[self.n_autoregression, self.predictions_columns] = self.normalize(self.init_temp)
            self.current_data[0, self.predictions_columns] = self.normalize(self.init_temp)


            #############################################################################################
            self.model_state = None
            # for i in range(self.n_autoregression+1):
            #     pred, self.model_state = self.umar_model.model(
            #             self.current_data[i, :], self.model_state, warm_start=False)
            m = 1
            for i in range(self.n_autoregression+1):
                for j in range(m):
                    pred, self.model_state = self.umar_model.model(
                            self.current_data[i, :], self.model_state, warm_start=True)

                ###
                self.current_data[self.n_autoregression , self.predictions_columns] = \
                    pred[0].detach().numpy()
                ####
        ###########################################################################################################
            # Get the new UMAR observation
        observation = np.array([])
        # print("mina")
        # print(self.autoregressive_terms)
        if self.her:
            for x in self.autoregressive_terms:
                observation = np.append(observation,
                                        self.current_data[self.n_autoregression - x, self.observation_columns])

        # print("mina ")
        # # print(len(self.current_data))
        # print(self.umar_model.data.columns[self.observation_columns])
        # print(self.umar_model.data.columns[self.control_columns])

        observation = np.append(observation, self.current_data[self.n_autoregression, self.observation_columns])
        # Reinitialize battery information and get the first SoC measurement (i.e. first observation)
        if self.battery:
            self.battery_soc = []
            self.battery_powers = []
            self.battery_soc.append(
                self.battery_model.data.loc[self.umar_model.data.index[self.current_sequence[self.n_autoregression]]][
                    0])

            # Append the battery observation
            observation = np.append(observation, self.battery_soc[0])
        else:
            observation = np.append(observation, 0.)

        print(observation)
        return self._get_observation(observation)

    def _get_observation(self, observation):
        """
        Function to wrap the observation in therequired format for HER if needed
        """
        if self.her:
            index = 5
            return OrderedDict([
                ('observation', observation.copy()),
                ('achieved_goal', observation[-index - 2 * len(self.rooms): -index - len(self.rooms)].copy()),
                ('desired_goal', self.desired_goal.copy())])
        else:
            return observation

    def step(self, action, rule_based: bool = False, force_done: bool = False):
        """
        Function used to make a step according to the chosen action by the algorithm.
        Args:
            action:         Action taken by the agent
            rule_based:     If this is a rule-based agent or not
            force_done:     If we want to force an episode to end
        Returns:
            observation:    The new observation after the step is taken
            reward:         The reward obtained at that step
            done:           Whether the episode is over or not
            info            Can pass some other information
        """
        # Retrieve the current observation from data
        # print("narges")
        # print(action)
        observation = self.current_data[self.n_autoregression + self.current_step, self.observation_columns]
        if self.her:
            observation = np.array([])
            for x in self.autoregressive_terms:
                observation = np.append(observation, self.current_data[
                    self.n_autoregression + self.current_step - x, self.observation_columns])
            observation = np.append(observation,
                                    self.current_data[
                                        self.n_autoregression + self.current_step, self.observation_columns])

        if not self.battery:
            action = np.append(action, 0.)


        # print(self.current_step)

        # We firstly need to rescale actions to 0.1-0.9 because this is what our UMAR model requires as input
        # And we also rescale battery actions to lie within bounds
        # if not rule_based:
        #     print("ghghghg")
        #     action = self.scale_action(observation[-5], action)
        #     print(action)
        #
        #     if self.backup:
        #         action, _ = self.check_backup(observation, action)

        # Record the valves openings in the data
        # print(action[:-1])
        self.current_data[self.n_autoregression + self.current_step, self.control_columns] = action[:-1]

        # Recall the battery power input and compute its impact on the SoC
        if self.battery:
            self.battery_powers.append(action[-1])
            # self.battery_soc.append((self.battery_soc[-1] + self.battery_model.predict(action[-1])).clip(0, 100))
            self.battery_soc.append(self.battery_soc[-1] + action[-1] / 60 * self.umar_model.interval)

            # Append the battery observation to the other one
            observation = np.append(observation, self.battery_soc[-1])

        else:
            observation = np.append(observation, 0.)

        observation = self._get_observation(observation)
        # print("aaaaaa")
        # print(self.current_data[self.n_autoregression + self.current_step, :])
        # print(observation)
        # Compute the reward using the custom function
        if self.her:
            reward = self._compute_reward(observation['achieved_goal'], observation['desired_goal'], None)
            _, temperatures, bounds = self.compute_reward(self, observation['observation'],
                                                          np.array([0.] * self.action_space.shape[0]))
        else:
            reward, temperatures, bounds, price_reward, comfort_reward = self.compute_reward(self, observation, action)


        # Store everything
        self.rewards.append(reward)
        # self.price_reward.append(price_reward)
        # self.comfort_reward.append(comfort_reward)
        self.lower_bounds.append(bounds[0])
        self.upper_bounds.append(bounds[1])

        comfort_violation = []
        for i in range(len(temperatures)):
            if temperatures[i] < bounds[0]:
                comfort_violation.append(bounds[0] - temperatures[i])
            elif temperatures[i] > bounds[1]:
                comfort_violation.append(temperatures[i] - bounds[1])
        self.comfort_violations.append(np.sum(comfort_violation))

        self.electricity_imports.append(self.compute_electricity_from_grid(observation, action))
        # print("mmmmm")
        # print(len(self.electricity_imports))
        # price = (observation[9] - 0.1) / 0.8 * (850 - 100) + 100
        price = (observation[9] - 0.1) / 0.8 * (90 - 4) + 4

        if len(self.price_levels) == 1:
            self.prices.append(self.price_levels[0] * self.electricity_imports[-1]*price)
            # print("mmmmm")
            # print(sum(self.prices))
        else:
            self.prices.append(
                ((self.current_data[self.n_autoregression + self.current_step, -3] - 0.1) / 0.8 * \
                                (self.price_levels[-1] - self.price_levels[0]) + self.price_levels[0]) * \
                               self.electricity_imports[-1]
                               )

        # Advance of one step
        self.current_step += 1
        # print("1111111")
        # print(self.predictions_columns)
        # print("soghraaaaaaa")
        # print(self.current_step)
        # print("111111")
        # print(self.current_data[self.n_autoregression + self.current_step - 1, :])
        a = self.inverse_normalize(
                        torch.from_numpy(self.current_data[self.n_autoregression + self.current_step - 1, :]).view(1,1, -1)
                        )
        # print("222222")
        # print(type(a))
        # print(type(self.current_data[self.n_autoregression + self.current_step - 1, :]))
        # print((a))
        # print((self.current_data[self.n_autoregression + self.current_step - 1, :]))
        # print(type(self.normalize((self.umar_model.model(a)))))
        # print(type(self.umar_model.model(self.current_data[self.n_autoregression + self.current_step - 1, :])))
        # self.current_data[self.n_autoregression + self.current_step, self.predictions_columns] = \
        #     self.normalize(
        #         self.umar_model.model(
        #             a
        #         )
        #     )
        pred, self.model_state = self.umar_model.model(self.current_data[self.n_autoregression + self.current_step - 1, :],self.model_state, warm_start=False)
        self.current_data[self.n_autoregression + self.current_step, self.predictions_columns] = pred[0].detach().numpy()
        # print('minaaaaaaaaa')
        # print(self.current_data[self.n_autoregression + self.current_step - 1, :])
        # print(type(self.current_data[self.n_autoregression + self.current_step - 1, :]))
        # print(self.current_data[self.n_autoregression + self.current_step - 1, :].dtype)
        # print(self.current_data[self.n_autoregression + self.current_step, self.predictions_columns])
        # t1 = self.scale_back_temperatures(
        #     self.current_data[self.n_autoregression + self.current_step, self.predictions_columns])
        # print(t1)
        # Get the new observation
        observation = np.array([])
        if self.her:
            for x in self.autoregressive_terms:
                observation = np.append(
                    observation,
                    self.current_data[
                    self.n_autoregression + self.current_step - x, self.autoregressive_columns]
                    )
        observation = np.append(
            observation,
            self.current_data[self.n_autoregression + self.current_step, self.observation_columns]
            )

        # Recall the battery power input and compute its impact on the SoC
        if self.battery:
            observation = np.append(observation, self.battery_soc[-1])
        else:
            observation = np.append(observation, 0.)

        observation = self._get_observation(observation)

        # Check if we are at the end of an episode, i.e. at the end of the sequence being analyzed
        # done = (self.current_step == len(self.current_sequence) - self.n_autoregression - 1) | force_done
        done = (self.current_step == self.current_sequence[1]-self.current_sequence[0] - self.n_autoregression - 1) | force_done
        # print("alain")
        # print(self.current_sequence)
        # Can use this to recall information
        info = {}

        # When done, make sure to store everything for further use
        if done:
            if self.battery:
                _, temperatures, bounds, _, _ = self.compute_reward(self, observation,
                                                              np.array([0.] * self.action_space.shape[0]))
            else:
                _, temperatures, bounds, _, _ = self.compute_reward(self, observation,
                                                              np.array([0.] * (self.action_space.shape[0] + 1)))

            self.lower_bounds.append(bounds[0])
            self.upper_bounds.append(bounds[1])

            comfort_violation = []
            # print("kkkkk")
            # print(bounds)
            for i in range(len(temperatures)):
                if temperatures[i] < bounds[0]:
                    comfort_violation.append(bounds[0] - temperatures[i])
                elif temperatures[i] > bounds[1]:
                    comfort_violation.append(temperatures[i] - bounds[1])
            #self.comfort_violations.append(np.sum(comfort_violation))
            # print("sara")
            # print(self.current_sequence)
            self.last_sequence = tuple(self.current_sequence)
            # self.last_sequence = self.current_sequence.copy()
            self.last_goal_number = self.goal_number
            self.last_init_temp = self.init_temp
            self.last_rewards = self.rewards.copy()
            self.last_data = self.current_data.copy()
            self.last_electricity_imports = self.electricity_imports.copy()
            self.last_battery_soc = self.battery_soc.copy()
            self.last_battery_powers = self.battery_powers.copy()
            self.last_lower_bounds = self.lower_bounds.copy()
            self.last_upper_bounds = self.upper_bounds.copy()
            self.last_prices = self.prices.copy()
            self.last_comfort_violations = self.comfort_violations.copy()

        # Return everything needed for gym
        return observation, reward, done, info, price_reward, comfort_reward

    def _compute_reward(self, achieved_goal, desired_goal, info):
        """
        Function computing the reward for the HER case
        """
        return - np.sum(np.abs(achieved_goal - desired_goal))

    def inverse_normalize(self, data):
        """
        Function o inverse the normalization of the data, as the models require inputs with their
        original units
        Args:
            data:   The data to inverse, either a DataFrame or an array
        Returns:
            The noramlized data
        """

        # Copy the data else it gets modified
        # data = data.copy()
        data = data.clone()

        # Define the columns where min_ and max_ exist, as well as temperature columns, which are treated
        # separately
        temps = [i for i in range(len(self.umar_model.data.columns)) if "Temperature 27" in self.umar_model.data.columns[i]]

        # print("ghghghg")
        # print(temps) # result is 2 ,3 which means column temp272 and temp273


        cols = [i for i in range(len(self.umar_model.data.columns)) if
                self.umar_model.data.columns[i] in self.min_.index]


        # for i in range(len(self.umar_model.data.columns)):
        #     if "Temperature 27" in self.umar_model.data.columns[i]:
        #         print("ghghghg")
        #         print(self.umar_model.data.columns[i])

        # DataFrame case: used for plotting
        # print("ghghghg")
        # print( isinstance(data, pd.DataFrame))
        if isinstance(data, pd.DataFrame):
            data.iloc[:, cols] = (data.iloc[:, cols] - 0.1) * (self.max_ - self.min_) / 0.8 + self.min_
            data.iloc[:, temps] = (data.iloc[:, temps] - 0.1) * (self.temp_max - self.temp_min) / 0.8 + self.temp_min
            # For plotting, we ned the values of the valve opening as "control columns"
            data.iloc[:, self.control_columns] = (data.iloc[:, self.control_columns] - 0.1) / 0.8

        # Array case: used to preprocess the data to predict the next state
        else:
            # print("mina")
            # print(type(data))
            # print(data.shape)
            data = data[:,:,:].numpy()
            data = data.reshape(-1)
            # print("soghraaaaa")
            # print(self.max_)
            # print(self.min_)
            # print(data)
            data[cols] = (data[cols] - 0.1) * (self.max_ - self.min_) / 0.8 + self.min_
            data[temps] = (data[temps] - 0.1) * (self.temp_max - self.temp_min) / 0.8 + self.temp_min
            # print("aliiii")
            # print(data)
            # For prediction, the required info is the actual energy consumption, not the valves
            # if data[-4] > 0.5:
            data[self.control_columns] = self.electricity_imports[-1] * self.COP[1] * (60 / self.interval)
            # else:
            #     data[self.control_columns] = - self.electricity_imports[-1] * self.COP[0] * (60 / self.interval)
            # print("narges")
            # print(type(data))
            # print(data.shape)


        return data

    def normalize(self, temperatures):
        """
        Function to normalize the temperatures given by the model
        """
        if isinstance(temperatures[0], torch.Tensor):
            tem = temperatures[0].detach().numpy()
        else:
            tem = temperatures[0]
        return (tem - self.temp_min) / (self.temp_max - self.temp_min) * 0.8 + 0.1

    def dnormalize(self, temperatures):
        """
        Function to normalize the temperatures given by the model
        """
        if isinstance(temperatures[0], torch.Tensor):
            tem = temperatures[0].detach().numpy()
        else:
            tem = temperatures[0]
        return self.temp_min + (tem - 0.1) * (self.temp_max - self.temp_min) / 0.8

    def scale_action(self, case, action):
        """
        Function to scale the agent's output to 0.1-0.9
        """

        if self.discrete:
            # Discrete action are recorded as integers and need to be put to floats for the next manipulations to work
            action = action.astype(np.float64)
            # Rescale the valves openings to 0.1-0.9
            if self.umar_model.heating & self.umar_model.cooling:
                if case > 0.8999:
                    action[:-1] = 0.8 * action[:-1] / (self.discrete_actions[:-1] - 1) + 0.1
                else:
                    action[:-1] = 0.8 * (-(action[:-1] / (self.discrete_actions[:-1] - 1) + 1)) + 0.1
            else:
                action[:-1] = 0.8 * action[:-1] / (self.discrete_actions[:-1] - 1) + 0.1
            # Rescale the battery power to [-max, max]
            if self.battery:
                action[-1] = self.battery_max_power * 2 * (action[-1] / (self.discrete_actions[-1] - 1) - 0.5)

        elif not self.ddpg:
            # Rescale the valves openings to 0.1-0.9
            if self.umar_model.heating & self.umar_model.cooling:
                if case > 0.8999:
                    action[:-1] = 0.4 * (action[:-1] + 1) + 0.1
                else:
                    action[:-1] = 0.4 * (-action[:-1] + 1) + 0.1
            else:
                action[:-1] = (action[:-1] + 1) * 0.4 + 0.1
            # Rescale the battery power to [-max, max]
            if self.battery:
                action[-1] = self.battery_max_power * action[-1]

        else:
            # Nothing to do in the DDPG case as the actions already lie in the right intervals
            #pass
# =============================================================================
#             if case > 0.8999:
#                 action[:-1] = np.clip(action[:-1],0,1)
#                 action[:-1] = 0.8*action[:-1] + 0.1
#             else:
#                 action[:-1] = -action[:-1]
#                 action[:-1] = np.clip(action[:-1],0,1)
#                 action[:-1] = 0.8*action[:-1] + 0.1
# =============================================================================
            if len(action) > 1:
                action[:-1] = 0.8*action[:-1] + 0.1
            else:
                action = 0.8*action + 0.1
        return action

    def check_backup(self, observation, action):
        """
        Function to check if any backup is needed, i.e. correct stupid actions taken by the agents.
        As currently implemented, this ensures no heating is on when the temperature is above 1 degree
        above the comfort bound, and that there is full heating when the temperature is below 1 degree
        below the lower bound. similar for cooling
        Args:
            observation:    The observation
            action:         The action taken
        Returns:
            action:         The corrected action
            flag:           Whether a correction was needed
        """

        index = 10
        print("Warning: Is this right?")

        # Define temperatures and bounds
        temperatures = self.scale_back_temperatures(observation[index])
        bounds = (observation[-2:] - 0.1) / 0.8 * (self.temp_bounds[-1] - self.temp_bounds[0]) + self.temp_bounds[0]

        no_heating = []
        full_heating = []
        no_cooling = []
        full_cooling = []

        # Check for excessive or missing heating
        no_heating = np.where(bounds[1] + 1 < temperatures)[0]
        full_heating = np.where(bounds[0] - 1 > temperatures)[0]
        action[:-1][no_heating] = 0.1
        action[:-1][full_heating] = 0.9

        # Battery case
        if self.battery:
            if action[-1] > 0:
                margin = self.battery_margins[1] - self.battery_soc[-1]
                if action[-1] > margin:
                    action[-1] = margin
            elif action[-1] < 0:
                margin = self.battery_soc[-1] - self.battery_margins[0]
                if action[-1] < - margin:
                    action[-1] = - margin

        # Check if any correction was needed
        flag = np.sum(no_heating) + np.sum(full_heating) + np.sum(no_cooling) + np.sum(full_cooling)

        return action, flag

    def scale_back_temperatures(self, data):
        """
        function to scale back given room temperatures
        """
        return self.temp_min + (data - 0.1) * (self.temp_max - self.temp_min) / 0.8

    def compute_electricity_from_grid(self, observation, action):
        """
        Function computing the electricity needed to take the given action
        """

        # Compute the energy according to the valve openings, with differentiation between heating and
        # cooling as the system has different efficiencies
        energy = np.sum(
            [abs((action[i] - 0.1) / 0.8 * self.heating_powers[room]) for i, room in enumerate(self.rooms)])
        # if action[0]> 0.12 and action[0]< 0.7 :
        #     print("gfggh")
        #     print(energy)
        #     print(action)
        # if  action[0] > 0.7:
        #     print("momomom")
        #     print(energy)
        #     print(action)


        # Compute the energy from the power and use the COP to transform thermal energy to electrical one
        energy /= self.COP[1] * (60 / self.interval)


        # Add the battery power if needed
        if self.battery:
            return energy + action[-1]
        else:
            return energy

    def training_setup(self):
        """
        Small helper function to detect GPU and make it work on Colab
        """

        # Detect GPU and define te device accordingly
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("GPU acceleration on!")

        # Local case
        else:
            device = "cpu"

        # Return the computed stuff
        return device

    def render(self, mode='human', force: bool = False):
        """
        Custom render function: here it plots what the agent did during the episode, i.e. the temperature
        evolution of the rooms with the corresponding valves pattern, the energy consumption, as well
        as the battery power and SoC
        """
        if (self.current_step == len(self.current_sequence) - self.n_autoregression - 2) | force:
            _, data = prepare_performance_plot(env=self,
                                               sequence=self.current_sequence[:-2],
                                               data=self.current_data[:-2, :],
                                               rewards=self.rewards[:-1],
                                               electricity_imports=self.electricity_imports[:-1],
                                               lower_bounds=self.lower_bounds,
                                               upper_bounds=self.upper_bounds,
                                               prices=self.prices[:-1],
                                               comfort_violations=self.comfort_violations,
                                               battery_soc=self.battery_soc,
                                               battery_powers=self.battery_powers[:-1],
                                               show_=False)

            analyze_agent(env=self,
                          name="RL agent",
                          data=data,
                          rewards=self.rewards,
                          comfort_violations=self.comfort_violations,
                          prices=self.prices[:-1],
                          electricity_imports=self.electricity_imports[:-1],
                          lower_bounds=self.lower_bounds,
                          upper_bounds=self.upper_bounds,
                          battery_soc=self.battery_soc,
                          battery_powers=self.battery_powers[:-1])

            plt.tight_layout()
            plt.show()
            plt.close()

    def close(self):
        pass
        
        
def analysis_lstm(agents, types, eval_env, sequences,
             print_: bool = False, print_frequency: int = 25,
             plot_: bool = False, plot_frequency: int = 25,
                 normalizing: bool = False,
                 deterministic: bool = True,
                 verbose: int = 1):

    # Convert to VecEnv for consistency
    if not isinstance(eval_env, VecEnv) and not isinstance(eval_env, UMAREnv):
        eval_env = DummyVecEnv([lambda: eval_env])
        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

    _eval_env = eval_env

    if isinstance(eval_env, UMAREnv):
        eval_env = _eval_env
    else:
        if normalizing:
            eval_env = _eval_env.venv.venv.envs[0]
        else:
            eval_env = _eval_env.venv.envs[0]
            
    rewards = {name: [] for name in agents.keys()}
    comfort_violations = {name: [] for name in agents.keys()}
    prices = {name: [] for name in agents.keys()}
            
    for num, sequence in enumerate(sequences):
                
        if num % 50 == 49:
            print(num+1)
        
        for i, name in enumerate(agents.keys()):
            
            if types[name] == "RL":
        
                evaluate_lstm_policy(agents[name].model,
                                     eval_env,
                                     n_eval_episodes=1,
                                     sequence=sequence,
                                     render=False,
                                     deterministic=deterministic,
                                     normalizing=normalizing,
                                     all_goals=False,
                                     return_episode_rewards=True)
            
                rewards[name].append(eval_env.last_rewards)
                comfort_violations[name].append(eval_env.last_comfort_violations)
                prices[name].append(eval_env.last_prices)
            
            else:
        
                agents[name].run(sequence,
                                 eval_env.last_goal_number,
                                 render=False)

                rewards[name].append(agents[name].env.last_rewards)
                comfort_violations[name].append(agents[name].env.last_comfort_violations)
                prices[name].append(agents[name].env.last_prices)
                
            if print_ & (num % print_frequency == 0):
        
                print(f"__________________________\n\n{name}:")
                print(f"\nReward: {np.sum(np.array(rewards[name][-1])):.2f}  -  ", end="")
                print(f"Comfort violations:   {np.sum(np.array(comfort_violations[name][-1])):.2f}  -  ", end="")
                print(f"Total benefits/costs: {np.sum(np.array(prices[name][-1])):.2f}", end="")
                
            if plot_ & (num % plot_frequency == 0):
                
                if types[name] == "RL":
                    env = eval_env
                else:
                    env = agents[name].env

                axes, data = prepare_performance_plot(env=env,
                                                   sequence=env.last_sequence,
                                                   data=env.last_data,
                                                   rewards=env.last_rewards,
                                                   electricity_imports=env.last_electricity_imports,
                                                   lower_bounds=env.last_lower_bounds,
                                                   upper_bounds=env.last_upper_bounds,
                                                   prices=env.last_prices,
                                                   comfort_violations=env.last_comfort_violations,
                                                   battery_soc=env.last_battery_soc,
                                                   battery_powers=env.last_battery_powers,
                                                   label=name,
                                                   elec_price=True if i == 0 else False,
                                                   print_=True if i == 0 else False,
                                                   show_=True if i == len(agents)-1 else False,
                                                    axes=axes if i != 0 else None)


        if plot_ & (num % plot_frequency == 0):
            plt.tight_layout()
            plt.show()
            plt.close()
        
    return rewards, comfort_violations, prices
    
    
def analysis_toy(agents, types, eval_env, indices,
             print_: bool = False, print_frequency: int = 25,
             plot_: bool = False, plot_frequency: int = 25,
                 normalizing: bool = False,
                 deterministic: bool = True,
                 verbose: int = 1):

    # Convert to VecEnv for consistency
    if not isinstance(eval_env, VecEnv) and not isinstance(eval_env, UMAREnv):
        eval_env = DummyVecEnv([lambda: eval_env])
        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

    _eval_env = eval_env

    if isinstance(eval_env, UMAREnv):
        eval_env = _eval_env
    else:
        if normalizing:
            eval_env = _eval_env.venv.venv.envs[0]
        else:
            eval_env = _eval_env.venv.envs[0]
            
    rewards = {name: [] for name in agents.keys()}
    comfort_violations = {name: [] for name in agents.keys()}
    prices = {name: [] for name in agents.keys()}
            
    for num, index in enumerate(indices):
        
        sequence = np.arange(index-20, index+96)
        
        if num % 50 == 49:
            print(num+1)
        
        for i, name in enumerate(agents.keys()):
            
            if types[name] == "RL":
        
                evaluate_lstm_policy(agents[name].model,
                                     eval_env,
                                     n_eval_episodes=1,
                                     sequence=sequence,
                                     render=False,
                                     deterministic=deterministic,
                                     normalizing=normalizing,
                                     all_goals=False,
                                     return_episode_rewards=True)
            
                rewards[name].append(eval_env.last_rewards)
                comfort_violations[name].append(eval_env.last_comfort_violations)
                prices[name].append(eval_env.last_prices)
            
            else:
        
                agents[name].run(sequence,
                                 eval_env.last_goal_number,
                                 render=False)

                rewards[name].append(agents[name].env.last_rewards)
                comfort_violations[name].append(agents[name].env.last_comfort_violations)
                prices[name].append(agents[name].env.last_prices)
                
            if print_ & (num % print_frequency == 0):
        
                print(f"__________________________\n\n{name}:")
                print(f"\nReward: {np.sum(np.array(rewards[name][-1])):.2f}  -  ", end="")
                print(f"Comfort violations:   {np.sum(np.array(comfort_violations[name][-1])):.2f}  -  ", end="")
                print(f"Total benefits/costs: {np.sum(np.array(prices[name][-1])):.2f}", end="")
                
            if plot_ & (num % plot_frequency == 0):
                
                if types[name] == "RL":
                    env = eval_env
                else:
                    env = agents[name].env

                axes, data = prepare_performance_plot(env=env,
                                                   sequence=env.last_sequence,
                                                   data=env.last_data,
                                                   rewards=env.last_rewards,
                                                   electricity_imports=env.last_electricity_imports,
                                                   lower_bounds=env.last_lower_bounds,
                                                   upper_bounds=env.last_upper_bounds,
                                                   prices=env.last_prices,
                                                   comfort_violations=env.last_comfort_violations,
                                                   battery_soc=env.last_battery_soc,
                                                   battery_powers=env.last_battery_powers,
                                                   label=name,
                                                   elec_price=True if i == 0 else False,
                                                   print_=True if i == 0 else False,
                                                   show_=True if i == len(agents)-1 else False,
                                                    axes=axes if i != 0 else None)


        if plot_ & (num % plot_frequency == 0):
            plt.tight_layout()
            plt.show()
            plt.close()
        
    return rewards, comfort_violations, prices


#################################################################################
# ################################################################################

# OLD versions

#################################################################################
# ################################################################################


class UMAREnv_old_three(gym.GoalEnv):
    """Custom Environment of UMAR that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, umar_model, battery_model, agent_kwargs, compute_reward=compute_reward):
        """
        Initialize a custom environment, using the learned models of the battery and of the
        UMAR dynamics.

        Args:
            umar_model:     Model of the room temperature and energy consumption
            battery_model:  Model of the battery
            agent_kwargs:   See 'parameters.py', all arguments needed for an agent
            compute_reward: Function to use to compute rewards if HER is not used
        """
        super(UMAREnv, self).__init__()

        # Define the models
        self.umar_model = umar_model
        self.battery_model = battery_model

        # Recall constants
        self.n_autoregression = umar_model.n_autoregression
        self.threshold_length = umar_model.threshold_length
        self.temp_bounds = agent_kwargs["temp_bounds"]
        self.discrete = agent_kwargs["discrete"]
        self.ddpg = agent_kwargs["ddpg"]
        self.rooms = agent_kwargs["rooms"]
        self.battery = agent_kwargs["battery"]
        self.discrete_actions = np.append(agent_kwargs["discrete_actions"][:len(self.rooms)],
                                          agent_kwargs["discrete_actions"][-1])
        self.battery_max_power = agent_kwargs["battery_max_power"]
        self.COP = agent_kwargs["COP"]
        self.save_path = agent_kwargs["save_path"]
        self.battery_margins = agent_kwargs["battery_margins"]
        self.battery_size = agent_kwargs["battery_size"]
        self.battery_barrier_penalty = agent_kwargs["battery_barrier_penalty"]
        self.temperature_penalty_factor = agent_kwargs["temperature_penalty_factor"]
        self.backup = agent_kwargs["backup"]
        self.small_model = agent_kwargs["small_model"]
        self.her = agent_kwargs["her"]
        self.autoregressive_terms = agent_kwargs["autoregressive_terms"]

        self._compute_reward = compute_reward

        self.components = [component for room in self.rooms for component
                           in self.umar_model.components if room in component]

        # The data will be handled as a numpy array, so we recall which column is what
        self.observation_columns, self.autoregressive_columns, self.control_columns, self.predictions_columns = self.get_columns()
        self.elec_column = \
            np.where(self.umar_model.data.columns[self.observation_columns] == f"Electricity total consumption")[
                0].item()

        # Define the observation space (Recall that the UMAR model needs normalized inputs)
        if self.her:
            self.obs_space = spaces.Box(low=np.array([0.099] * (
                        len(self.observation_columns) + len(self.autoregressive_columns) * len(
                    self.autoregressive_terms)) + [0.]),
                                        high=np.array([0.901] * (len(self.observation_columns) + len(
                                            self.autoregressive_columns) * len(self.autoregressive_terms)) + [100.]),
                                        dtype=np.float64)

            self.observation_space = spaces.Dict({
                'observation': spaces.Box(low=np.array([0.099] * (
                            len(self.observation_columns) + len(self.autoregressive_columns) * len(
                        self.autoregressive_terms)) + [0.]),
                                          high=np.array([0.901] * (len(self.observation_columns) + len(
                                              self.autoregressive_columns) * len(self.autoregressive_terms)) + [100.]),
                                          dtype=np.float64),
                'achieved_goal': spaces.Box(low=np.array([0.099] * len(self.rooms)),
                                            high=np.array([0.901] * len(self.rooms)),
                                            dtype=np.float64),
                'desired_goal': spaces.Box(low=np.array([0.099] * len(self.rooms)),
                                           high=np.array([0.901] * len(self.rooms)),
                                           dtype=np.float64)
            })
        else:
            self.observation_space = spaces.Box(low=np.array([0.099] * len(self.observation_columns) + [0.]),
                                                high=np.array([0.901] * len(self.observation_columns) + [100.]),
                                                dtype=np.float64)

        # Define different action spaces
        if self.discrete:
            if self.battery:
                self.action_space = spaces.MultiDiscrete(self.discrete_actions)
            else:
                self.action_space = spaces.MultiDiscrete(self.discrete_actions[:-1])

        elif self.ddpg:
            if self.battery:
                self.action_space = spaces.Box(low=np.array([0.1] * len(self.rooms) + [-self.battery_max_power]),
                                               high=np.array([0.9] * len(self.rooms) + [self.battery_max_power]),
                                               dtype=np.float64)
            else:
                self.action_space = spaces.Box(low=np.array([0.1] * len(self.rooms)),
                                               high=np.array([0.9] * len(self.rooms)),
                                               dtype=np.float64)
        else:
            if self.battery:
                self.action_space = spaces.Box(low=np.array([-1.] * (len(self.rooms) + 1)),
                                               high=np.array([1.] * (len(self.rooms) + 1)),
                                               dtype=np.float64)
            else:
                self.action_space = spaces.Box(low=np.array([-1.] * len(self.rooms)),
                                               high=np.array([1.] * len(self.rooms)),
                                               dtype=np.float64)

        # Initialize various needed variables
        self.current_sequence = None
        self.last_sequence = None
        self.last_goal_number = 1
        self.goal_number = 1
        self.current_data = None
        self.current_step = 0
        self.h = {}
        self.c = {}
        self.battery_soc = []
        self.battery_powers = []
        self.base = {}
        self.past_heating = {}
        self.past_cooling = {}

        # Define the scaled values needed for the reward computation (as the data for the model of
        # UMAR is normalized)
        # self.zero_energy, self.scaled_temp_bounds = self.compute_scaled_limits(self.temp_bounds)
        self.scaled_temp_bounds = self.compute_scaled_limits()
        self.min_maxes = self.get_min_maxes()

        self.desired_goal = None
        self.desired_goals = [self.scaled_temp_bounds[0, :],
                              self.scaled_temp_bounds.mean(axis=0),
                              self.scaled_temp_bounds[1, :]]

        # Set up the training environment, detecting the presence of GPU
        self.device = self.training_setup()
        self.rewards = []

        self.good_sequences = []
        self.bad_sequences = []

    def reset(self, sequence=None, goal_number=None):
        """
        Function called at the end of an episode to reset and start a new one. Here we pick a random
        sequence of data from NEST and consider it as an episode
        """
        if sequence is not None:
            self.current_sequence = sequence
        else:
            lengths = np.array([len(seq) for seq in self.umar_model.train_sequences])
            sequence = np.random.choice(self.umar_model.train_sequences, p=lengths / sum(lengths))
            if len(sequence) > self.threshold_length + self.n_autoregression:
                start = np.random.randint(self.n_autoregression, len(sequence) - self.threshold_length + 1)
                self.current_sequence = sequence[start - self.n_autoregression: start + self.threshold_length]
            else:
                self.current_sequence = sequence

        if self.her:
            if goal_number is not None:
                self.goal_number = goal_number
                self.desired_goal = self.desired_goals[self.goal_number]
            else:
                # Chose a goal (keep the number for the plot)
                self.goal_number = np.random.randint(0, len(self.desired_goals))
                self.desired_goal = self.desired_goals[self.goal_number]

        # Put the current step to 0
        self.current_step = 0
        self.electricity_imports = []
        self.rewards = []

        self.current_data = self.umar_model.data.iloc[self.current_sequence, :].copy().values

        # Put the control actions (i.e. the valves) and tee predictions (i.e. the room temperatures, as
        # they will e defined by the model) to nans
        self.current_data[self.n_autoregression + 1:, self.predictions_columns] = np.nan
        self.current_data[self.n_autoregression:, self.control_columns] = np.nan

        # Define the current observation of the agent from UMAR data
        output = {}
        self.input_tensor = {}
        for room in self.rooms:

            self.input_tensor[room], _ = self.umar_model.room_models[room].build_input_output_from_sequences(
                sequences=[self.current_sequence])
            (base, heating, cooling), (self.h[room], self.c[room]) = self.umar_model.room_models[room].model(
                self.input_tensor[room][:, :self.n_autoregression, :])
            output[f"Energy room {room}"] = torch.ones(self.input_tensor[room].shape[0], 1).to(self.device)
            output[f"Energy room {room}"] *= self.umar_model.room_models[room].zero_energy

            output[f"Thermal temperature measurement {room}"] = base
            self.base[room] = base

            if heating is not None:
                self.past_heating[room] = heating["Temperature"]
                output[f"Thermal temperature measurement {room}"] += self.past_heating[room]
                output[f"Energy room {room}"] += heating["Energy"]
            if cooling is not None:
                self.past_cooling[room] = cooling["Temperature"]
                output[f"Thermal temperature measurement {room}"] -= self.past_cooling[room]
                output[f"Energy room {room}"] -= cooling["Energy"]

        prediction = np.array([output[x].squeeze().detach().numpy()
                               for x in self.umar_model.data.columns
                               if (x in self.components) & (x[-3:] in self.rooms)])
        prediction = prediction.clip(0.1, 0.9)

        # Input the predictions made for this new step
        self.current_data[self.n_autoregression, self.predictions_columns] = prediction

        # Get the new UMAR observation
        observation = np.array([])
        if self.her:
            for x in self.autoregressive_terms:
                observation = np.append(observation,
                                        self.current_data[self.n_autoregression - x, self.autoregressive_columns])
        observation = np.append(observation, self.current_data[self.n_autoregression, self.observation_columns])

        # Reinitialize battery information and get the first SoC measurement (i.e. first observation)
        if self.battery:
            self.battery_soc = []
            self.battery_powers = []
            self.battery_soc.append(
                self.battery_model.data.loc[self.umar_model.data.index[self.current_sequence[0]]][0])

            # Append the battery observation
            observation = np.append(observation, self.battery_soc[0])
        else:
            observation = np.append(observation, 0.)

        # Return the first observation to start the episode, as per gym requirements
        return self._get_observation(observation)

    def _get_observation(self, observation):
        if self.her:
            index = 7 if self.small_model else 8
            return OrderedDict([
                ('observation', observation.copy()),
                ('achieved_goal', observation[-index - 2 * len(self.rooms): -index - len(self.rooms)].copy()),
                ('desired_goal', self.desired_goal.copy())])
        else:
            return observation

    def step(self, action, rule_based: bool = False):
        """
        Function used to make a step according to the chosen action by the algorithm
        """
        observation = self.current_data[self.n_autoregression + self.current_step, self.observation_columns]
        assert (observation[-2] == 0.9) | (observation[-2] == 0.1), "Impossible!"

        if not self.battery:
            action = np.append(action, 0.)

        # We firstly need to rescale actions to 0.1-0.9 because this is what our UMAR model requires as input
        # And we also rescale battery actions to lie within bounds
        if not rule_based:

            action = self.scale_action(observation[-2], action)

            if self.backup:
                action = self.check_backup(observation[:-1], action)

        # Record the valves openings in the data
        self.current_data[self.n_autoregression + self.current_step, self.control_columns] = action[:-1]
        
        if self.her:
            observation = np.array([])
            for x in self.autoregressive_terms:
                observation = np.append(observation, self.current_data[
                    self.n_autoregression + self.current_step - x, self.autoregressive_columns])
            observation = np.append(observation,
                                    self.current_data[self.n_autoregression + self.current_step, self.observation_columns])

        # Recall the battery power input and compute its impact on the SoC
        if self.battery:
            self.battery_powers.append(action[-1])
            # self.battery_soc.append((self.battery_soc[-1] + self.battery_model.predict(action[-1])).clip(0, 100))
            self.battery_soc.append(self.battery_soc[-1] + action[-1] / 60 * self.umar_model.interval)

            # Append the battery observation to the other one
            observation = np.append(observation, self.battery_soc[-1])

        else:
            observation = np.append(observation, 0.)

        observation = self._get_observation(observation)
        
        # Compute the reward using the custom function
        if self.her:
            reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], None)
        else:
            reward = self._compute_reward(self, observation, action)
            
        self.rewards.append(reward)

        output = {}
        for room in self.rooms:

            self.input_tensor[room][:, self.n_autoregression + self.current_step,
            self.umar_model.room_models[room].pred_column] = self.base[room].squeeze()
            self.input_tensor[room][:, self.n_autoregression + self.current_step,
            self.umar_model.room_models[room].ctrl_column] = action[np.where(str(room) in self.rooms)[0]].item()
            (base, heating, cooling), (self.h[room], self.c[room]) = self.umar_model.room_models[room].model(
                self.input_tensor[room][:, self.n_autoregression + self.current_step, :].view(
                    self.input_tensor[room].shape[0], 1, -1), self.h[room], self.c[room])

            output[f"Energy room {room}"] = torch.ones(self.input_tensor[room].shape[0], 1).to(self.device)
            output[f"Energy room {room}"] *= self.umar_model.room_models[room].zero_energy

            output[f"Thermal temperature measurement {room}"] = base
            self.base[room] = base

            if heating is not None:
                self.past_heating[room] += heating["Temperature"]
                output[f"Thermal temperature measurement {room}"] += self.past_heating[room]
                output[f"Energy room {room}"] += heating["Energy"]
            if cooling is not None:
                self.past_cooling[room] += cooling["Temperature"]
                output[f"Thermal temperature measurement {room}"] -= self.past_cooling[room]
                output[f"Energy room {room}"] -= cooling["Energy"]
        prediction = np.array([output[x].squeeze().detach().numpy()
                               for x in self.umar_model.data.columns
                               if (x in self.components) & (x[-3:] in self.rooms)])
        prediction = prediction.clip(0.1, 0.9)

        # Advance of one step
        self.current_step += 1

        # Input the predictions made for this new step
        self.current_data[self.n_autoregression + self.current_step, self.predictions_columns] = prediction

        observation = np.array([])
        if self.her:
            for x in self.autoregressive_terms:
                observation = np.append(observation, self.current_data[
                    self.n_autoregression + self.current_step - x, self.autoregressive_columns])
        observation = np.append(observation,
                                self.current_data[self.n_autoregression + self.current_step, self.observation_columns])

        # Recall the battery power input and compute its impact on the SoC
        if self.battery:
            observation = np.append(observation, self.battery_soc[-1])
        else:
            observation = np.append(observation, 0.)

        self.electricity_imports.append(self.compute_electricity_from_grid(observation, action))

        observation = self._get_observation(observation)

        # Check if we are at the end of an episode, i.e. at the end of the sequence being analyzed
        done = True if self.current_step == len(self.current_sequence) - self.n_autoregression - 1 else False

        # Can use this to recall information
        info = {}

        if done:
            self.last_sequence = self.current_sequence.copy()
            self.last_goal_number = self.goal_number
            self.last_rewards = self.rewards.copy()
            self.last_data = self.current_data.copy()
            self.last_electricity_imports = self.electricity_imports.copy()
            self.last_battery_soc = self.battery_soc.copy()
            self.last_battery_powers = self.battery_powers.copy()

        # Return everything needed for gym
        return observation, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        return - np.sum(np.abs(achieved_goal - desired_goal))

    def get_columns(self):
        """
        Helper function to get the columns of the data corresponding to different needed groups
        Returns the columns corresponding to observations, controls and predictions
        """

        columns = []
        autoregressive = []
        for component in self.components:
            for sensor in self.umar_model.components_inputs_dict[component]:
                if sensor not in columns:
                    columns.append(sensor)
                    if sensor in self.umar_model.autoregressive:
                        autoregressive.append(sensor)
        columns.append("Electricity price")
        rooms = [f"Thermal valve {room}" for room in self.rooms]

        observation_columns = [i for i in range(len(self.umar_model.data.columns))
                               if ("valve" not in self.umar_model.data.columns[i]) &
                               (self.umar_model.data.columns[i] in columns)]
        observation_columns = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13]

        autoregressive_columns = [i for i in range(len(self.umar_model.data.columns))
                                  if ("valve" not in self.umar_model.data.columns[i]) &
                                  (self.umar_model.data.columns[i] in autoregressive)]
        control_columns = [i for i in range(len(self.umar_model.data.columns))
                           if self.umar_model.data.columns[i] in rooms]
        predictions_columns = [i for i in range(len(self.umar_model.data.columns))
                               if self.umar_model.data.columns[i] in self.components]

        return observation_columns, autoregressive_columns, control_columns, predictions_columns

    def compute_scaled_limits(self):
        """
        Function to compute the scaled version of 0 for the energy consumption.
        This also transforms temperature bounds from human-readable form (e.g. 21, 23 degrees) to
        the scaled version between 0.1 and 0.9.
        Note that this is different for each room, since their min and max temperatures differ.
        """

        # Computation of the scaled value for zero energy
        # zero_energy = (-self.umar_model.dataset.min_["Thermal total energy"]) /\
        #             (self.umar_model.dataset.max_["Thermal total energy"] -
        #             self.umar_model.dataset.min_["Thermal total energy"]) * 0.8 + .1

        # Build an array that will contain the lower and higher bound values for each room
        scaled_temp_bounds = np.zeros((2, len(self.rooms)))

        # Loop over the rooms
        i = 0
        for room in self.components:
            # Compute the scaled bounds and store them
            if "temperature" in room:
                scaled_temp_bounds[:, i] = (self.temp_bounds - self.umar_model.min_[room]) / \
                                           (self.umar_model.max_[room] - self.umar_model.min_[room]) \
                                           * 0.8 + .1
                i += 1

        self.temp_bounds = np.array([self.temp_bounds[0]] * len(self.rooms) + [self.temp_bounds[1]] * len(self.rooms)). \
            reshape(2, len(self.rooms))

        # Return everything
        return scaled_temp_bounds  # zero_energy, scaled_temp_bounds

    def get_min_maxes(self):
        min_maxes = np.zeros((2, len(self.rooms)))
        i = 0
        for room in self.components:
            # Compute the scaled bounds and store them
            if "temperature" in room:
                min_maxes[:, i] = [self.umar_model.min_[room], self.umar_model.max_[room]]
                i += 1
        return min_maxes

    def scale_action(self, case, action):

        if self.discrete:
            # Discrete action are recorded as integers and need to be put to floats for the next manipulations to work
            action = action.astype(np.float64)
            # Rescale the valves openings to 0.1-0.9
            if self.umar_model.heating & self.umar_model.cooling:
                if case > 0.8999:
                    action[:-1] = 0.8 * (2 * action[:-1] / (self.discrete_actions[:-1] - 1) - 1).clip(0., 1.) + 0.1
                else:
                    action[:-1] = 0.8 * (-(2 * action[:-1] / (self.discrete_actions[:-1] - 1) - 1)).clip(0., 1.) + 0.1
            else:
                action[:-1] = 0.8 * action[:-1] / (self.discrete_actions[:-1] - 1) + 0.1
            # Rescale the battery power to [-max, max]
            if self.battery:
                action[-1] = self.battery_max_power * 2 * (action[-1] / (self.discrete_actions[-1] - 1) - 0.5)

        elif not self.ddpg:
            # Rescale the valves openings to 0.1-0.9
            if self.umar_model.heating & self.umar_model.cooling:
                if case > 0.8999:
                    action[:-1] = 0.8 * (action[:-1].clip(0., 1.)) + 0.1
                else:
                    action[:-1] = 0.8 * ((-action[:-1]).clip(0., 1.)) + 0.1
            else:
                action[:-1] = (action[:-1] + 1) * 0.4 + 0.1
            # Rescale the battery power to [-max, max]
            if self.battery:
                action[-1] = self.battery_max_power * action[-1]

        else:
            # Nothing to do in the DDPG case as the actions already lie in the right intervals
            pass

        return action

    def check_backup(self, observation, action):

        if self.small_model:
            index = 6
        else:
            index = 7

        if observation[-2] > 0.8999:
            no_heating = np.where(self.scaled_temp_bounds[1, :] <
                                  observation[-(index + 2 * len(self.rooms)):-(index + len(self.rooms))])[0]
            full_heating = np.where(self.scaled_temp_bounds[0, :] >
                                    observation[-(index + 2 * len(self.rooms)):-(index + len(self.rooms))])[0]
            action[:-1][no_heating] = 0.1
            action[:-1][full_heating] = 0.9
        else:
            no_cooling = np.where(self.scaled_temp_bounds[0, :] >
                                  observation[-(index + 2 * len(self.rooms)):-(index + len(self.rooms))])[0]
            full_cooling = np.where(self.scaled_temp_bounds[1, :] <
                                    observation[-(index + 2 * len(self.rooms)):-(index + len(self.rooms))])[0]
            action[:-1][no_cooling] = 0.1
            action[:-1][full_cooling] = 0.9

        if self.battery:
            if action[-1] > 0:
                margin = self.battery_margins[1] - self.battery_soc[-1]
                if action[-1] > margin:
                    action[-1] = margin
            elif action[-1] < 0:
                margin = self.battery_soc[-1] - self.battery_margins[0]
                if action[-1] < - margin:
                    action[-1] = - margin

        return action

    def scale_back_temperatures(self, data):
        return self.min_maxes[0, :] + (data - 0.1) * (self.min_maxes[1, :] - self.min_maxes[0, :]) / 0.8

    def compute_electricity_from_grid(self, observation, action):
        energies = observation[-(3 + len(self.rooms)):-3]

        electricity_consumption = observation[self.elec_column]

        electricity_consumption = self.umar_model.min_["Electricity total consumption"] + (
                electricity_consumption - 0.1) * (self.umar_model.max_["Electricity total consumption"] -
                                                  self.umar_model.min_["Electricity total consumption"]) / 0.8
        energy = abs(sum([self.umar_model.min_[f"Energy room {room}"] + (energies[i] - 0.1) * (
                self.umar_model.max_[f"Energy room {room}"] - self.umar_model.min_[f"Energy room {room}"]) / 0.8 for
                          i, room in enumerate(self.rooms)]))

        if self.battery:
            return energy / self.COP + action[-1]  # + electricity_consumption 
        else:
            return energy / self.COP  # + electricity_consumption

    def training_setup(self):
        """
        Small helper function to detect GPU and make it work on Colab
        """

        # Detect GPU and define te device accordingly
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("GPU acceleration on!")

        # Local case
        else:
            device = "cpu"

        # Return the computed stuff
        return device

    def render(self, mode='human', force: bool = False):
        """
        Custom render function: here it plots what the agent did during the episode, i.e. the temperature
        evolution of the rooms with the corresponding valves pattern, the energy consumption, as well
        as the battery power and SoC
        """
        if (self.current_step == len(self.current_sequence) - self.n_autoregression - 2) | force:
            data = prepare_performance_plot(env=self,
                                            sequence=self.current_sequence[:-1],
                                            data=self.current_data[:-1, :],
                                            electricity_imports=self.electricity_imports,
                                            battery_soc=self.battery_soc,
                                            battery_powers=self.battery_powers)

            analyze_agent(env=self,
                          name="RL agent",
                          data=data,
                          rewards=self.rewards,
                          electricity_imports=self.electricity_imports,
                          battery_soc=self.battery_soc[1:],
                          battery_powers=self.battery_powers)

    def close(self):
        pass


class UMAREnv_old_bis(gym.Env):
    """Custom Environment of UMAR that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, umar_model, battery_model, agent_kwargs, compute_reward=compute_reward):
        """
        Initialize a custom environment, using the learned models of the battery and of the
        UMAR dynamics.

        Parameters: see 'parameters.py'
        """
        super(UMAREnv, self).__init__()

        # Define the models
        self.umar_model = umar_model
        self.battery_model = battery_model

        # Recall constants
        self.n_autoregression = umar_model.n_autoregression
        self.threshold_length = umar_model.threshold_length
        self.temp_bounds = agent_kwargs["temp_bounds"]
        self.discrete = agent_kwargs["discrete"]
        self.ddpg = agent_kwargs["ddpg"]
        self.her = agent_kwargs["her"]
        self.rooms = agent_kwargs["rooms"]
        self.battery = agent_kwargs["battery"]
        self.discrete_actions = np.append(agent_kwargs["discrete_actions"][:len(self.rooms)],
                                          agent_kwargs["discrete_actions"][-1])
        self.battery_max_power = agent_kwargs["battery_max_power"]
        self.COP = agent_kwargs["COP"]
        self.save_path = agent_kwargs["save_path"]
        self.battery_margins = agent_kwargs["battery_margins"]
        self.battery_size = agent_kwargs["battery_size"]
        self.battery_barrier_penalty = agent_kwargs["battery_barrier_penalty"]
        self.temperature_penalty_factor = agent_kwargs["temperature_penalty_factor"]
        self.backup = agent_kwargs["backup"]
        self.small_model = agent_kwargs["small_model"]

        self.components = [component for room in self.rooms for component
                           in self.umar_model.components if room in component]

        # The data will be handled as a numpy array, so we recall which column is what
        self.observation_columns, self.control_columns, self.predictions_columns = self.get_columns()
        self.elec_column = \
        np.where(self.umar_model.data.columns[self.observation_columns] == f"Electricity total consumption")[0].item()

        # Define the observation space (Recall that the UMAR model needs normalized inputs)
        self.observation_space = spaces.Box(low=np.array([0.099] * len(self.observation_columns) + [0.]),
                                            high=np.array([0.901] * len(self.observation_columns) + [100.]),
                                            dtype=np.float64)

        # Define different action spaces
        if self.discrete:
            if self.battery:
                self.action_space = spaces.MultiDiscrete(self.discrete_actions)
            else:
                self.action_space = spaces.MultiDiscrete(self.discrete_actions[:-1])

        elif self.ddpg:
            if self.battery:
                self.action_space = spaces.Box(low=np.array([0.1] * len(self.rooms) + [-self.battery_max_power]),
                                               high=np.array([0.9] * len(self.rooms) + [self.battery_max_power]),
                                               dtype=np.float64)
            else:
                self.action_space = spaces.Box(low=np.array([0.1] * len(self.rooms)),
                                               high=np.array([0.9] * len(self.rooms)),
                                               dtype=np.float64)
        else:
            if self.battery:
                self.action_space = spaces.Box(low=np.array([-1.] * (len(self.rooms) + 1)),
                                               high=np.array([1.] * (len(self.rooms) + 1)),
                                               dtype=np.float64)
            else:
                self.action_space = spaces.Box(low=np.array([-1.] * len(self.rooms)),
                                               high=np.array([1.] * len(self.rooms)),
                                               dtype=np.float64)

        # Define the reward function
        self.compute_reward = compute_reward

        # Initialize various needed variables
        self.current_sequence = None
        self.last_sequence = None
        self.current_data = None
        self.current_step = 0
        self.h = {}
        self.c = {}
        self.battery_soc = []
        self.battery_powers = []
        self.base = {}
        self.past_heating = {}
        self.past_cooling = {}

        # Define the scaled values needed for the reward computation (as the data for the model of
        # UMAR is normalized)
        # self.zero_energy, self.scaled_temp_bounds = self.compute_scaled_limits(self.temp_bounds)
        self.scaled_temp_bounds = self.compute_scaled_limits()
        self.min_maxes = self.get_min_maxes()

        # Set up the training environment, detecting the presence of GPU
        self.device = self.training_setup()
        self.rewards = []

        self.good_sequences = []
        self.bad_sequences = []

    def reset_test(self, sequence=None):
        """
        Function called at the end of an episode to reset and start a new one. Here we pick a random
        sequence of data from NEST and consider it as an episode
        """
        if sequence is not None:
            self.current_sequence = sequence
        else:
            cont = True
            max_bad = 0
            while cont:
                if max_bad >= 3:
                    sequence = np.random.choice(self.good_sequences)
                    cont = False
                else:
                    lengths = np.array([len(seq) for seq in self.umar_model.train_sequences])
                    sequence = np.random.choice(self.umar_model.train_sequences, p=lengths / sum(lengths))
                    if len(sequence) > self.threshold_length + self.n_autoregression:
                        start = np.random.randint(self.n_autoregression, len(sequence) - self.threshold_length + 1)
                        sequence = sequence[start - self.n_autoregression: start + self.threshold_length]

                    if sequence in self.good_sequences:
                        cont = False
                    elif sequence in self.bad_sequences:
                        cont = True
                        max_bad += 1

                    else:
                        good = []
                        for i, room in enumerate(self.rooms):
                            # Put the valves to the closed state
                            self.umar_model.room_models[room].data.loc[self.umar_model.room_models[room].data.index[sequence], f"Thermal valve {room}"] = 0.1
                            # Compute the predictions in that case
                            y, _ = self.umar_model.room_models[room].predict_sequence(sequence)
                            all_close = y[f"Thermal temperature measurement {room}"][-1]

                            # Put the valves to the opened state
                            self.umar_model.room_models[room].data.loc[self.umar_model.room_models[room].data.index[sequence], f"Thermal valve {room}"] = 0.9
                            # Compute the predictions in that case
                            y, _ = self.umar_model.room_models[room].predict_sequence(sequence)
                            all_open = y[f"Thermal temperature measurement {room}"][-1]

                            if self.umar_model.room_models[room].data.loc[self.umar_model.room_models[room].data.index[sequence], "Case"][-1] >= 0.89999:
                                good.append(all_close < (self.temp_bounds[0, i] + self.temp_bounds[1, i]) / 2 < all_open)
                            else:
                                good.append(all_close > (self.temp_bounds[0, i] + self.temp_bounds[1, i]) / 2 > all_open)
                        if np.all(good):
                            self.good_sequences.append(sequence)
                            cont = False
                        else:
                            self.bad_sequences.append(sequence)
                            cont = True

            self.current_sequence = sequence

        # Put the current step to 0
        self.current_step = 0
        self.electricity_imports = []
        self.rewards = []

        self.current_data = self.umar_model.data.iloc[self.current_sequence[self.n_autoregression:], :].copy().values

        # Put the control actions (i.e. the valves) and tee predictions (i.e. the room temperatures, as
        # they will e defined by the model) to nans
        self.current_data[1:, self.predictions_columns] = np.nan
        self.current_data[:, self.control_columns] = np.nan

        # Define the current observation of the agent from UMAR data
        output = {}
        self.input_tensor = {}
        for room in self.rooms:

            self.input_tensor[room], _ = self.umar_model.room_models[room].build_input_output_from_sequences(
                sequences=[self.current_sequence])
            (base, heating, cooling), (self.h[room], self.c[room]) = self.umar_model.room_models[room].model(
                self.input_tensor[room][:, :self.n_autoregression, :])
            output[f"Energy room {room}"] = torch.ones(self.input_tensor[room].shape[0], 1).to(self.device)
            output[f"Energy room {room}"] *= self.umar_model.room_models[room].zero_energy

            output[f"Thermal temperature measurement {room}"] = base
            self.base[room] = base

            if heating is not None:
                self.past_heating[room] = heating["Temperature"]
                output[f"Thermal temperature measurement {room}"] += self.past_heating[room]
                output[f"Energy room {room}"] += heating["Energy"]
            if cooling is not None:
                self.past_cooling[room] = cooling["Temperature"]
                output[f"Thermal temperature measurement {room}"] -= self.past_cooling[room]
                output[f"Energy room {room}"] -= cooling["Energy"]

        prediction = np.array([output[x].squeeze().detach().numpy()
                               for x in self.umar_model.data.columns
                               if (x in self.components) & (x[-3:] in self.rooms)])
        prediction = prediction.clip(0.1, 0.9)

        # Input the predictions made for this new step
        self.current_data[0, self.predictions_columns] = prediction

        # Get the new UMAR observation
        observation = self.current_data[0, self.observation_columns]

        # Reinitialize battery information and get the first SoC measurement (i.e. first observation)
        if self.battery:
            self.battery_soc = []
            self.battery_powers = []
            self.battery_soc.append(
                self.battery_model.data.loc[self.umar_model.data.index[self.current_sequence[0]]][0])

            # Append the battery observation
            observation = np.append(observation, self.battery_soc[0])
        else:
            observation = np.append(observation, 0.)

        # Return the first observation to start the episode, as per gym requirements
        return observation

    def reset(self, sequence=None):
        """
        Function called at the end of an episode to reset and start a new one. Here we pick a random
        sequence of data from NEST and consider it as an episode
        """
        if sequence is not None:
            self.current_sequence = sequence
        else:
            lengths = np.array([len(seq) for seq in self.umar_model.train_sequences])
            sequence = np.random.choice(self.umar_model.train_sequences, p=lengths/sum(lengths))
            if len(sequence) > self.threshold_length + self.n_autoregression:
                start = np.random.randint(self.n_autoregression, len(sequence) - self.threshold_length + 1)
                self.current_sequence = sequence[start - self.n_autoregression: start + self.threshold_length]
            else:
                self.current_sequence = sequence

        # Put the current step to 0
        self.current_step = 0
        self.electricity_imports = []
        self.rewards = []

        self.current_data = self.umar_model.data.iloc[self.current_sequence[self.n_autoregression:], :].copy().values

        # Put the control actions (i.e. the valves) and tee predictions (i.e. the room temperatures, as
        # they will e defined by the model) to nans
        self.current_data[1:, self.predictions_columns] = np.nan
        self.current_data[:, self.control_columns] = np.nan

        # Define the current observation of the agent from UMAR data
        output = {}
        self.input_tensor = {}
        for room in self.rooms:

            self.input_tensor[room], _ = self.umar_model.room_models[room].build_input_output_from_sequences(
                sequences=[self.current_sequence])
            (base, heating, cooling), (self.h[room], self.c[room]) = self.umar_model.room_models[room].model(
                self.input_tensor[room][:, :self.n_autoregression, :])
            output[f"Energy room {room}"] = torch.ones(self.input_tensor[room].shape[0], 1).to(self.device)
            output[f"Energy room {room}"] *= self.umar_model.room_models[room].zero_energy

            output[f"Thermal temperature measurement {room}"] = base
            self.base[room] = base

            if heating is not None:
                self.past_heating[room] = heating["Temperature"]
                output[f"Thermal temperature measurement {room}"] += self.past_heating[room]
                output[f"Energy room {room}"] += heating["Energy"]
            if cooling is not None:
                self.past_cooling[room] = cooling["Temperature"]
                output[f"Thermal temperature measurement {room}"] -= self.past_cooling[room]
                output[f"Energy room {room}"] -= cooling["Energy"]

        prediction = np.array([output[x].squeeze().detach().numpy()
                               for x in self.umar_model.data.columns
                               if (x in self.components) & (x[-3:] in self.rooms)])
        prediction = prediction.clip(0.1, 0.9)

        # Input the predictions made for this new step
        self.current_data[0, self.predictions_columns] = prediction

        # Get the new UMAR observation
        observation = self.current_data[0, self.observation_columns]

        # Reinitialize battery information and get the first SoC measurement (i.e. first observation)
        if self.battery:
            self.battery_soc = []
            self.battery_powers = []
            self.battery_soc.append(
                self.battery_model.data.loc[self.umar_model.data.index[self.current_sequence[0]]][0])

            # Append the battery observation
            observation = np.append(observation, self.battery_soc[0])
        else:
            observation = np.append(observation, 0.)

        # Return the first observation to start the episode, as per gym requirements
        return observation

    def step(self, action, rule_based: bool = False):
        """
        Function used to make a step according to the chosen action by the algorithm
        """
        observation = self.current_data[self.current_step, self.observation_columns]
        assert (observation[-2] == 0.9) | (observation[-2] == 0.1), "Impossible!"

        if not self.battery:
            action = np.append(action, 0.)

        # We firstly need to rescale actions to 0.1-0.9 because this is what our UMAR model requires as input
        # And we also rescale battery actions to lie within bounds
        if not rule_based:

            action = self.scale_action(observation[-2], action)

            if self.backup:
                action = self.check_backup(observation[:-1], action)


        # Record the valves openings in the data
        self.current_data[self.current_step, self.control_columns] = action[:-1]

        output = {}
        for room in self.rooms:

            self.input_tensor[room][:, self.n_autoregression + self.current_step,
            self.umar_model.room_models[room].pred_column] = self.base[room].squeeze()
            self.input_tensor[room][:, self.n_autoregression + self.current_step,
            self.umar_model.room_models[room].ctrl_column] = action[np.where(str(room) in self.rooms)[0]].item()
            (base, heating, cooling), (self.h[room], self.c[room]) = self.umar_model.room_models[room].model(
                self.input_tensor[room][:, self.n_autoregression + self.current_step, :].view(
                    self.input_tensor[room].shape[0], 1, -1), self.h[room], self.c[room])

            output[f"Energy room {room}"] = torch.ones(self.input_tensor[room].shape[0], 1).to(self.device)
            output[f"Energy room {room}"] *= self.umar_model.room_models[room].zero_energy

            output[f"Thermal temperature measurement {room}"] = base
            self.base[room] = base

            if heating is not None:
                self.past_heating[room] += heating["Temperature"]
                output[f"Thermal temperature measurement {room}"] += self.past_heating[room]
                output[f"Energy room {room}"] += heating["Energy"]
            if cooling is not None:
                self.past_cooling[room] += cooling["Temperature"]
                output[f"Thermal temperature measurement {room}"] -= self.past_cooling[room]
                output[f"Energy room {room}"] -= cooling["Energy"]
        prediction = np.array([output[x].squeeze().detach().numpy()
                               for x in self.umar_model.data.columns
                               if (x in self.components) & (x[-3:] in self.rooms)])
        prediction = prediction.clip(0.1, 0.9)

        # Advance of one step
        self.current_step += 1

        # Input the predictions made for this new step
        self.current_data[self.current_step, self.predictions_columns] = prediction

        # Get the new UMAR observation
        observation = self.current_data[self.current_step, self.observation_columns]

        # Recall the battery power input and compute its impact on the SoC
        if self.battery:
            self.battery_powers.append(action[-1])
            # self.battery_soc.append((self.battery_soc[-1] + self.battery_model.predict(action[-1])).clip(0, 100))
            self.battery_soc.append(self.battery_soc[-1] + action[-1] / 60 * self.umar_model.interval)

            # Append the battery observation to the other one
            observation = np.append(observation, self.battery_soc[-1])

        else:
            observation = np.append(observation, 0.)

        # Compute the reward using the custom function
        reward, electricity_from_grid = self.compute_reward(self, observation, action)

        self.electricity_imports.append(electricity_from_grid)
        self.rewards.append(reward)

        # Check if we are at the end of an episode, i.e. at the end of the sequence being analyzed
        done = True if self.current_step == len(self.current_sequence) - self.n_autoregression - 1 else False

        # Can use this to recall information
        info = {}

        if done:
            self.last_sequence = self.current_sequence.copy()

        # Return everything needed for gym
        return observation, reward, done, info

    def get_columns(self):
        """
        Helper function to get the columns of the data corresponding to different needed groups
        Returns the columns corresponding to observations, controls and predictions
        """

        columns = []
        for component in self.components:
            for sensor in self.umar_model.components_inputs_dict[component]:
                if sensor not in columns:
                    columns.append(sensor)
        columns.append("Electricity price")
        rooms = [f"Thermal valve {room}" for room in self.rooms]

        observation_columns = [i for i in range(len(self.umar_model.data.columns))
                               if ("valve" not in self.umar_model.data.columns[i]) &
                               (self.umar_model.data.columns[i] in columns)]
        observation_columns = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13]

        control_columns = [i for i in range(len(self.umar_model.data.columns))
                           if self.umar_model.data.columns[i] in rooms]
        predictions_columns = [i for i in range(len(self.umar_model.data.columns))
                               if self.umar_model.data.columns[i] in self.components]

        return observation_columns, control_columns, predictions_columns

    def compute_scaled_limits(self):
        """
        Function to compute the scaled version of 0 for the energy consumption.
        This also transforms temperature bounds from human-readable form (e.g. 21, 23 degrees) to
        the scaled version between 0.1 and 0.9.
        Note that this is different for each room, since their min and max temperatures differ.
        """

        # Computation of the scaled value for zero energy
        # zero_energy = (-self.umar_model.dataset.min_["Thermal total energy"]) /\
        #             (self.umar_model.dataset.max_["Thermal total energy"] -
        #             self.umar_model.dataset.min_["Thermal total energy"]) * 0.8 + .1

        # Build an array that will contain the lower and higher bound values for each room
        scaled_temp_bounds = np.zeros((2, len(self.rooms)))

        # Loop over the rooms
        i = 0
        for room in self.components:
            # Compute the scaled bounds and store them
            if "temperature" in room:
                scaled_temp_bounds[:, i] = (self.temp_bounds - self.umar_model.min_[room]) / \
                                           (self.umar_model.max_[room] - self.umar_model.min_[room]) \
                                           * 0.8 + .1
                i += 1

        self.temp_bounds = np.array([self.temp_bounds[0]] * len(self.rooms) + [self.temp_bounds[1]] * len(self.rooms)). \
            reshape(2, len(self.rooms))

        # Return everything
        return scaled_temp_bounds  # zero_energy, scaled_temp_bounds

    def get_min_maxes(self):
        min_maxes = np.zeros((2, len(self.rooms)))
        i = 0
        for room in self.components:
            # Compute the scaled bounds and store them
            if "temperature" in room:
                min_maxes[:, i] = [self.umar_model.min_[room], self.umar_model.max_[room]]
                i += 1
        return min_maxes

    def scale_action(self, case, action):

        if self.discrete:
            # Discrete action are recorded as integers and need to be put to floats for the next manipulations to work
            action = action.astype(np.float64)
            # Rescale the valves openings to 0.1-0.9
            if self.umar_model.heating & self.umar_model.cooling:
                if case > 0.8999:
                    action[:-1] = 0.8 * (2 * action[:-1] / (self.discrete_actions[:-1] - 1) - 1).clip(0., 1.) + 0.1
                else:
                    action[:-1] = 0.8 * (-(2 * action[:-1] / (self.discrete_actions[:-1] - 1) - 1)).clip(0., 1.) + 0.1
            else:
                action[:-1] = 0.8 * action[:-1] / (self.discrete_actions[:-1] - 1) + 0.1
            # Rescale the battery power to [-max, max]
            if self.battery:
                action[-1] = self.battery_max_power * 2 * (action[-1] / (self.discrete_actions[-1] - 1) - 0.5)

        elif not self.ddpg:
            # Rescale the valves openings to 0.1-0.9
            if self.umar_model.heating & self.umar_model.cooling:
                if case > 0.8999:
                    action[:-1] = 0.8 * (action[:-1].clip(0., 1.)) + 0.1
                else:
                    action[:-1] = 0.8 * ((-action[:-1]).clip(0., 1.)) + 0.1
            else:
                action[:-1] = (action[:-1] + 1) * 0.4 + 0.1
            # Rescale the battery power to [-max, max]
            if self.battery:
                action[-1] = self.battery_max_power * action[-1]

        else:
            # Nothing to do in the DDPG case as the actions already lie in the right intervals
            pass

        return action

    def check_backup(self, observation, action):

        if self.small_model:
            index = 6
        else:
            index = 7

        if observation[-2] > 0.8999:
            no_heating = np.where(self.scaled_temp_bounds[1, :] <
                                  observation[-(index + 2 * len(self.rooms)):-(index + len(self.rooms))])[0]
            full_heating = np.where(self.scaled_temp_bounds[0, :] >
                                    observation[-(index + 2 * len(self.rooms)):-(index + len(self.rooms))])[0]
            action[:-1][no_heating] = 0.1
            action[:-1][full_heating] = 0.9
        elif observation[-2] < 0.1001:
            no_cooling = np.where(self.scaled_temp_bounds[0, :] >
                                  observation[-(index + 2 * len(self.rooms)):-(index + len(self.rooms))])[0]
            full_cooling = np.where(self.scaled_temp_bounds[1, :] <
                                    observation[-(index + 2 * len(self.rooms)):-(index + len(self.rooms))])[0]
            action[:-1][no_cooling] = 0.1
            action[:-1][full_cooling] = 0.9

        if self.battery:
            if action[-1] > 0:
                margin = self.battery_margins[1] - self.battery_soc[-1]
                if action[-1] > margin:
                    action[-1] = margin
            elif action[-1] < 0:
                margin = self.battery_soc[-1] - self.battery_margins[0]
                if action[-1] < - margin:
                    action[-1] = - margin

        return action

    def scale_back_temperatures(self, data):
        return self.min_maxes[0, :] + (data - 0.1) * (self.min_maxes[1, :] - self.min_maxes[0, :]) / 0.8

    def compute_electricity_from_grid(self, observation, action):
        energies = observation[-(3 + len(self.rooms)):-3]

        electricity_consumption = observation[self.elec_column]

        electricity_consumption = self.umar_model.min_["Electricity total consumption"] + (
                    electricity_consumption - 0.1) * (self.umar_model.max_["Electricity total consumption"] -
                                                      self.umar_model.min_["Electricity total consumption"]) / 0.8
        energy = abs(sum([self.umar_model.min_[f"Energy room {room}"] + (energies[i] - 0.1) * (
                    self.umar_model.max_[f"Energy room {room}"] - self.umar_model.min_[f"Energy room {room}"]) / 0.8 for
                          i, room in enumerate(self.rooms)]))

        if self.battery:
            return energy / self.COP + electricity_consumption + action[-1]
        else:
            return energy / self.COP + electricity_consumption

    def training_setup(self):
        """
        Small helper function to detect GPU and make it work on Colab
        """

        # Detect GPU and define te device accordingly
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("GPU acceleration on!")

        # Local case
        else:
            device = "cpu"

        # Return the computed stuff
        return device

    def render(self, mode='human', force: bool = False):
        """
        Custom render function: here it plots what the agent did during the episode, i.e. the temperature
        evolution of the rooms with the corresponding valves pattern, the energy consumption, as well
        as the battery power and SoC
        """
        if (self.current_step == len(self.current_sequence) - self.n_autoregression - 2) | force:

            # Define the case we are in
            # Normalize the data back
            index = self.umar_model.data.index[self.current_sequence][self.n_autoregression + 1:-1]
            data = self.umar_model.room_models[self.rooms[0]].dataset.inverse_normalize(
                pd.DataFrame(data=self.current_data[1:-1],
                             columns=self.umar_model.data.columns,
                             index=index))

            # Check if this is a heating or cooling case
            if data["Case"][0] == -1:
                case = "Cooling"
            else:
                case = "Heating"

            # Print the information
            print(f"\n====================================================================")
            print(f"{case} case, starting point: {self.current_sequence[self.n_autoregression]},")
            print(f"\n====================================================================")

            # Define the figure, with several subplots and dates as x-axis
            x = 2 * (len(self.rooms) // 2) + 2 if len(self.rooms) % 2 == 1 else 2 * (len(self.rooms) // 2) + 1
            if self.battery:
                x += 1
            fig, axes = plt.subplots(x, 2, figsize=(16, x * 4), sharex=True)
            fig.autofmt_xdate()

            # Plot the behavior for each room
            for num, (i, j) in enumerate(zip(self.predictions_columns, self.control_columns)):
                # Get the subplot position of the room temperature
                axis = axes[num - (num % 2), num % 2]
                # Plot the temperature evolution, as well as the bounds
                axis.plot(data.iloc[:, i], label="Temperature")
                axis.plot(index, [self.temp_bounds[0, 0]] * len(index), linestyle="dashed", color="black", label="Bounds")
                axis.plot(index, [self.temp_bounds[1, 0]] * len(index), linestyle="dashed", color="black")
                axis.plot(index, [(self.temp_bounds[0, 0] + self.temp_bounds[1, 0]) / 2] * np.ones(len(index)),
                          color="red")
                # To make the plot look good
                axis.legend(prop={'size': 15})
                axis.set_title(f"Room temperature {self.umar_model.data.columns[i][-3:]}", size=22)
                axis.set_ylabel("Degrees ($^\circ$C)", size=20)

                # Get the subplot position of the valves action (just below the temperature)
                axis = axes[num - (num % 2) + 1, num % 2]
                # Plot the valves action
                axis.fill_between(index, 0, data.iloc[:, j] * 100, color="orange")
                # Customize
                axis.set_title(f"Valves {self.umar_model.data.columns[i][-3:]}", size=22)
                axis.set_ylim((0, 100))
                axis.set_ylabel("Opening (%)", size=20)

            if not self.battery:
                x += 1

            # Plot the electricity price
            if len(self.rooms) % 2 == 1:
                axes[x - 3, 1].plot(index, data.loc[:, "Electricity price"])
                axes[x - 3, 1].set_title("Electricity price", size=22)
                axes[x - 3, 1].set_ylabel("Price ($/kWh)", size=20)
            else:
                axes[x - 2, 0].plot(index, data.loc[:, "Electricity price"])
                axes[x - 2, 0].set_title("Electricity price", size=22)
                axes[x - 2, 0].set_ylabel("Price ($/kWh)", size=20)

            # Just below it plot the electricity imports
            axes[x - 2, 1].plot(index, self.electricity_imports)
            axes[x - 2, 1].plot(index, np.zeros(len(index)), linestyle="dashed", color="black")
            axes[x - 2, 1].set_title("Electricity imports/exports from the grid", size=22)
            axes[x - 2, 1].set_ylabel("kWh/15 min", size=20)

            if not self.battery:
                x -= 1

            if self.battery:
                # Plot the battery SoC
                axes[x - 1, 0].plot(index, self.battery_soc[1:])
                axes[x - 1, 0].plot(index, self.battery_margins[0] * np.ones(len(index)), linestyle="dashed",
                                    color="black")
                axes[x - 1, 0].plot(index, self.battery_margins[1] * np.ones(len(index)), linestyle="dashed",
                                    color="black")
                axes[x - 1, 0].set_title("Battery SoC", size=22)
                axes[x - 1, 0].set_ylabel("SoC (%)", size=20)

                # Plot the battery power inputs
                axes[x - 1, 1].plot(index, self.battery_powers)
                axes[x - 1, 1].plot(index, np.zeros(len(index)), linestyle="dashed", color="black")
                axes[x - 1, 1].set_title("Battery Power Input", size=22)
                axes[x - 1, 1].set_ylabel("Power (kW)", size=20)

            # Resize plot ticks and define xlabels
            for i in range(2):
                for j in range(x):
                    axes[j, i].tick_params(axis='y', which='major', labelsize=15)
                if j == x - 1:
                    axes[j, i].tick_params(axis='x', which='major', labelsize=15)
                    axes[j, i].set_xlabel("Time", size=20)

            # Show the plot
            plt.tight_layout()
            plt.show()

            print(f"\nReward: {sum(self.rewards):.2f}")
            temp_violations = np.sum(data.iloc[:, self.predictions_columns[:len(self.rooms)]].values > 23.) + np.sum(
                data.iloc[:, self.predictions_columns[:len(self.rooms)]].values < 21.)
            print(f"Number of timesteps the temperature went out of the 21-23 degrees bound: {temp_violations}\n")
            if self.battery:
                batt_violations = np.sum(np.array(self.battery_soc[1:]) > self.battery_margins[1]) + np.sum(
                    np.array(self.battery_soc[1:]) < self.battery_margins[0])
                print(f"Number of timesteps the battery went out of bound: {batt_violations}")
                charged = np.sum(
                    np.array(self.battery_powers)[np.array(self.battery_powers) > 0]) * 60 / self.umar_model.interval
                discharged = np.sum(
                    np.array(self.battery_powers)[np.array(self.battery_powers) < 0]) * 60 / self.umar_model.interval
                print(f"Battery usage: Charged {charged:.2f} kWh and discharged {discharged:.2f} kWh\n")

            costs = np.sum(np.array(self.electricity_imports)[np.array(self.electricity_imports) > 0] *
                           data.loc[:, "Electricity price"][np.array(self.electricity_imports) > 0])
            imports = np.sum(np.array(self.electricity_imports)[np.array(self.electricity_imports) > 0])
            exports = np.sum(np.array(self.electricity_imports)[np.array(self.electricity_imports) < 0])
            print(f"Grid exchanges: Exported {exports:.2f} kWh and imported {imports:.2f} kWh for {costs:.2f}.-")
            print(f"Total benefits: {exports * 0.1 - costs:.2f}.-\n")

    def close(self):
        pass


class UMAREnv_old(gym.Env):
    """Custom Environment of UMAR that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, umar_model, battery_model, agent_kwargs, compute_reward=compute_reward):
        """
        Initialize a custom environment, using the learned models of the battery and of the
        UMAR dynamics.

        Parameters: see 'parameters.py'
        """
        super(UMAREnv, self).__init__()

        # Define the models
        self.umar_model = umar_model
        self.battery_model = battery_model

        # Recall constants
        self.temp_bounds = agent_kwargs["temp_bounds"]
        self.discrete = agent_kwargs["discrete"]
        self.ddpg = agent_kwargs["ddpg"]
        self.rooms = agent_kwargs["rooms"]
        self.battery = agent_kwargs["battery"]
        if self.battery:
            self.discrete_actions = np.append(agent_kwargs["discrete_actions"][:len(self.rooms)],
                                              agent_kwargs["discrete_actions"][-1])
        else:
            self.discrete_actions = agent_kwargs["discrete_actions"][:len(self.rooms)]
        self.battery_max_power = agent_kwargs["battery_max_power"]
        self.COP = agent_kwargs["COP"]
        self.save_path = agent_kwargs["save_path"]
        self.battery_margins = agent_kwargs["battery_margins"]
        self.battery_size = agent_kwargs["battery_size"]
        self.backup = agent_kwargs["backup"]

        self.components = [component for room in self.rooms for component
                           in self.umar_model.components if room in component]

        # The data will be handled as a numpy array, so we recall which column is what
        self.observation_columns, self.control_columns, self.predictions_columns = self.get_columns()

        # Define the observation space (Recall that the UMAR model needs normalized inputs)
        self.observation_space = spaces.Box(low=np.array([0.099] * len(self.observation_columns) + [0.]),
                                            high=np.array([0.901] * len(self.observation_columns) + [100.]),
                                            dtype=np.float64)

        # Define different action spaces
        if self.discrete:
            self.action_space = spaces.MultiDiscrete(self.discrete_actions)
        elif self.ddpg:
            if self.battery:
                self.action_space = spaces.Box(low=np.array([0.1] * len(self.rooms) + [-self.battery_max_power]),
                                               high=np.array([0.9] * len(self.rooms) + [self.battery_max_power]),
                                               dtype=np.float64)
            else:
                self.action_space = spaces.Box(low=np.array([0.1] * len(self.rooms)),
                                               high=np.array([0.9] * len(self.rooms)),
                                               dtype=np.float64)
        else:
            if self.battery:
                self.action_space = spaces.Box(low=np.array([-1.] * (len(self.rooms) + 1)),
                                               high=np.array([1.] * (len(self.rooms) + 1)),
                                               dtype=np.float64)
            else:
                self.action_space = spaces.Box(low=np.array([-1.] * len(self.rooms)),
                                               high=np.array([1.] * len(self.rooms)),
                                               dtype=np.float64)

        # Define the reward function
        self.compute_reward = compute_reward

        # Initialize various needed variables
        self.current_sequence = None
        self.last_sequence = None
        self.current_data = None
        self.current_step = 0
        self.h = None
        self.c = None
        self.battery_soc = []
        self.battery_powers = []
        self.past_base = None
        self.past_heating = None
        self.past_cooling = None
        self.past_heating_input = None
        self.past_cooling_input = None

        # Put the model in the right mode
        self.umar_model.model.running_agent = True

        # Define the scaled values needed for the reward computation (as the data for the model of
        # UMAR is normalized)
        #self.zero_energy, self.scaled_temp_bounds = self.compute_scaled_limits(self.temp_bounds)
        self.scaled_temp_bounds = self.compute_scaled_limits()
        self.min_maxes = self.get_min_maxes()

        # Set up the training environment, detecting the presence of GPU
        self.device = self.training_setup()
        self.rewards = []

    def reset(self, sequence=None):
        """
        Function called at the end of an episode to reset and start a new one. Here we pick a random
        sequence of data from NEST and consider it as an episode
        """

        #if len(self.rewards) > 0:
         #   print(np.mean(self.rewards))

        if sequence is not None:
            self.current_sequence = sequence
        else:
            # Pick a random sequence of existing historical data
            self.current_sequence = random.choice(self.umar_model.sequences)
        self.current_data = self.umar_model.data.iloc[self.current_sequence, :].copy().values

        # Put the control actions (i.e. the valves) and tee predictions (i.e. the room temperatures, as
        # they will e defined by the model) to nans
        self.current_data[1:, self.predictions_columns] = np.nan
        self.current_data[:, self.control_columns] = np.nan

        # Put the current step to 0
        self.current_step = 0
        self.electricity_imports = []

        # Define the current observation of the agent from UMAR data
        observation = self.current_data[0, self.observation_columns]

        # Reinitialize battery information and get the first SoC measurement (i.e. first observation)
        if self.battery:
            self.battery_soc = []
            self.battery_powers = []
            self.battery_soc.append(self.battery_model.data.loc[self.umar_model.data.index[self.current_sequence[0]]][0])

            # Append the battery observation
            observation = np.append(observation, self.battery_soc[0])
        else:
            observation = np.append(observation, 0)

        # Warm start for the umar_model: build up the LSTM hidden and cell state
        self.past_base = {component: torch.FloatTensor([0]).view(1, 1, 1).to(self.device)
                          for component in self.umar_model.components}
        self.past_heating = {component: torch.FloatTensor([0]).view(1, 1, 1).to(self.device)
                             for component in self.umar_model.components}
        self.past_cooling = {component: torch.FloatTensor([0]).view(1, 1, 1).to(self.device)
                             for component in self.umar_model.components}
        self.past_heating_input = {component: torch.FloatTensor([0.1]).view(1, 1, 1).to(self.device)
                                   for component in self.umar_model.components}
        self.past_cooling_input = {component: torch.FloatTensor([0.1]).view(1, 1, 1).to(self.device)
                                   for component in self.umar_model.components}
        self.h, self.c = self.umar_model.warm_start(sequence=self.current_sequence)

        # Return the first observation to start the episode, as per gym requirements
        return observation

    def step(self, action, rule_based: bool = False):
        """
        Function used to make a step according to the chosen action by the algorithm
        """
        if len(action) == 2:
            print(0, action)
        observation = self.current_data[self.current_step, self.observation_columns]
        assert (observation[-2] == 0.9) | (observation[-2] == 0.1), "Impossible!"

        if not self.battery:
            action = np.append(action, 0.)
        total_reward = 0

        # We firstly need to rescale actions to 0.1-0.9 because this is what our UMAR model requires as input
        # And we also rescale battery actions to lie within bounds
        if not rule_based:

            if self.discrete:
                # Discrete action are recorded as integers and need to be put to floats for the next manipulations to work
                action = action.astype(np.float64)
                # Rescale the valves openings to 0.1-0.9
                action[:-1] = 0.8 * action[:-1] / (self.discrete_actions[:-1] - 1) + 0.1
                # Rescale the battery power to [-max, max]
                if self.battery:
                    action[-1] = self.battery_max_power * 2 * (action[-1] / (self.discrete_actions[-1]-1) - 0.5)

            elif not self.ddpg:
                # Rescale the valves openings to 0.1-0.9
                action[:-1] = 0.8 * np.abs(action[:-1]) + 0.1
                # Rescale the battery power to [-max, max]
                if self.battery:
                    action[-1] = self.battery_max_power * action[-1]

            else:
                # Nothing to do in the DDPG case as the actions already lie in the right intervals
                pass

            if len(action) == 3:
                print(1, action)

            if observation[-2] == 0.9:
                no_heating = np.where(self.scaled_temp_bounds[1, :] <
                                      observation[-(7 + 2*len(self.rooms)):-(7 + len(self.rooms))])[0]
                total_reward -= len(np.where(action[:-1][no_heating] != 0.1)[0])
                full_heating = np.where(self.scaled_temp_bounds[0, :] >
                                        observation[-(7 + 2*len(self.rooms)):-(7 + len(self.rooms))])[0]
                total_reward -= len(np.where(action[:-1][full_heating] != 0.9)[0])
            else:
                no_cooling = np.where(self.scaled_temp_bounds[0, :] >
                                      observation[-(7 + 2*len(self.rooms)):-(7 + len(self.rooms))])[0]
                total_reward -= len(np.where(action[:-1][no_cooling] != 0.1)[0])
                full_cooling = np.where(self.scaled_temp_bounds[1, :] <
                                        observation[-(7 + 2*len(self.rooms)):-(7 + len(self.rooms))])[0]
                total_reward -= len(np.where(action[:-1][full_cooling] != 0.9)[0])

            if self.battery:
                if action[-1] > 0:
                    margin = 0.96 / self.battery_size * (self.battery_margins[1] - self.battery_soc[-1])\
                             * 60 / self.umar_model.dataset.interval
                    if action[-1] > margin:
                        total_reward -= 1
                    limit = 0.96 / self.battery_size * (100 - self.battery_soc[-1]) \
                             * 60 / self.umar_model.dataset.interval
                    if action[-1] > limit:
                        action[-1] = limit
                if action[-1] < 0:
                    margin = 0.96 / self.battery_size * (self.battery_soc[-1] - self.battery_margins[0])\
                             * 60 / self.umar_model.dataset.interval
                    if action[-1] < -margin:
                        total_reward -= 1
                    limit = 0.96 / self.battery_size * (100 - self.battery_soc[-1]) \
                             * 60 / self.umar_model.dataset.interval
                    if action[-1] < -limit:
                        action[-1] = -limit
                else:
                    margin = 0

            if len(action) == 3:
                print(2, action)

            if self.backup:

                if observation[-2] == 0.9:
                    action[:-1][no_heating] = 0.1
                    action[:-1][full_heating] = 0.9
                else:
                    action[:-1][no_cooling] = 0.1
                    action[:-1][full_cooling] = 0.9

                if len(action) == 3:
                    print(3, action)
                if self.battery:
                    if (action[-1] > 0) & (action[-1] > margin):
                        action[-1] = margin
                    if (action[-1] < 0) & (action[-1] < -margin):
                        action[-1] = -margin

        # Record the valves openings in the data
        try:
            self.current_data[self.current_step, self.control_columns] = action[:-1]
        except:
            print(action)
            print(self.current_data[self.current_step, self.control_columns])

        # Use this data as input for the UMAR model
        input_tensor = torch.FloatTensor(self.current_data[self.current_step, :-1]).view(1, 1, -1).to(self.device)

        # Compute the prediction of the model, i.e. the next room temperatures and energy consumption
        output, (self.h, self.c), self.past_base, self.past_heating, self.past_cooling,\
        self.past_heating_input, self.past_cooling_input = \
            self.umar_model.model(input_tensor,
                                  h_0=self.h,
                                  c_0=self.c,
                                  past_base=self.past_base,
                                  past_heating=self.past_heating,
                                  past_cooling=self.past_cooling,
                                  past_heating_input=self.past_heating_input,
                                  past_cooling_input=self.past_cooling_input)

        # Put the predictions in the wanted form of an array and make sure they lie in the right interval
        prediction = np.array([output[x].squeeze().detach().numpy()
                               for x in self.umar_model.data.columns
                               if x in self.components])
        prediction = prediction.clip(0.1, 0.9)

        # Advance of one step
        self.current_step += 1

        # Input the predictions made for this new step
        self.current_data[self.current_step, self.predictions_columns] = prediction

        # Get the new UMAR observation
        observation = self.current_data[self.current_step, self.observation_columns]

        # Recall the battery power input and compute its impact on the SoC
        if self.battery:
            self.battery_powers.append(action[-1])
            #self.battery_soc.append((self.battery_soc[-1] + self.battery_model.predict(action[-1])).clip(0, 100))
            self.battery_soc.append(self.battery_soc[-1] + action[-1] / 60 * self.umar_model.dataset.interval)

            # Append the battery observation to the other one
            observation = np.append(observation, self.battery_soc[-1])

        else:
            observation = np.append(observation,0.)

        # Compute the reward using the custom function
        reward, electricity_from_grid = self.compute_reward(self, observation, action, rule_based=rule_based)
        total_reward += reward

        self.electricity_imports.append(electricity_from_grid)

        # Check if we are at the end of an episode, i.e. at the end of the sequence being analyzed
        done = True if self.current_step == len(self.current_sequence) - 1 else False

        # Can use this to recall information
        info = {}

        if done:
            self.last_sequence = self.current_sequence.copy()

        # Return everything needed for gym
        return observation, total_reward, done, info

    def temperature_violations(self):
        """
        Small helper function to compute temperature violations
        """

        # Get the current temperatures
        temperatures = self.scale_back_temperatures(self.current_data[self.current_step, self.predictions_columns])

        # Compute the lower violations, only keep the values if there is a violation
        lower_violations = (self.temp_bounds[0, :] - temperatures)
        lower_violations = np.sum(lower_violations[lower_violations > 0]) + len(lower_violations[lower_violations > 0])
        # Compute the higher violations, only keep the values if there is a violation
        higher_violations = (temperatures - self.temp_bounds[1, :])
        higher_violations = np.sum(higher_violations[higher_violations > 0]) + len(higher_violations[higher_violations > 0])

        # Return them
        return lower_violations, higher_violations

    def get_columns(self):
        """
        Helper function to get the columns of the data corresponding to different needed groups
        Returns the columns corresponding to observations, controls and predictions
        """

        columns = []
        for component in self.components:
            for sensor in self.umar_model.components_inputs_dict[component]:
                if sensor not in columns:
                    columns.append(sensor)
        columns.append("Electricity price")
        rooms = [f"Thermal valve {room}" for room in self.rooms]

        observation_columns = [i for i in range(len(self.umar_model.data.columns))
                               if ("valve" not in self.umar_model.data.columns[i]) &
                               (self.umar_model.data.columns[i] in columns)]
        observation_columns = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13]

        control_columns = [i for i in range(len(self.umar_model.data.columns))
                           if self.umar_model.data.columns[i] in rooms]
        predictions_columns = [i for i in range(len(self.umar_model.data.columns))
                               if self.umar_model.data.columns[i] in
                               self.components]

        return observation_columns, control_columns, predictions_columns

    def compute_scaled_limits(self):
        """
        Function to compute the scaled version of 0 for the energy consumption.
        This also transforms temperature bounds from human-readable form (e.g. 21, 23 degrees) to
        the scaled version between 0.1 and 0.9.
        Note that this is different for each room, since their min and max temperatures differ.
        """

        # Computation of the scaled value for zero energy
        #zero_energy = (-self.umar_model.dataset.min_["Thermal total energy"]) /\
         #             (self.umar_model.dataset.max_["Thermal total energy"] -
          #             self.umar_model.dataset.min_["Thermal total energy"]) * 0.8 + .1

        # Build an array that will contain the lower and higher bound values for each room
        scaled_temp_bounds = np.zeros((2, len(self.rooms)))

        # Loop over the rooms
        i = 0
        for room in self.components:
            # Compute the scaled bounds and store them
            if "temperature" in room:
                scaled_temp_bounds[:, i] = (self.temp_bounds - self.umar_model.dataset.min_[room]) / \
                                           (self.umar_model.dataset.max_[room] - self.umar_model.dataset.min_[room]) \
                                           * 0.8 + .1
                i += 1

        self.temp_bounds = np.array([self.temp_bounds[0]]*len(self.rooms)+[self.temp_bounds[1]]*len(self.rooms)).\
            reshape(2, len(self.rooms))

        # Return everything
        return scaled_temp_bounds #zero_energy, scaled_temp_bounds

    def get_min_maxes(self):
        min_maxes = np.zeros((2, len(self.rooms)))
        i = 0
        for room in self.components:
            # Compute the scaled bounds and store them
            if "temperature" in room:
                min_maxes[:, i] = [self.umar_model.dataset.min_[room], self.umar_model.dataset.max_[room]]
                i += 1
        return min_maxes

    def scale_back_temperatures(self, data):
        return self.min_maxes[0, :] + (data - 0.1) * (self.min_maxes[1, :] - self.min_maxes[0, :])/0.8

    def training_setup(self):
        """
        Small helper function to detect GPU and make it work on Colab
        """

        # Detect GPU and define te device accordingly
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("GPU acceleration on!")

        # Local case
        else:
            device = "cpu"

        # Return the computed stuff
        return device

    def render(self, mode='human'):
        """
        Custom render function: here it plots what the agent did during the episode, i.e. the temperature
        evolution of the rooms with the corresponding valves pattern, the energy consumption, as well
        as the battery power and SoC
        """
        if self.current_step == len(self.current_sequence) - 2:

            # Define the case we are in
            # Normalize the data back
            index = self.umar_model.data.index[self.current_sequence][1:-1]
            data = self.umar_model.dataset.inverse_normalize(pd.DataFrame(data=self.current_data[1:-1, :],
                                                                          columns=self.umar_model.data.columns,
                                                                          index=index))

            # Check if this is a heating or cooling case
            if data["Case"][0] == -1:
                case = "Cooling"
            else:
                case = "Heating"

            # Print the information
            print(f"\n====================================================================")
            print(f"{case} case, starting point: {self.current_sequence[0]},")
            print(f"\n====================================================================")

            # Define the figure, with several subplots and dates as x-axis
            x = 2 * (len(self.rooms)//2) + 2
            if self.battery:
                x += 1
            fig, axes = plt.subplots(x, 2, figsize=(16, x*4), sharex=True)
            fig.autofmt_xdate()

            # Plot the behavior for each room
            for num, (i, j) in enumerate(zip(self.predictions_columns, self.control_columns)):
                # Get the subplot position of the room temperature
                axis = axes[num - (num % 2), num % 2]
                # Plot the temperature evolution, as well as the bounds
                axis.plot(data.iloc[:, i], label="Temperature")
                axis.plot(index, [self.temp_bounds[0,0]] * len(index), color="red", label="Bounds")
                axis.plot(index, [self.temp_bounds[1,0]] * len(index), color="red")
                # To make the plot look good
                axis.legend(prop={'size': 15})
                axis.set_title(f"Room temperature {self.umar_model.data.columns[i][-3:]}", size=22)
                axis.set_ylabel("Degrees ($^\circ$C)", size=20)

                # Get the subplot position of the valves action (just below the temperature)
                axis = axes[num - (num % 2) + 1, num % 2]
                # Plot the valves action
                axis.fill_between(index, 0, data.iloc[:, j] * 100, color="orange")
                # Customize
                axis.set_title(f"Valves {self.umar_model.data.columns[i][-3:]}", size=22)
                axis.set_ylim((0, 100))
                axis.set_ylabel("Opening (%)", size=20)

            if not self.battery:
                x += 1

            # Plot the electricity price
            axes[x-3, 1].plot(index, data.loc[:, "Electricity price"])
            axes[x-3, 1].set_title("Electricity price", size=22)
            axes[x-3, 1].set_ylabel("Price ($/kWh)", size=20)

            # Just below it plot the electricity imports
            axes[x-2, 1].plot(index, self.electricity_imports)
            axes[x-2, 1].plot(index, np.zeros(len(index)), linestyle="dashed", color="black")
            axes[x-2, 1].set_title("Electricity imports/exports from the grid", size=22)
            axes[x-2, 1].set_ylabel("kWh/15 min", size=20)

            if not self.battery:
                x -= 1

            if self.battery:
                # Plot the battery SoC
                axes[x-1, 0].plot(index, self.battery_soc[1:])
                axes[x-1, 0].plot(index, self.battery_margins[0] * np.ones(len(index)), linestyle="dashed", color="black")
                axes[x-1, 0].plot(index, self.battery_margins[1] * np.ones(len(index)), linestyle="dashed", color="black")
                axes[x-1, 0].set_title("Battery SoC", size=22)
                axes[x-1, 0].set_ylabel("SoC (%)", size=20)

                # Plot the battery power inputs
                axes[x-1, 1].plot(index, self.battery_powers)
                axes[x-1, 1].plot(index, np.zeros(len(index)), linestyle="dashed", color="black")
                axes[x-1, 1].set_title("Battery Power Input", size=22)
                axes[x-1, 1].set_ylabel("Power (kW)", size=20)

            # Resize plot ticks and define xlabels
            for i in range(2):
                for j in range(x):
                    axes[j, i].tick_params(axis='y', which='major', labelsize=15)
                if j == x-1:
                    axes[j, i].tick_params(axis='x', which='major', labelsize=15)
                    axes[j, i].set_xlabel("Time", size=20)

            # Show the plot
            plt.tight_layout()
            plt.show()

    def close(self):
        pass
