import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import warnings
import typing
from abc import ABC
from models.Models_mina import Model

from typing import Union, Optional, List, Dict, Any

import gym
from gym import spaces

from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan, VecEnv, sync_envs_normalization
from stable_baselines import logger

if typing.TYPE_CHECKING:
    from stable_baselines.common.base_class import BaseRLModel  # pytype: disable=pyi-error

from models.data_handling import NESTData
from models.NN_models import ARX
from models.recurrent import ModelList
from models.battery_models import prepare_linear_battery_model

from parameters import data_kwargs,  agent_kwargs
from models.parameters import model_kwargs
# from parameters import data_kwargs, model_kwargs, agent_kwargs
from data_preprocessing.dataset import create_time_data
import kosta.hyperparams as hp
from models.parameters import DATA_SAVE_PATH
from models.parameters import parameters
from models.util import load_data

def add_electricity_price(data, price_type, price_levels, hours=None):
    """
    Function to add a column to the data representing the electricity price

    args:
        model:          The model we work with (to access the data)
        price_type:     The pricing scheme to use
        price_levels:   The different prices at different times
        const:          A price constant used in some schemes

    Returns:
        A list of prices corresponding to each datatime Index
    """

    # Get the hour corresponding to each time step
    if hours is None:
        hours = data.index.hour.astype(np.float64)

    levels = np.array(price_levels)
    if len(levels) > 1:
        levels = (levels - price_levels[0]) / (price_levels[-1] - price_levels[0]) * 0.8 + 0.1
    else:
        level = 0.5

    # Constant (thus useless) price
    if price_type == "Constant":
        if len(hours) > 1:
            prices = np.array([level] * len(hours))
        else:
            prices = level

    # Three level pricing: low at night, peaks in the morning and evening and medium (const) price in the middle
    elif price_type == "ThreeLevels":
        prices = np.zeros_like(hours)

        if len(prices) > 1:
            # Low price during the night
            prices[np.where((hours < 6) | (hours > 21))[0]] = levels[0]

            # Const during the day
            prices[np.where((hours > 9) & (hours < 17))[0]] = levels[1]

            # Peak pricing in the morning and evening
            prices[np.where(((hours >= 6) & (hours <= 9)) | ((hours >= 17) & (hours <= 21)))[0]] = levels[2]

        # Needed to make it work when only one timestamp is provided (i.e. when the agent is running live)
        else:
            # Low price during the night
            if (hours < 6) | (hours > 21):
                prices = levels[0]

            # Const during the day
            elif (hours > 9) & (hours < 17):
                prices = levels[1]

            # Peak pricing in the morning and evening
            else:
                prices = levels[2]

    else:
        raise NotImplementedError(f"{price_type} does not exist!")

    # Return it
    return prices


# def add_bounds(data, temp_bounds, cases=None, hours=None):
#     """
#     Function to add a column to the data representing the temperature bounds
#
#     args:
#         umar_model: The model we work with (to access the data)
#         temp_bounds: Temperature bounds
#
#     Returns:
#         A list of prices corresponding to each datatime Index
#     """
#     # Get the hour corresponding to each time step
#     if hours is None:
#         hours = data.index.hour.astype(np.float64)
#
#     if cases is None:
#         cases = data.Case.astype(np.float64)
#     bounds = np.array(temp_bounds)
#     bounds = (bounds - temp_bounds[0]) / (temp_bounds[-1] - temp_bounds[0]) * 0.8 + 0.1
#
#     if hp.dynamic_bounds:
#         if len(hours) > 1:
#             lower_bounds = np.zeros(len(hours))
#             lower_bounds[(hours>8) & (hours<20)] = bounds[0]
#             lower_bounds[(hours<=8) | (hours>=20)] = bounds[1]
#             upper_bounds = np.ones(len(hours))*bounds[2]
#             return lower_bounds, upper_bounds
#         else:
#             raise ValueError("Not implemented yet")
#     if len(hours) > 1:
#         if len(bounds) == 2:
#             lower_bounds = np.ones(len(hours)) * bounds[0]
#             upper_bounds = np.ones(len(hours)) * bounds[1]
#
#         elif len(bounds) == 4:
#             lower_bounds = np.zeros_like(hours)
#             upper_bounds = np.zeros_like(hours)
#             # When its heating (case>0.5): bounds[0] < x < bounds[2],
#             # when its cooling (case<0.5): bounds[1] < x < bounds[3] and between 8 and 20
#             # if between 20 and 8: bounds[1] < x < bounds[2]
#             lower_bounds[np.where((hours >= 8) & (hours < 20) & (cases > 0.5))[0]] = bounds[0]
#             upper_bounds[np.where((hours >= 8) & (hours < 20) & (cases > 0.5))[0]] = bounds[2]
#
#             lower_bounds[np.where((hours >= 8) & (hours < 20) & (cases < 0.5))[0]] = bounds[1]
#             upper_bounds[np.where((hours >= 8) & (hours < 20) & (cases < 0.5))[0]] = bounds[3]
#
#             lower_bounds[np.where((hours < 8) | (hours >= 20))[0]] = bounds[1]
#             upper_bounds[np.where((hours < 8) | (hours >= 20))[0]] = bounds[2]
#
#     # Needed to make it work when only one timestamp is provided (i.e. when the agent is running live)
#     else:
#         if len(bounds) == 2:
#             lower_bounds = np.ones(len(hours)) * bounds[0]
#             upper_bounds = np.ones(len(hours)) * bounds[1]
#
#         elif len(bounds) == 4:
#             if (hours >= 8) & (hours < 20):
#                 if cases > 0.5:
#                     lower_bounds = bounds[0]
#                     upper_bounds = bounds[2]
#                 else:
#                     lower_bounds = bounds[1]
#                     upper_bounds = bounds[3]
#
#             else:
#                 lower_bounds = bounds[1]
#                 upper_bounds = bounds[2]
#
#     # Return it
#     return lower_bounds, upper_bounds
#add bound by mina for pcnn
def add_bounds(model, temp_bounds: list, change_times: list, cases=None, hours=None):
    """
    Function to add a column to the data representing the temperature bounds

    args:
        model:          The model to add bounds to
        temp_bounds:    Temperature bounds to include
        change_times:   When to switch between stricter and looser bounds, e.g. 8h and 20h
        cases:          To provide custom cases (heating or cooling)
        hours:          To provide custom hours

    Returns:
        A list of lower and upper bounds at each time step (normalized)
    """

    # Get the hour and cases corresponding to each time step when needed
    if hours is None:
        hours = model.data.index.hour.astype(np.float64)
    if cases is None:
        cases = model.data.Case.astype(np.float64)

    # Normalize the bounds
    bounds = np.array(temp_bounds)
#    bounds = (bounds - model.normalization_variables['Room'][0].values) / model.normalization_variables['Room'][1].values * 0.8 + 0.1
    bounds = (bounds - temp_bounds[0]) / (temp_bounds[-1] - temp_bounds[0]) * 0.8 + 0.1
    # bounds = (bounds - model.normalization_variables['Room'][0].item()) / model.normalization_variables['Room'][1].item() * 0.8 + 0.1
    # print("pegah")
    # print(bounds)
    # print(len(hours))
    if len(hours) > 1:
        # Case when only two bounds are given (i.e. a lower and an upper one)
        if len(bounds) == 2:
            lower_bounds = np.ones(len(hours)) * bounds[0]
            upper_bounds = np.ones(len(hours)) * bounds[1]

        # Case when stricter and looser bounds are prvided
        elif len(bounds) == 4:
            lower_bounds = np.zeros_like(hours)
            upper_bounds = np.zeros_like(hours)

            # During the "change times"
            # Heating case: loose lower bound, strict upper one
            lower_bounds[np.where((hours >= change_times[0]) & (hours < change_times[1]) & (cases > 0.5))[0]] = bounds[
                0]
            upper_bounds[np.where((hours >= change_times[0]) & (hours < change_times[1]) & (cases > 0.5))[0]] = bounds[
                2]
            # Cooling case: strict lower bound, loose upper one
            lower_bounds[np.where((hours >= change_times[0]) & (hours < change_times[1]) & (cases < 0.5))[0]] = bounds[
                0]
            upper_bounds[np.where((hours >= change_times[0]) & (hours < change_times[1]) & (cases < 0.5))[0]] = bounds[
                2]

            # Outside the changing times, strict upper and lower bound
            lower_bounds[np.where((hours < change_times[0]) | (hours >= change_times[1]))[0]] = bounds[0]
            upper_bounds[np.where((hours < change_times[0]) | (hours >= change_times[1]))[0]] = bounds[2]

    # Needed to make it work when only one timestamp is provided (i.e. when the agent is running live)
    else:
        if len(bounds) == 2:
            lower_bounds = np.ones(len(hours)) * bounds[0]
            upper_bounds = np.ones(len(hours)) * bounds[1]

        elif len(bounds) == 4:
            if (hours >= 8) & (hours < 20):
                if cases > 0.5:
                    lower_bounds = bounds[0]
                    upper_bounds = bounds[2]
                else:
                    lower_bounds = bounds[1]
                    upper_bounds = bounds[3]

            else:
                lower_bounds = bounds[1]
                upper_bounds = bounds[2]

    # Return it
    lower_bounds = bounds[0]
    upper_bounds = bounds[-1]
    return lower_bounds, upper_bounds


def prepare_models(data_kwargs: dict = data_kwargs, model_kwargs: dict = model_kwargs,
                   agent_kwargs: dict = agent_kwargs):
    """
    Function to prepare an agent: load the corresponding battery and UMAR models and define the
    wanted agent in the right form
    """

    # Prepare the battery model
    battery_model = prepare_linear_battery_model(data_kwargs, show_model=False, show_predictions=False)

    # Prepare the NEST model
    nest_data = NESTData(data_kwargs)
    umar_model = ModelList(nest_data=nest_data,
                           model_kwargs=model_kwargs)

    # Add the electricity data
    umar_model.data["Electricity price"] = add_electricity_price(data=umar_model.data,
                                                                 price_type=agent_kwargs["price_type"],
                                                                 price_levels=agent_kwargs["price_levels"])

    # Add the temperature bounds
    umar_model.data["Lower bound"], umar_model.data["Upper bound"] = add_bounds(data=umar_model.data,
                                                                                temp_bounds=agent_kwargs["temp_bounds"])

    for model in umar_model.room_models.values():
        model.model.eval()

        # To make multiprocessing possible: SubProcVecEnv relies on pickle, which cannot serialize complex
        # pytorch objects (Why?) - delete them since we don't need them anyways
        del model.losses
        del model.optimizer
        try:
            del model.writer
        except AttributeError:
            pass

    # Small modification to ensure that we don't have missing values at odd places for the battery SoC
    # i.e. everywhere where the NEST model is defined, we need a SoC value --> done by interpolation

    # First join the battery data frame on the index of the UMAR model (we need the same index for both)

    temp = pd.DataFrame(index=umar_model.room_models[agent_kwargs["rooms"][0]].data.index)
    battery_model.data = temp.join(battery_model.data)
    # Fill missing values
    battery_model.data.iloc[:, 0].interpolate(inplace=True)
    battery_model.data.iloc[:, 0] = battery_model.data.iloc[:, 0].clip(lower=agent_kwargs["battery_margins"][0] + 0.5,
                                                                       upper=agent_kwargs["battery_margins"][1] - 0.5)
    # Models
    return umar_model, battery_model


def prepare_arx_data(model_kwargs, agent_kwargs, save_path: str = None):
    """
    Function to prepare the data for a simple ARX model
    Args:
        model_kwargs:   All model parameters
        agent_kwargs:   All agent parameters
        save_path:      Where to load the data from
    """

    if save_path is None:
        save_path = model_kwargs["save_path"]

    # Get the data and put the time as index
    # data_path = os.path.join(save_path, model_kwargs["model_name"],
    #                          "ambientconditions_" + str(model_kwargs["interval"]) + ".csv")
    data_path = os.path.join(save_path, "PCNN",
                             "ambientconditions_" + str(model_kwargs["interval"]) + ".csv")
    
    data = pd.read_csv(data_path, index_col="timestamp")
    data.index = pd.to_datetime(data.index)

    # Add room temperature and control columns (to fill later)
    data["T_272"] = np.nan
    data["T_273"] = np.nan
    data["T_274"] = np.nan
    data["u_272"] = np.nan
    data["u_273"] = np.nan
    data["u_274"] = np.nan

    # Define the cases, here depending on whether the outside temperature is above 15 (cooling) or below
    # (heating) for entire sequences (i.e. with autoregression and over the horizon)
    cases = [
        (np.mean(data["T_amb"][x - model_kwargs["n_autoregression"]: x + model_kwargs["threshold_length"]]) < 12) * 1 * 2 - 1
        for x in range(model_kwargs["n_autoregression"], len(data) - model_kwargs["threshold_length"])]
    data["Case"] = np.nan
    data["Case"].iloc[model_kwargs["n_autoregression"]: len(data) - model_kwargs["threshold_length"]] = cases

    # Data normalization, where we avoid the room temperature and control columns (all NaNs)
    max_ = np.max(data[[x for x in data.columns if "27" not in x]], axis=0)
    min_ = np.min(data[[x for x in data.columns if "27" not in x]], axis=0)
    data[[x for x in data.columns if "27" not in x]] = 0.8 * (
                data[[x for x in data.columns if "27" not in x]] - min_) / (max_ - min_) + 0.1

    # Add the electricity data
    data["Electricity price"] = add_electricity_price(data=data,
                                                      price_type=agent_kwargs["price_type"],
                                                      price_levels=agent_kwargs["price_levels"])

    # Add the temperature bounds
    data["Lower bound"], data["Upper bound"] = add_bounds(data=data,
                                                          temp_bounds=agent_kwargs["temp_bounds"])
    return data, min_, max_


def prepare_toy_models(data_kwargs, model_kwargs, agent_kwargs, save_path: str = None):
    """
    Small function to create the data and the models for the toy case
    Args:
        data_kwargs:    All data parameters
        model_kwargs:   All model parameters
        agent_kwargs:   All agent parameters
        save_path:      Where to find the data and the model
    Returns:
        One UMAR model and a battery model
    """
    if agent_kwargs['battery']:
        # Prepare the battery model
        battery_model = prepare_linear_battery_model(data_kwargs, show_model=False, show_predictions=False)
    else:
        battery_model = None
    # Prepare the data for the ARX model
    data, min_, max_ = prepare_arx_data(model_kwargs, agent_kwargs, save_path)

    # Prepare the corresponding model
    umar_model = ARX(data, min_, max_, model_kwargs)
    # print("mina")
    # print(umar_model)
    if hp.dynamic_bounds:
        umar_model = create_time_data(umar_model)

    return umar_model, battery_model


def evaluate_lstm_policy(model, env, sequence=None, n_eval_episodes=10, sequences=None, deterministic=True,
                    render=False, callback=None, reward_threshold=None, normalizing: bool = True,
                    return_episode_rewards=False, all_goals=False, init_temps=None):
    """
    Function to evaluate an LSTM policy. This is just an adaptation from the original function
    'evaluate_policy' of stable-baselines, found on github
    https://github.com/hill-a/stable-baselines/issues/780
    The comments were added here
    """

    # Check the input
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    if not isinstance(env, VecEnv):
        curr_env = env
    else:
        if normalizing:
            curr_env = env.venv.venv.envs[0]
        else:
            curr_env = env.venv.envs[0]

    # Loop for the wanted number of episode to evaluate the performance of the policy and recall the rewards
    episode_rewards, episode_lengths = [], []
    comfort_violations = []
    prices = []
    for i in range(n_eval_episodes):
        # Reset everything$

        if sequences is not None:
            sequence = sequences[i]
        else:
            if sequence is not None:
                sequence = sequence
            else:
                lengths = np.array([len(seq) for seq in curr_env.umar_model.test_sequences])
                if len(lengths) > 1:
                    sequence = np.random.choice(curr_env.umar_model.test_sequences, p=lengths / sum(lengths))
                else:
                    sequence = curr_env.umar_model.test_sequences[0]
                if len(sequence) > curr_env.threshold_length + curr_env.n_autoregression:
                    start = np.random.randint(curr_env.n_autoregression,
                                              len(sequence) - curr_env.threshold_length + 1)
                    sequence = sequence[start - curr_env.n_autoregression: start + curr_env.threshold_length]
                else:
                    pass

        if all_goals and curr_env.her:
            counter = len(curr_env.desired_goals)
        else:
            counter = 1

        while counter > 0:
            goal_number = counter - 1 if all_goals and curr_env.her else None
            obs = curr_env.reset(sequence=sequence,
                                 goal_number=goal_number,
                                 init_temp=init_temps[i] if init_temps is not None else None)

            done, state = False, None
            episode_reward = 0.0
            episode_length = 0
            while not done:
                # Transform the observation
                ##### Modification #####
                if not curr_env.her:
                    zero_completed_obs = np.zeros((model.env.num_envs,) + env.observation_space.shape)
                    zero_completed_obs[0, :] = obs
                    obs = zero_completed_obs
                ########################
                # Get the action from the policy
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                if not curr_env.her:
                    action = action[0, :]
                # Do a step and recall the reward
                obs, reward, done, _info = curr_env.step(action)
                episode_reward += reward

                # Call potential callbacks
                if callback is not None:
                    callback(locals(), globals())
                episode_length += 1

                # Show plots if wanted
                if render:
                    curr_env.render()

            # Record stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            comfort_violations.append(np.sum(curr_env.last_comfort_violations))
            prices.append(np.sum(curr_env.last_prices))

            counter -= 1

    # Compute evaluation statistics
    #mean_reward = np.mean(episode_rewards)
    #std_reward = np.std(episode_rewards)

    # Return the wanted information
    if reward_threshold is not None:
        assert np.mean(episode_rewards) > reward_threshold, 'Mean reward below threshold: '\
                                         '{:.2f} < {:.2f}'.format(np.mean(episode_rewards), reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths

    return episode_rewards, comfort_violations, prices, episode_lengths
    #return mean_reward, std_reward


def analyze_agent(name: str, env: gym.GoalEnv, data: pd.DataFrame, rewards: list, comfort_violations: list,
                  prices: list, lower_bounds: list, upper_bounds: list, electricity_imports: list,
                  battery_soc: list, battery_powers: list):
    """
    Helper function printing the results of an agent running over an episode. This uses the 'data'
    prepared by the function 'prepare_performance_plot' which contains most informations about the
    agent, as well as some other mesurements.

    Args:
        name:                   Agent name
        env:                    Environment where the episode was running
        data:                   pd.DataFrame containing all the information of the episode (in particular
                                 the room temperatures and valves openenings)
        rewards:                Obtained rewards
        comfort_violations:     How much out of bounds you were
        prices:                 List of prices paid by the agent while controlling the room
        lower_bounds:           Lower comfort bounds
        upper_bounds:           Upper comfort bounds
        electricity_imports:    Used electricity from the grid
        battery_soc:            Battery SoC over the episode
        battery_powers:         Used battery power
    """
    # Rewards
    print(f"__________________________\n{name}:")
    print(f"\nReward:           {sum(rewards):.2f}")
    print(f"Comfort violations: {np.sum(comfort_violations):.2f} amount of violation.\n")

    # Battery analysis: bounds violations and energy usage
    if env.battery:
        batt_violations = np.sum(np.array(battery_soc) > env.battery_margins[1]) + np.sum(
            np.array(battery_soc) < env.battery_margins[0])
        print(f"Number of timesteps the battery went out of bound: {batt_violations}")
        charged = np.sum(
            np.array(battery_powers)[np.array(battery_powers) > 0]) * 60 / env.umar_model.interval
        discharged = np.sum(
            np.array(battery_powers)[np.array(battery_powers) < 0]) * 60 / env.umar_model.interval
        print(f"Battery usage: Charged {charged:.2f} kWh and discharged {discharged:.2f} kWh\n")

    # Electricity consumption analysis: imports, exports and coses
    costs = np.sum(np.array(electricity_imports)[np.array(electricity_imports) > 0] *
                   data.loc[data.index[:-1], "Electricity price"][np.array(electricity_imports) > 0])
    imports = np.sum(np.array(electricity_imports)[np.array(electricity_imports) > 0])
    exports = np.sum(np.array(electricity_imports)[np.array(electricity_imports) < 0])
    print(f"Exported energy: {exports:.2f} kWh")
    print(f"Imported energy: {imports:.2f} kWh for {costs:.2f}.-")
    print(f"Total benefits/costs: {np.sum(prices):.2f}\n")


def prepare_performance_plot(env: gym.GoalEnv, sequence: list, data: pd.DataFrame, rewards: list,
                             electricity_imports: list, lower_bounds: list, upper_bounds: list, prices: list,
                             comfort_violations: list, battery_soc: list = None, battery_powers: list = None,
                             label: str = None, color: str = "blue", elec_price: bool = True,
                             print_: bool = True, show_: bool = True, axes=None):
    """
    Function to prepare the plot of an agent over one episode. Designed to be chained over several
    agents to get comparative plots.

    Args:
        env:                    Environment where the episode was running
        sequence:               Sequence of data over which the episode ran (to find it in the original dataset)
        data:                   pd.DataFrame containing all the information of the episode (in particular
                                 the room temperatures and valves openings)
        rewards:                Rewards obtained during the episode
        electricity_imports:    Used electricity from the grid
        lower_bounds:           Lower comfort bounds
        upper_bounds:           Upper comfort bounds
        prices:                 List of Energy prices
        comfort_violations:     List of comfort violations encountered
        battery_soc:            Battery SoC over the episode
        battery_powers:         Used battery power
        label:                  Label of the plotted curves
        color:                  Color of the plotted curves
        elec_price:             Flag whether to plot the electricity price
        print_:                 Flag whether to print the case we are in (heating or cooling mode)
        show_:                  Flag whether to show the plot or return the axes
        axes:                   Axes of the plot, if None new ones are created
    """

    # Get the index of the data (i.e. the time at which it happened, refering to the original dataset)
    # and normalize the data back to the original scale
    index = env.umar_model.data.index[sequence][env.n_autoregression:]
    data = env.inverse_normalize(
        pd.DataFrame(data=data[env.n_autoregression:],
                     columns=env.umar_model.data.columns,
                     index=index))

    data.loc[:, "Electricity price"] = (data.loc[:, "Electricity price"] - 0.1) / 0.8 * \
                                       (env.price_levels[-1] - env.price_levels[0]) + env.price_levels[0]

    # One decision is missing (after the last observation) so we add Nones to make the dimensions match
    electricity_imports.append(None)
    battery_powers.append(None)

    # Check if this is a heating or cooling case
    if data["Case"][0] < 0:
        case = "Cooling"
    else:
        case = "Heating"

    # Print the information
    if print_:
        print(f"\n====================================================================")
        print(f"{case} case, starting point: {sequence[env.n_autoregression]},")
        print(f"\n====================================================================")

    # Helper variable x used to make the plot in variable sizes (adapts to the number of
    # rooms and whether the battery is used or not)
    x = len(env.rooms) + 5
    #x = 2 * (len(env.rooms) // 2) + 2 if len(env.rooms) % 2 == 1 else 2 * (len(env.rooms) // 2) + 1
    if env.battery:
        x += 1

    # Define the figure, with several subplots and dates as x-axis
    if axes is None:
        if env.simple:
            fig, axes = plt.subplots(x - 2, 2, figsize=(16, x * 3), sharex=False)
        else:
            fig, axes = plt.subplots(x + 1, 2, figsize=(16, x * 4), sharex=False)
        fig.autofmt_xdate()

    # Plot the behavior for each room
    for num, (i, j) in enumerate(zip(env.predictions_columns, env.control_columns)):

        # Get the subplot position of the room temperature
        axis = axes[num - (num % 2), num % 2]

        # Plot the temperature evolution, as well as the bounds
        if env.her:
            axis.plot(index, [env.temp_bounds[0, 0] + env.goal_number * (env.temp_bounds[1, 0] - env.temp_bounds[0, 0])/2]
                      * np.ones(len(index)), color="red")
        else:
            if data.loc[index[0], "Case"] > 0.5:
                axis.fill_between(index, lower_bounds, [x + 1 for x in lower_bounds], color="grey", alpha=0.05)
                axis.plot(index, [x + 0.5 for x in lower_bounds], color="grey", alpha=0.25, linestyle="dashed")
            else:
                axis.fill_between(index, [x - 1 for x in upper_bounds], upper_bounds, color="grey", alpha=0.05)
                axis.plot(index, [x - 0.5 for x in upper_bounds], color="grey", alpha=0.25, linestyle="dashed")
            axis.plot(index, lower_bounds, color="black", linestyle="dashed")
            axis.plot(index, upper_bounds, color="black", linestyle="dashed")

        axis.plot(data.iloc[:, i], label=label, color=color)

        # To make the plot look good
        if label is not None:
            axis.legend(prop={'size': 15})
        axis.set_title(f"Room temperature {env.umar_model.data.columns[i][-3:]}", size=22)
        axis.set_ylabel("Degrees ($^\circ$C)", size=20)

        # Get the subplot position of the valves action (just below the temperature)
        axis = axes[num - (num % 2) + 1, num % 2]

        # Plot the valves action
        axis.plot(index, data.iloc[:, j] * 100, label=label, color=color)

        # Customize
        axis.set_title(f"Valves {env.umar_model.data.columns[i][-3:]}", size=22)
        if label is not None:
            axis.legend(prop={'size': 15})
        axis.set_ylim((-1, 101))
        axis.set_ylabel("Opening (%)", size=20)

    # Needed to plot everything at the right place
    if not env.battery:
        x += 1

    # Plot the electricity price
    if elec_price:
        if len(env.rooms) % 2 == 1:
            axis = axes[x - 7, 1]
        else:
            axis = axes[x - 6, 0]
        axis.plot(index, data.loc[:, "Electricity price"], color="black")
        axis.set_title("Electricity price", size=22)
        axis.set_ylabel("Price (CHF/kWh)", size=20)

    # Just below or next to it plot the electricity imports
    axes[x - 6, 1].plot(index, electricity_imports, label=label, color=color)
    axes[x - 6, 1].plot(index, np.zeros(len(index)), linestyle="dashed", color="black")
    if label is not None:
        axes[x - 6, 1].legend(prop={'size': 15})
    axes[x - 6, 1].set_title("Electricity imports/exports from the grid", size=22)
    axes[x - 6, 1].set_ylabel("kWh/15 min", size=20)

    axes[x - 5, 0].plot(index[:-1], np.cumsum(comfort_violations), label=label, color=color)
    if label is not None:
        axes[x - 5, 0].legend(prop={'size': 15})
    axes[x - 5, 0].set_title("Comfort violations", size=22)
    axes[x - 5, 0].set_ylabel("Degrees * 15 min", size=20)

    axes[x - 5, 1].plot(index[:-1], np.cumsum(prices), label=label, color=color)
    if label is not None:
        axes[x - 5, 1].legend(prop={'size': 15})
    axes[x - 5, 1].set_title("Total price", size=22)
    axes[x - 5, 1].set_ylabel("CHF", size=20)

    axes[x - 4, 1].plot(index[:-1], rewards, label=label, color=color)
    if label is not None:
        axes[x - 4, 1].legend(prop={'size': 15})
    axes[x - 4, 1].set_title("Rewards", size=22)

    if env.simple:
        axes[x - 4, 0].plot(index[:-1], np.cumsum(rewards), label=label, color=color)
        if label is not None:
            axes[x - 4, 0].legend(prop={'size': 15})
        axes[x - 4, 0].set_title("Cumulative rewards", size=22)

        x -= 4

    else:
        axes[x - 3, 1].plot(index[:-1], np.cumsum(rewards), label=label, color=color)
        if label is not None:
            axes[x - 3, 1].legend(prop={'size': 15})
        axes[x - 3, 1].set_title("Cumulative rewards", size=22)

        # To add solar irradiations and inlet temp
        axes[x - 4, 0].plot(index, data.loc[index, "Weather solar irradiation"], label="Solar irradiations", color="black")
        axes[x - 4, 0].legend(prop={'size': 15})
        axes[x - 4, 0].set_ylabel("W/m^2 (?)", size=20)
        axes[x - 4, 0].set_title("Outside conditions", size=22)

        ax = axes[x - 4, 0].twinx()
        ax.plot(index, data.loc[index, "Weather outside temperature"], label="Temperature", color="black", linestyle="dashed")
        ax.legend(prop={'size': 15})
        ax.tick_params(axis='y', which='major', labelsize=15)
        ax.set_ylabel("Degrees", size=20)

        axes[x - 3, 0].plot(index, data.loc[index, "Thermal inlet temperature"], color="black")
        axes[x - 3, 0].set_title("Water inlet temperature", size=22)
        axes[x - 3, 0].set_ylabel("Degrees", size=20)

        inside_outside = data.loc[index, "Weather outside temperature"] - data.loc[index, f"Thermal temperature measurement {env.rooms[0]}"]
        inlet_inside = data.loc[index, "Thermal inlet temperature"] - data.loc[index, f"Thermal temperature measurement {env.rooms[0]}"]
        energy_outside = np.cumsum(inside_outside)
        energy_inlet = np.cumsum(data.loc[index, f"Thermal valve {env.rooms[0]}"] * inlet_inside)

        axes[x - 2, 0].plot(index, inside_outside, label=label, color=color)
        if label is not None:
            axes[x - 2, 0].legend(prop={'size': 15})
        axes[x - 2, 0].set_title(f"Difference Inside-Outside {env.rooms[0]}", size=22)
        axes[x - 2, 0].set_ylabel("Delta T", size=20)

        axes[x - 2, 1].plot(index, inlet_inside, label=label, color=color)
        if label is not None:
            axes[x - 2, 1].legend(prop={'size': 15})
        axes[x - 2, 1].set_title(f"Difference Inlet-Inside {env.rooms[0]}", size=22)
        axes[x - 2, 1].set_ylabel("Delta T", size=20)

        axes[x - 1, 0].plot(index, energy_outside, label=label, color=color)
        if label is not None:
            axes[x - 1, 0].legend(prop={'size': 15})
        axes[x - 1, 0].set_title(f"Cumulative losses outside {env.rooms[0]}", size=22)
        axes[x - 1, 0].set_ylabel("Delta T", size=20)

        axes[x - 1, 1].plot(index, energy_inlet, label=label, color=color)
        if label is not None:
            axes[x - 1, 1].legend(prop={'size': 15})
        axes[x - 1, 1].set_title(f"Cumulative energy input from ceiling {env.rooms[0]}", size=22)
        axes[x - 1, 1].set_ylabel("Delta T", size=20)


        # Needed to plot everything at the right place
        if not env.battery:
            x -= 1

        # Plot the battery information
        if env.battery:
            # Plot the battery SoC
            axes[x, 0].plot(index, battery_soc, label=label, color=color)
            axes[x, 0].plot(index, env.battery_margins[0] * np.ones(len(index)), linestyle="dashed",
                                color="black")
            axes[x, 0].plot(index, env.battery_margins[1] * np.ones(len(index)), linestyle="dashed",
                                color="black")
            if label is not None:
                axes[x, 0].legend(prop={'size': 15})
            axes[x, 0].set_title("Battery SoC", size=22)
            axes[x, 0].set_ylabel("SoC (%)", size=20)

            # Plot the battery power inputs
            axes[x, 1].plot(index, battery_powers, label=label, color=color)
            axes[x, 1].plot(index, np.zeros(len(index)), linestyle="dashed", color="black")
            if label is not None:
                axes[x, 1].legend(prop={'size': 15})
            axes[x, 1].set_title("Battery Power Input", size=22)
            axes[x, 1].set_ylabel("Power (kW)", size=20)

    # Resize plot ticks and define xlabels
    for i in range(2):
        for j in range(x+1):
            axes[j, i].tick_params(axis='y', which='major', labelsize=15)
            axes[j, i].tick_params(axis='x', which='major', labelsize=15)
        axes[x-1, i].set_xlabel("Time", size=20)

    # Remove the added actions for further analysis down the line
    electricity_imports.pop(-1)
    battery_powers.pop(-1)

    # Show the plot or return the axes, and return the created data for further analysis
    if show_:
        plt.tight_layout()
        plt.show()
        return data
    else:
        return axes, data

    
    
def autolabel(rects, axis, perc=False, small=False):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            axis.annotate(f'{height*100:.1f}%' if perc else f'{height:.1f}' if small else f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3) if height > 0 else (0, -3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top', size=20)
            

def bar_plot_comparison(rewards, comfort_violations, prices):
    
    fig, ax = plt.subplots(3, 1, figsize=(20, 30))

    labels = ["Reward", "Comfort", "Energy"]
    
    data = {}
    for agent in rewards.keys():
        data[agent] = np.array([np.mean([np.sum(x) for x in rewards[agent]]),
                               np.mean([np.sum(x) for x in comfort_violations[agent]]),
                               np.mean([np.sum(x) for x in prices[agent]])])

    x = np.arange(len(labels))  # the label locations
    width = 0.8/len(rewards.keys())  # the width of the bars
    
    ordered_labels = [x for x in ["Unavoidable", "Bang-bang", "Rule-based", "Last agent", "Best price", "Best comfort", "Best agent"] if x in rewards.keys()]
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, 10))
    
    bars = []
    for i, agent in enumerate(ordered_labels):
        if len(ordered_labels) % 2 == 0:
            pos = x + (-len(ordered_labels)/2 + 0.5 + i) * width
        else:
            pos = x + (-(len(ordered_labels)//2) + i) * width
        bars.append(ax[0].bar(pos, data[agent], width, label=agent, color=colors[i]))
            
    [autolabel(x, ax[0], small=True) for x in bars]
    ax[0].set_xticklabels(labels)
    ax[0].set_xticks(x)

    for agent in rewards.keys():
        if agent != "Unavoidable":
            data[agent] -= data["Unavoidable"]
    del data["Unavoidable"]

    width = 0.8/len(data.keys())

    del ordered_labels[0]
    colors = colors[1:,:]
    del labels[0]
    for key in data.keys():
        data[key] = data[key][1:]
    x = np.arange(len(labels))
    width = 0.8/len(rewards.keys())  # the width of the bars
    
    bars = []
    for i, agent in enumerate(ordered_labels):
        if len(ordered_labels) % 2 == 0:
            pos = x + (-len(ordered_labels)/2 + 0.5 + i) * width
        else:
            pos = x + (-(len(ordered_labels)//2) + i) * width
        bars.append(ax[1].bar(pos, data[agent], width, label=agent, color=colors[i]))
            
    [autolabel(x, ax[1]) for x in bars]
    ax[1].set_xticklabels(labels)
    ax[1].set_xticks(x)
    
    data2 = {}
    for agent in data.keys():
        data2[agent] = np.array([np.std([np.sum(x) for x in rewards[agent]]),
                               np.std([np.sum(x) for x in comfort_violations[agent]]),
                               np.std([np.sum(x) for x in prices[agent]])])

    div = data["Bang-bang"].copy()
    for agent in data.keys():
        data[agent] /= div
        
    bars = []
    for i, agent in enumerate(ordered_labels):
        if len(ordered_labels) % 2 == 0:
            pos = x + (-len(ordered_labels)/2 + 0.5 + i) * width
        else:
            pos = x + (-(len(ordered_labels)//2) + i) * width
        bars.append(ax[2].bar(pos, data[agent], width, label=agent, color=colors[i]))

    [autolabel(x, ax[2], perc=True) for x in bars]
    ax[2].set_xticklabels(labels)
    ax[2].set_xticks(x)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    for axis in ax:
        axis.set_ylabel('Scores', size=25)
        axis.set_title('Performance', size=35)
        axis.tick_params(axis="x", which="major", labelsize=25)
        axis.tick_params(axis="y", which="major", labelsize=25)
        axis.legend()
        axis.legend(prop={'size': 20})

    fig.tight_layout()
    plt.savefig(os.path.join("..", "saves", "Figures", "Performance_nice.pdf"), format="pdf")
    plt.show()
    
def save_analysis(rewards, comfort_violations, prices, agent_kwargs):
    if not os.path.isdir(os.path.join("..", "saves", "Experiments", agent_kwargs["name"])):
        os.mkdir(os.path.join("..", "saves", "Experiments", agent_kwargs["name"]))
        
    data = pd.DataFrame({key: [np.mean(x) for x in rewards[key]] for key in rewards.keys()})
    data.to_csv(os.path.join("..", "saves", "Experiments", agent_kwargs["name"], "rewards.csv"))

    data = pd.DataFrame({key: [np.mean(x) for x in comfort_violations[key]] for key in comfort_violations.keys()})
    data.to_csv(os.path.join("..", "saves", "Experiments", agent_kwargs["name"], "comfort.csv"))

    data = pd.DataFrame({key: [np.mean(x) for x in prices[key]] for key in prices.keys()})
    data.to_csv(os.path.join("..", "saves", "Experiments", agent_kwargs["name"], "prices.csv"))


######################################
##  From stable_baselines.common.env_checker

## This small modifications are needed because our environments are always of type gym.GoalEnv
## even if HER is not used. This breaks their original 'check_env' function, which assumes
## a normal gym.Env if HER is not used

## Modification in '_check_returned_values'

######################################
def _enforce_array_obs(observation_space: spaces.Space) -> bool:
    """
    Whether to check that the returned observation is a numpy array
    it is not mandatory for `Dict` and `Tuple` spaces.
    """
    return not isinstance(observation_space, (spaces.Dict, spaces.Tuple))


def _check_image_input(observation_space: spaces.Box) -> None:
    """
    Check that the input will be compatible with Stable-Baselines
    when the observation is apparently an image.
    """
    if observation_space.dtype != np.uint8:
        warnings.warn("It seems that your observation is an image but the `dtype` "
                      "of your observation_space is not `np.uint8`. "
                      "If your observation is not an image, we recommend you to flatten the observation "
                      "to have only a 1D vector")

    if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
        warnings.warn("It seems that your observation space is an image but the "
                      "upper and lower bounds are not in [0, 255]. "
                      "Because the CNN policy normalize automatically the observation "
                      "you may encounter issue if the values are not in that range."
                      )

    if observation_space.shape[0] < 36 or observation_space.shape[1] < 36:
        warnings.warn("The minimal resolution for an image is 36x36 for the default CnnPolicy. "
                      "You might need to use a custom `cnn_extractor` "
                      "cf https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html")


def _check_unsupported_obs_spaces(env: gym.Env, observation_space: spaces.Space) -> None:
    """Emit warnings when the observation space used is not supported by Stable-Baselines."""

    if isinstance(observation_space, spaces.Dict) and not isinstance(env, gym.GoalEnv):
        warnings.warn("The observation space is a Dict but the environment is not a gym.GoalEnv "
                      "(cf https://github.com/openai/gym/blob/master/gym/core.py), "
                      "this is currently not supported by Stable Baselines "
                      "(cf https://github.com/hill-a/stable-baselines/issues/133), "
                      "you will need to use a custom policy. "
                      )

    if isinstance(observation_space, spaces.Tuple):
        warnings.warn("The observation space is a Tuple,"
                      "this is currently not supported by Stable Baselines "
                      "(cf https://github.com/hill-a/stable-baselines/issues/133), "
                      "you will need to flatten the observation and maybe use a custom policy. "
                      )


def _check_nan(env: gym.Env) -> None:
    """Check for Inf and NaN using the VecWrapper."""
    vec_env = VecCheckNan(DummyVecEnv([lambda: env]))
    for _ in range(10):
        action = [env.action_space.sample()]
        _, _, _, _ = vec_env.step(action)


def _check_obs(obs: Union[tuple, dict, np.ndarray, int],
               observation_space: spaces.Space,
               method_name: str) -> None:
    """
    Check that the observation returned by the environment
    correspond to the declared one.
    """
    if not isinstance(observation_space, spaces.Tuple):
        assert not isinstance(obs, tuple), ("The observation returned by the `{}()` "
                                            "method should be a single value, not a tuple".format(method_name))

    # The check for a GoalEnv is done by the base class
    if isinstance(observation_space, spaces.Discrete):
        assert isinstance(obs, int), "The observation returned by `{}()` method must be an int".format(method_name)
    elif _enforce_array_obs(observation_space):
        assert isinstance(obs, np.ndarray), ("The observation returned by `{}()` "
                                             "method must be a numpy array".format(method_name))

    assert observation_space.contains(obs), ("The observation returned by the `{}()` "
                                             "method does not match the given observation space".format(method_name))


def _check_returned_values(env: gym.Env, observation_space: spaces.Space, action_space: spaces.Space) -> None:
    """
    Check the returned values by the env when calling `.reset()` or `.step()` methods.
    """
    # because env inherits from gym.Env, we assume that `reset()` and `step()` methods exists
    obs = env.reset()

    _check_obs(obs, observation_space, 'reset')

    # Sample a random action
    action = action_space.sample()
    data = env.step(action)

    assert len(data) == 4, "The `step()` method must return four values: obs, reward, done, info"

    # Unpack
    obs, reward, done, info = data

    _check_obs(obs, observation_space, 'step')

    # We also allow int because the reward will be cast to float
    assert isinstance(reward, (float, int)), "The reward returned by `step()` must be a float"
    assert isinstance(done, bool), "The `done` signal must be a boolean"
    assert isinstance(info, dict), "The `info` returned by `step()` must be a python dictionary"

    #############################################
    ## Modification
    #############################################

    if env.her:
        # For a GoalEnv, the keys are checked at reset
        assert reward == env.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)


def _check_spaces(env: gym.Env) -> None:
    """
    Check that the observation and action spaces are defined
    and inherit from gym.spaces.Space.
    """
    # Helper to link to the code, because gym has no proper documentation
    gym_spaces = " cf https://github.com/openai/gym/blob/master/gym/spaces/"

    assert hasattr(env, 'observation_space'), "You must specify an observation space (cf gym.spaces)" + gym_spaces
    assert hasattr(env, 'action_space'), "You must specify an action space (cf gym.spaces)" + gym_spaces

    assert isinstance(env.observation_space,
                      spaces.Space), "The observation space must inherit from gym.spaces" + gym_spaces
    assert isinstance(env.action_space, spaces.Space), "The action space must inherit from gym.spaces" + gym_spaces


def _check_render(env: gym.Env, warn: bool = True, headless: bool = False) -> None:
    """
    Check the declared render modes and the `render()`/`close()`
    method of the environment.
    :param env: (gym.Env) The environment to check
    :param warn: (bool) Whether to output additional warnings
    :param headless: (bool) Whether to disable render modes
        that require a graphical interface. False by default.
    """
    render_modes = env.metadata.get('render.modes')
    if render_modes is None:
        if warn:
            warnings.warn("No render modes was declared in the environment "
                          " (env.metadata['render.modes'] is None or not defined), "
                          "you may have trouble when calling `.render()`")

    else:
        # Don't check render mode that require a
        # graphical interface (useful for CI)
        if headless and 'human' in render_modes:
            render_modes.remove('human')
        # Check all declared render modes
        for render_mode in render_modes:
            env.render(mode=render_mode)
        env.close()


def check_env(env: gym.Env, warn: bool = True, skip_render_check: bool = True) -> None:
    """
    Check that an environment follows Gym API.
    This is particularly useful when using a custom environment.
    Please take a look at https://github.com/openai/gym/blob/master/gym/core.py
    for more information about the API.
    It also optionally check that the environment is compatible with Stable-Baselines.
    :param env: (gym.Env) The Gym environment that will be checked
    :param warn: (bool) Whether to output additional warnings
        mainly related to the interaction with Stable Baselines
    :param skip_render_check: (bool) Whether to skip the checks for the render method.
        True by default (useful for the CI)
    """
    assert isinstance(env, gym.Env), ("Your environment must inherit from the gym.Env class "
                                      "cf https://github.com/openai/gym/blob/master/gym/core.py")

    # ============= Check the spaces (observation and action) ================
    _check_spaces(env)

    # Define aliases for convenience
    observation_space = env.observation_space
    action_space = env.action_space

    # Warn the user if needed.
    # A warning means that the environment may run but not work properly with Stable Baselines algorithms
    if warn:
        _check_unsupported_obs_spaces(env, observation_space)

        # If image, check the low and high values, the type and the number of channels
        # and the shape (minimal value)
        if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
            _check_image_input(observation_space)

        if isinstance(observation_space, spaces.Box) and len(observation_space.shape) not in [1, 3]:
            warnings.warn("Your observation has an unconventional shape (neither an image, nor a 1D vector). "
                          "We recommend you to flatten the observation "
                          "to have only a 1D vector")

        # Check for the action space, it may lead to hard-to-debug issues
        if (isinstance(action_space, spaces.Box) and
                (np.any(np.abs(action_space.low) != np.abs(action_space.high))
                 or np.any(np.abs(action_space.low) > 1) or np.any(np.abs(action_space.high) > 1))):
            warnings.warn("We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) "
                          "cf https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html")

    # ============ Check the returned values ===============
    _check_returned_values(env, observation_space, action_space)

    # ==== Check the render method and the declared render modes ====
    if not skip_render_check:
        _check_render(env, warn=warn)

    # The check only works with numpy arrays
    if _enforce_array_obs(observation_space):
        _check_nan(env)



import copy
from enum import Enum

import numpy as np


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """
    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # Select a goal that was achieved
    # at some point in the training procedure
    # (and that is present in the replay buffer)
    RANDOM = 3


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM
}


class HindsightExperienceReplayWrapper(object):
    """
    Wrapper around a replay buffer in order to use HER.
    This implementation is inspired by to the one found in https://github.com/NervanaSystems/coach/.
    :param replay_buffer: (ReplayBuffer)
    :param n_sampled_goal: (int) The number of artificial transitions to generate for each actual transition
    :param goal_selection_strategy: (GoalSelectionStrategy) The method that will be used to generate
        the goals for the artificial transitions.
    :param wrapped_env: (HERGoalEnvWrapper) the GoalEnv wrapped using HERGoalEnvWrapper,
        that enables to convert observation to dict, and vice versa
    """

    def __init__(self, replay_buffer, n_sampled_goal, goal_selection_strategy, wrapped_env):
        super(HindsightExperienceReplayWrapper, self).__init__()

        assert isinstance(goal_selection_strategy, GoalSelectionStrategy), "Invalid goal selection strategy," \
                                                                           "please use one of {}".format(
            list(GoalSelectionStrategy))

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.env = wrapped_env
        # Buffer for storing transitions of the current episode
        self.episode_transitions = []
        self.replay_buffer = replay_buffer

    def add(self, obs_t, action, reward, obs_tp1, done, info):
        """
        add a new transition to the buffer
        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        :param info: (dict) extra values used to compute reward
        """
        assert self.replay_buffer is not None
        # Update current episode buffer
        self.episode_transitions.append((obs_t, action, reward, obs_tp1, done, info))
        if done:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []

    def sample(self, *args, **kwargs):
        return self.replay_buffer.sample(*args, **kwargs)

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return self.replay_buffer.can_sample(n_samples)

    def __len__(self):
        return len(self.replay_buffer)

    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling strategy.
        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random goal achieved, from the entire replay buffer
            selected_idx = np.random.choice(np.arange(len(self.replay_buffer)))
            selected_transition = self.replay_buffer.storage[selected_idx]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal']

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.
        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        for transition_idx, transition in enumerate(self.episode_transitions):

            obs_t, action, reward, obs_tp1, done, info = transition

            # Add to the replay buffer
            self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(self.episode_transitions) - 1 and
                    self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                break

            # Sampled n goals per transition, where n is `n_sampled_goal`
            # this is called k in the paper
            sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
            # For each sampled goals, store a new transition
            for goal in sampled_goals:
                # Copy transition to avoid modifying the original one
                obs, action, reward, next_obs, done, info = copy.deepcopy(transition)

                # Convert concatenated obs to dict, so we can update the goals
                obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))

                # Update the desired goal in the transition
                obs_dict['desired_goal'] = goal
                next_obs_dict['desired_goal'] = goal

                # Update the reward according to the new desired goal
                reward = self.env.compute_reward(next_obs_dict['achieved_goal'], goal, info)
                # Can we use achieved_goal == desired_goal?
                done = False

                # Transform back to ndarrays
                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                # Add artificial transition to the replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)





def add_electricity_price_old(model, price_type, const, hours=None):
    """
    Function to add a column to the data representing the electricity price

    args:
        model:          The model we work with (to access the data)
        price_type:     The pricing scheme to use
        const:          The price constant
        const:          A price constant used in some schemes

    Returns:
        A list of prices corresponding to each datatime Index
    """

    # First check that the constant is valid, since we work with normalized data
    assert (const >= 0.1) & (const <= 0.9), f"The price constant needs to be set between 0.1 and" \
                                            f"0.9 since we work with normalized data"

    # Get the hour corresponding to each time step
    if hours is None:
        hours = model.data.index.hour.astype(np.float64)

    # Constant (thus useless) price
    if price_type == "constant":
        prices = np.array([const] * len(hours))

    # Three level pricing: low at night, peaks in the morning and evening and medium (const) price in the middle
    elif price_type == "ThreeLevels":
        prices = np.zeros_like(hours)

        if len(prices) > 1:
            # Low price during the night
            prices[np.where((hours < 6) | (hours > 21))[0]] = 0.3

            # Const during the day
            prices[np.where((hours > 9) & (hours < 17))[0]] = const

            # Peak pricing in the morning and evening
            prices[np.where(((hours >= 6) & (hours <= 9)) | ((hours >= 17) & (hours <= 21)))[0]] = 0.9

        # Needed to make it work when only one timestamp is provided (i.e. when the agent is running live)
        else:
            # Low price during the night
            if (hours < 6) | (hours > 21):
                prices = 0.3

            # Const during the day
            elif (hours > 9) & (hours < 17):
                prices = const

            # Peak pricing in the morning and evening
            else:
                prices = 0.9

    else:
        raise NotImplementedError(f"{price_type} does not exist!")

    # Return it
    return prices
def add_bounds_and_next_changes(model, temp_bounds: list, change_times: list, cases=None, hours=None):
    """
    Function to add a column to the data representing the temperature bounds

    args:
        model:          The model to add bounds to
        temp_bounds:    Lower and upper bounds limits
        change_times:   Limits to keep the same bounds
        cases:          To provide custom cases (heating or cooling)
        hours:          To provide custom hours

    Returns:
        A list of lower and upper bounds at each time step (normalized)
    """

    # Get the hour and cases corresponding to each time step when needed
    if hours is None:
        hours = model.data.index.hour.astype(np.float64)
    if cases is None:
        cases = model.data.Case.astype(np.float64)

    # Normalize the bounds
    bounds = np.array(temp_bounds)
    bounds = (bounds - model.normalization_variables['Room'][0].values) / model.normalization_variables['Room'][1].values * 0.8 + 0.1

    half_dregree_scaled = 0.5 / model.normalization_variables['Room'][1].values * 0.8
    low_possibilities = int((temp_bounds[1] - temp_bounds[0]) / 0.5 + 1)
    high_possibilities = int((temp_bounds[3] - temp_bounds[2]) / 0.5 + 1)

    changes = []
    low_bounds = []
    high_bounds = []
    time_till_change = []
    tot = 0

    if len(hours) > 1:
        mul = int(60 / model.dataset.interval)
        while tot < len(hours) + change_times[1] * mul + 1:
            changes.append(np.random.randint(change_times[0] * mul, change_times[1] * mul + 1))
            low = bounds[0] + np.random.randint(low_possibilities) * half_dregree_scaled
            high = bounds[2] + np.random.randint(high_possibilities) * half_dregree_scaled
            for i in range(changes[-1]):
                low_bounds.append(low)
                high_bounds.append(high)
                time_till_change.append(changes[-1] - i)
            tot += changes[-1]

        low_bounds = np.array(low_bounds)
        high_bounds = np.array(high_bounds)

        low_bounds[np.where(cases < 0.5)[0]] = bounds[1]
        high_bounds[np.where(cases > 0.5)[0]] = bounds[3]

        next_low_bounds = []
        next_high_bounds = []
        for i in range(len(high_bounds) - changes[-1]):
            next_low_bounds.append(low_bounds[i + time_till_change[i]])
            next_high_bounds.append(high_bounds[i + time_till_change[i]])

        next_low_bounds = np.array(next_low_bounds)
        next_high_bounds = np.array(next_high_bounds)
        time_till_change = np.array(time_till_change) / (change_times[1] * 60 / model.dataset.interval) * 0.8 + 0.1

    # Needed to make it work when only one timestamp is provided (i.e. when the agent is running live)
    else:
        raise NotImplementedError

    # Return it
    return low_bounds[:len(hours)], high_bounds[:len(hours)], time_till_change[:len(hours)],\
           next_low_bounds[:len(hours)], next_high_bounds[:len(hours)]

###### wrote by mina to use pcnn model instead of arx
def prepare_model_mina(agent_kwargs: dict, Y_columns: list, X_columns: list, base_indices: list, effect_indices: list,
                  room: int ):
    """
    Small helper function to load the model and add the needed information (e.g. bounds)

    Args:
        agent_kwargs:   Parameters of the agent, see 'parameters.py'
        Y_columns:      Name of the columns that are to be predicted
        X_columns:      Sensors (columns) of the input data
        base_indices:   Input features used to predict the base component of the network
        effect_indices: Input features used to predict the heating and cooling components of the network
        room:           Room to model

    Returns:
        The prepared model
    """

    # DATA_SAVE_PATH1 = os.path.join("..", "saves", "Data_preprocessed")
    DATA_SAVE_PATH1 = r'C:\Users\remok\OneDrive - Danmarks Tekniske Universitet\Skrivebord\RL TRV\Models\Price-Responsive RL\saves\Data_preprocessed'
    # DATA_SAVE_PATH1 = os.path.join("..", "DRL", "saves", "Data_preprocessed")
    #print("mina")
    #print(DATA_SAVE_PATH1)
    # data = load_data('umarnew',DATA_SAVE_PATH1)
    # data = load_data('umarnew_hourly4',DATA_SAVE_PATH1)
    # data = load_data('Book5',DATA_SAVE_PATH1)
    # data = load_data('Book51',DATA_SAVE_PATH1)
    # data = load_data('Book51_FuturePrice2',DATA_SAVE_PATH1)
    
    
    
    # data = load_data('Book52_FuturePrice11',DATA_SAVE_PATH1)
    # data = load_data('Book52_FuturePrice1',DATA_SAVE_PATH1)
    data = load_data('Book53_FuturePrice1',DATA_SAVE_PATH1)

    
    # data = load_data('Book53_FuturePrice1',DATA_SAVE_PATH1)
    # data = load_data('Book53_FuturePrice_RL2',DATA_SAVE_PATH1)
    # data = load_data('Book53_FuturePrice_RL2',DATA_SAVE_PATH1)

    # data = load_data('Data_RL0',DATA_SAVE_PATH1)
    # data = load_data('Data_RL1_Train',DATA_SAVE_PATH1)



    # data = load_data('Book52_FuturePrice11',DATA_SAVE_PATH1)
    # data = load_data('Book52',DATA_SAVE_PATH1)


    # data = load_data('umar_newp',DATA_SAVE_PATH1)
    # data = load_data('Book2_Switched',DATA_SAVE_PATH1)
    # data = load_data('umarnew_hourly_second2',DATA_SAVE_PATH1)
    # data = load_data('umarnew_hourly_second',DATA_SAVE_PATH1)


    interval = 15
    ###############################################################
    # model_kwargs = parameters(unit='UMAR',
    #                           to_normalize=True,
    #                           name="PCNN",
    #                           seed=0,
    #                           overlapping_distance=4,
    #                           warm_start_length=12,
    #                           # maximum_sequence_length=96 * 3,
    #                           minimum_sequence_length=48,
    #                           learn_initial_hidden_states=True,
    #                           decrease_learning_rate=True,
    #                           learning_rate=0.0005,
    #                           feed_input_through_nn=True,
    #                           input_nn_hidden_sizes=[64],
    #                           lstm_hidden_size=128,
    #                           lstm_num_layers=2,
    #                           layer_norm=True,
    #                           batch_size=256,
    #                           output_nn_hidden_sizes=[64],
    #                           division_factor=10.,
    #                           verbose=2)
    model_kwargs = parameters(unit='UMAR',
                              to_normalize=True,
                              name="PCNN",
                              seed=0,
                              overlapping_distance=4,
                              warm_start_length=12,
                              maximum_sequence_length=96 * 3,
                              minimum_sequence_length=48,
                              learn_initial_hidden_states=True,
                              decrease_learning_rate=False,
                              learning_rate=0.0005,
                              feed_input_through_nn=True,
                              input_nn_hidden_sizes=[32],
                              lstm_hidden_size=64,
                              lstm_num_layers=2,
                              layer_norm=True,
                              batch_size=256,
                              output_nn_hidden_sizes=[32],
                              division_factor=10.,
                              verbose=2)
    

    module = 'PCNN'
    # rooms = '272'

    # No need of info on room 274 as input
    # X_columns = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
    #              'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Power 272', 'Case', 'Valve 272']

    if room == 272:
        X_columns = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
                     'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 272', 'Case', 'Price', 'PriceChange']
        Y_columns = ['Temperature 272']
    elif room == 274:
        X_columns = ['Solar irradiation', 'Outside temperature','Temperature 274', 'Temperature 273', 'Month sin',
                     'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 274', 'Case', 'Price', 'PriceChange']
        Y_columns = ['Temperature 274']


    # Corresponding columns
    case_column = 10
    out_column = 1
    neigh_column = 3
    temperature_column = 2
    power_column = 9
    #valve_column = 11

    # Info to use in D
    inputs_D = [0, 4, 5, 6, 7, 8]

    topology = None  # Not needed in the single-zone case

    # Trying to load a model or not, if yes the last one or the best one
    load = True
    load_last = False

    # model = Model(data=data, interval=interval, model_kwargs=model_kwargs, inputs_D=inputs_D,
    #              module=module, rooms=room, case_column=case_column, out_column=out_column, neigh_column=neigh_column,
    #              temperature_column=temperature_column,power_column=power_column ,valve_column=valve_column,
    #              Y_columns=Y_columns, X_columns=X_columns, topology=topology, load_last=load_last,
    #              load=load)
    model = Model(data=data, interval=interval, model_kwargs=model_kwargs, inputs_D=inputs_D,
                 module=module, rooms=room, case_column=case_column, out_column=out_column, neigh_column=neigh_column,
                 temperature_column=temperature_column,power_column=power_column ,
                 Y_columns=Y_columns, X_columns=X_columns, topology=topology, load_last=load_last,
                 load=load)



    model.data = model.dataset.data.copy()

    print(model.data)
    # Add bounds
    if not agent_kwargs['random_bounds']:
        model.data['Lower bound'], model.data['Upper bound'] = add_bounds(model=model,
                                                                          temp_bounds=agent_kwargs['temperature_bounds'],
                                                                          change_times=agent_kwargs['change_times'])

    else:
        model.data['Lower bound'], model.data['Upper bound'], model.data['Next bound change'],\
        model.data['Next lower bound'], model.data['Next upper bound'] = add_bounds_and_next_changes(model=model,
                                                                                                 temp_bounds=agent_kwargs['temperature_bounds'],
                                                                                                 change_times=agent_kwargs['change_times'])

    model.max_ = model.dataset.max_.copy()
    model.min_ = model.dataset.min_.copy()
    # Put everything in the needed form
    model.all_columns = model.data.columns
    model.data = model.data

    return model, model_kwargs
