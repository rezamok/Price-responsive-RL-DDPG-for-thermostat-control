# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:31:41 2021

@author: kosta
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:34:56 2021

@author: kosta
"""

import os
import sys
import numpy as np

from agents.environments import ToyEnv
from parameters import prepare_kwargs
from kosta.reward import compute_reward
from agents.helpers import prepare_toy_models
from agents.helpers import prepare_model_mina
import kosta.hyperparams as hp
from models.util import build_physics_based_inputs_outputs_indices
from models.parameters import parameters

data_kwargs,  agent_kwargs, _ = prepare_kwargs(
        start_date='2021-1225-',
        end_date='2022-1-24',
        model_name="ARX_0",
        agent_name="201208_272_Toy_scale_10_vf_05_lr_000005_ent_0.05_squared_no_cliping",
        rooms=hp.rooms,
        algorithm="PPO2",
        simple_env=True,
        threshold_length=hp.threshold_length,
        n_autoregression=hp.n_autoregression,
        interval=hp.interval,
        discrete=False,
        backup=False,
        vf_loss_coef=0.5,
        ent_coef=0.05,
        agent_lr=0.00005,
        small_obs=True,
        normalizing=False,
        battery=False,
        temp_bounds=hp.dynamic_temp_bounds if hp.dynamic_bounds else hp.temp_bounds,
        ddpg=True,
        heating=hp.heating,
        cooling=hp.cooling,
        gamma=0.9,
        price_levels=hp.price_levels,
        price_type=hp.price_type,
        lstm_size=128,
        extraction_size=64,
        vf_layers=[64, 64],
        pi_layers=[64, 64],
        eval_freq=100,
        n_envs=8
    )
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
def get_env(env_model,set_room=None):
    if set_room is not None:
        assert(hp.heating or hp.cooling)
        data_kwargs_set, agent_kwargs_set, _ = prepare_kwargs(
            start_date='2018-01-01',
            end_date='2020-10-12',
            model_name="ARX_0",
            agent_name="201208_272_Toy_scale_10_vf_05_lr_000005_ent_0.05_squared_no_cliping",
            rooms=[set_room],
            algorithm="PPO2",
            simple_env=True,
            threshold_length=hp.threshold_length,
            n_autoregression=hp.n_autoregression,
            discrete=False,
            backup=False,
            interval=hp.interval,
            vf_loss_coef=0.5,
            ent_coef=0.05,
            agent_lr=0.00005,
            small_obs=True,
            normalizing=False,
            battery=False,
            temp_bounds=hp.temp_bounds,
            ddpg=True,
            heating=hp.heating,
            cooling=hp.cooling,
            gamma=0.9,
            price_levels=hp.price_levels,
            price_type=hp.price_type,
            lstm_size=128,
            extraction_size=64,
            vf_layers=[64, 64],
            pi_layers=[64, 64],
            eval_freq=100,
            n_envs=8
        )
    all_inputs, all_outputs, base_indices, effect_indices = build_physics_based_inputs_outputs_indices()

    # all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
    #               'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Power 272', 'Valve 272', 'Case']
    all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
            'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 272', 'Case']
    room = 274

    # following is not important data, i mean base and effect indices
    if room == 274:
        base_indices = [1, 4, 7, 8, 9, 10, 11]
        effect_indices = [2, 4, 5, 6]
    elif room == 272:
        base_indices = [1, 3, 7, 8, 9, 10, 11]
        effect_indices = [2, 3, 5, 6]
    _, model_kwargs_set = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                  Y_columns=all_outputs,
                                                  X_columns=all_inputs,
                                                  base_indices=base_indices,
                                                  effect_indices=effect_indices,
                                                  room=room)
    if env_model == 'ARX':
        if set_room is None:
            umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                Y_columns=all_outputs,
                                                X_columns=all_inputs,
                                                base_indices=base_indices,
                                                effect_indices=effect_indices,
                                                room=room)
            #umar_model, _ = prepare_toy_models(data_kwargs, model_kwargs, agent_kwargs)
            env = ToyEnv(umar_model, None, agent_kwargs, compute_reward)
        else:
            umar_model, model_kwargs_set = prepare_model_mina(agent_kwargs=agent_kwargs_set,
                                                          Y_columns=all_outputs,
                                                          X_columns=all_inputs,
                                                          base_indices=base_indices,
                                                          effect_indices=effect_indices,
                                                          room=room)
           # umar_model, _ = prepare_toy_models(data_kwargs_set, model_kwargs_set, agent_kwargs_set)
            env = ToyEnv(umar_model, None, agent_kwargs_set, compute_reward)
    if set_room is not None:
        return env, data_kwargs_set, model_kwargs_set, agent_kwargs_set
    return env

