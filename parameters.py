"""
Simulation parameters
"""
import os
import sys
import numpy as np
import tensorflow as tf

path = "saves" if sys.platform == "win32" else os.path.join("..", "saves")

def prepare_kwargs(model_name, agent_name, start_date="2021-12-25", end_date="2022-01-24", name="Full_Data",
                   interval=15, predict_differences=False, missing_values_threshold=2, to_normalize=True,
                   to_standardize=False, data_verbose=1, rooms=['272', '273', '274', '275', '276'],
                   unit="UMAR", save_path=os.path.join(path, "Models"), load_model=True,
                   show_plots=False, seed=0, save_checkpoints=True,
                   feature_extraction_sizes=[16, 4, 1],
                   n_autoregression=20, hidden_sizes=[16, 8, 4],  output_size=1,
                   NN=True, hidden_size=16, num_layers=1, threshold_length=96, overshoot=4,
                   learning_rate=0.0005, batch_size=32, n_epochs=20, validation_percentage=0.15, test_percentage=0.,
                   model_verbose=1, loss_hyperparameter=0.25, heating=True, cooling=True,
                   algorithm="PPO2", temp_bounds=[22, 23, 25, 26], agent_save_path=os.path.join(path, "Agents"),
                   discrete=False, ddpg=False, discrete_actions=[17, 18, 19, 20, 21, 22], battery_max_power=5.,
                   battery_size=5., battery_barrier_penalty=5., normalizing=False, backup=False, battery=False,
                   gamma=0.95, temperature_penalty_factor=5., her=False, goal_selection_strategy="future",
                   n_sampled_goal=4, autoregressive_terms=[20, 12, 4, 2, 1],
                   n_envs=8, lstm_size=64, extraction_size=16, vf_layers=[16, 8, 4], pi_layers=[16, 8, 4],
                   save_freq=25000, eval_freq=25000, price_levels=[ 1], price_type="OneLevel",
                   COP=[3., 4.], small_obs=False, agent_lr=1e-4, vf_loss_coef=0.01, ent_coef=0.01,
                   n_eval_episodes=100, small_model=False, simple_env=False, random_bounds: bool = False,temperature_bounds: list = [20, 22, 23, 25],change_times: list = [8, 20],autoregression: int = 6,):
# temp_bounds=[20, 22, 23, 25]

    ## Define/load data from the right unit

    from NEST_data.Weather import weather_names
#price_levels=[0.12, 0.15, 0.2], price_type="ThreeLevels",
    if unit == "DFAB":
        ROOMS = ['371', '472', '474', '476', '571', '573', '574']
        from NEST_data.DFAB import thermal_names, valves_controllable

    elif unit == "UMAR":
        ROOMS = ['272', '273', '274', '275', '276']
        from NEST_data.UMAR import thermal_names, valves_controllable, rooms_names

    else:
        raise ValueError(f"Unit {unit} is unknown")

    # Build the autoregressive names i.e. the sensor names for which autoregressive terms should be considered

    def list_items(dictionary):
        """
        Function to return a list of items in a dictionary
        """
        return [dictionary[key] for key in dictionary.keys()]

    #autoregressive_names = list_items(weather_names) \
                          # + ['HP electricity consumption', 'HP Boiler temperature'] \
                          # + list_items(thermal_names) \
                          # + valves_controllable \
                          # + ["Electricity total consumption"]

    #if unit == "UMAR":
        #autoregressive_names += list_items(rooms_names)

    # Build the components of the model
    components = [f"Thermal temperature measurement {room}" for room in ROOMS]

    if unit == "DFAB":
        components.append("Thermal total energy")
        not_differences_components = ["Thermal total energy"]
        components.append("HP Boiler temperature")
        not_differences_components.append("HP Boiler temperature")

    elif unit == "UMAR":
        components.append("Thermal total energy")
        not_differences_components = ["Thermal total energy"]
        #components.append("Thermal cooling energy")
        #not_differences_components.append("Thermal cooling energy")

    # Make sure what we are doing makes sense: if the hidden size (output size of LSTM) is bigger than one
    # we need to pass it through a NN to hve the right output dimesion (of 1)
    if hidden_size > 1:
        NN = True

    discrete_actions = np.array(discrete_actions)

    if (algorithm == "DDPG") or (algorithm == "TD3"):
        ddpg = True

    if algorithm == "DDPG":
        discrete = False

    if algorithm == "A2C":
        ddpg = False

    assert not (discrete and ddpg), "DDPG cannot work on discrete setups"

    battery_margins = [50. - battery_size / 0.96 / 2, 50. + battery_size / 0.96 / 2]

    if "small" in model_name:
        small_model = True

    assert (len(temp_bounds) == 4) | (len(temp_bounds) == 2), \
        f"Invaid tmperature bounds, provide either bounds of length 2 or 4, not {temp_bounds}"

    assert (len(price_levels) == 3) | (len(price_levels) == 1),\
        f"Invalid price levels, 1 or 3 levels are needed, given: {price_levels}"

   # if not tf.test.is_gpu_available():
        #n_envs = 1

    data_kwargs = dict(unit=unit,
                       start_date=start_date,
                       end_date=end_date,
                       name=name,
                       interval=interval,
                       missing_values_threshold=missing_values_threshold,
                       components=components,
                       not_differences_components=not_differences_components,
                       predict_differences=predict_differences,
                       to_normalize=to_normalize,
                       to_standardize=to_standardize,
                       verbose=data_verbose,
                       small_model=small_model)

    # model_kwargs = dict(unit=unit,
    #                     model_name=model_name,
    #                     save_path=save_path,
    #                     load_model=load_model,
    #                     show_plots=show_plots,
    #                     NN=NN,
    #                     predict_differences=predict_differences,
    #                     save_checkpoints=save_checkpoints,
    #                     n_autoregression=n_autoregression,
    #                     hidden_size=hidden_size,
    #                     num_layers=num_layers,
    #                     hidden_sizes=hidden_sizes,
    #                     feature_extraction_sizes=feature_extraction_sizes,
    #                     output_size=output_size,
    #                     learning_rate=learning_rate,
    #                     batch_size=batch_size,
    #                     n_epochs=n_epochs,
    #                     validation_percentage=validation_percentage,
    #                     test_percentage=test_percentage,
    #                     overshoot=overshoot,
    #                     components=components,
    #                     #autoregressive=autoregressive_names,
    #                     not_differences_components=not_differences_components,
    #                     threshold_length=threshold_length,
    #                     seed=seed,
    #                     verbose=model_verbose,
    #                     loss_hyperparameter=loss_hyperparameter,
    #                     heating=heating,
    #                     cooling=cooling,
    #                     interval=interval,
    #                     rooms=rooms)

    agent_kwargs = dict(algorithm=algorithm,
                        num_test_envs=50,
                        temp_bounds=temp_bounds,
                        save_path=agent_save_path,
                        discrete=discrete,
                        ddpg=ddpg,
                        her=her,
                        goal_selection_strategy=goal_selection_strategy,
                        n_sampled_goal=n_sampled_goal,
                        autoregressive_terms=autoregressive_terms,
                        rooms=rooms,
                        discrete_actions=discrete_actions,
                        battery=battery,
                        battery_max_power=battery_max_power,
                        n_envs=n_envs,
                        lstm_size=lstm_size,
                        extraction_size=extraction_size,
                        vf_layers=vf_layers,
                        pi_layers=pi_layers,
                        name=agent_name,
                        save_freq=save_freq,
                        eval_freq=eval_freq,
                        price_levels=price_levels,
                        price_type=price_type,
                        n_eval_episodes=n_eval_episodes,
                        COP=COP,
                        battery_margins=battery_margins,
                        battery_barrier_penalty=battery_barrier_penalty,
                        temperature_penalty_factor=temperature_penalty_factor,
                        gamma=gamma,
                        learning_rate=agent_lr,
                        vf_loss_coef=vf_loss_coef,
                        ent_coef=ent_coef,
                        battery_size=battery_size,
                        normalizing=normalizing,
                        backup=backup,
                        small_obs=small_obs,
                        simple_env=simple_env,
                        small_model=small_model,
                        random_bounds=random_bounds,
                        temperature_bounds=temperature_bounds,
                        change_times=change_times,
                        autoregression=autoregression,
                        )

    return data_kwargs, agent_kwargs, ROOMS
    # return data_kwargs, model_kwargs, agent_kwargs, ROOMS

## Main function: define your parameters!
data_kwargs,  agent_kwargs, ROOMS = prepare_kwargs(start_date='2018-01-01',
                                                                end_date='2020-06-23',
                                                                model_name="LSTMModel",
                                                                agent_name="Default")
# data_kwargs, model_kwargs, agent_kwargs, ROOMS = prepare_kwargs(start_date='2018-01-01',
#                                                                 end_date='2020-06-23',
#                                                                 model_name="LSTMModel",
#                                                                 agent_name="Default")

UNIT = "UMAR"
