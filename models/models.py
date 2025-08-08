"""
File containing recurrent black-box models
"""

import os
import pickle
import numpy as np
from typing import Union
import matplotlib.dates as mdates

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import optim
import torch.nn.functional as F

from models.base import TorchModel
from models.modules import PCNN_full_shared_mul, PCNN_full_shared, PCNN_full, PCNN_mul, PCNN, TemperaturePowerTogether, SeparateTemperaturePower,\
    SeparateInputs, DiscountedModules, DiscountedModulesBase, ResDiscountedModulesBase, ResDiscountedModulesBaseMany,\
    ResDiscountedModulesBaseManySep, ResDiscountedModulesBaseTrueEnergy, SimpleLSTMBase, PhysicsInspiredModuleBase, PCNNTestQuantiles

from util.plots import _plot_helpers, _save_or_show
from util.functions import inverse_normalize, inverse_standardize


class Model(TorchModel):
    """
    Main class of models
    """

    def __init__(self, data_kwargs: dict, model_kwargs: dict, Y_columns: list, X_columns: list = None,
                 base_indices: list = None, effect_indices: list = None, topology: dict = None, load_last: bool = False,
                 load: bool = True):
        """
        Initialize a model.

        Args:
            data_kwargs:    Parameters of the data, see 'parameters.py'
            model_kwargs:   Parameters of the models, see 'parameters.py'
            Y_columns:      Name of the columns that are to be predicted
            X_columns:      Sensors (columns) of the input data
            base_indices:   Input features used to predict the base component of the network
            effect_indices: Input features used to predict the heating and cooling components of the network
            load_last:      Flag to set to true when the last checkpoint of the model is needed and not the best one
            load:           Flag to set to False if you do not want to load the model at all
        """

        super().__init__(data_kwargs=data_kwargs, model_kwargs=model_kwargs, Y_columns=Y_columns, X_columns=X_columns)

        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.topology = topology
        self.feed_input_through_nn = model_kwargs["feed_input_through_nn"]
        self.input_nn_hidden_sizes = model_kwargs["input_nn_hidden_sizes"]
        self.lstm_hidden_size = model_kwargs["lstm_hidden_size"]
        self.lstm_num_layers = model_kwargs["lstm_num_layers"]
        self.layer_norm = model_kwargs["layer_norm"]
        self.feed_output_through_nn = model_kwargs["feed_output_through_nn"]
        self.output_nn_hidden_sizes = model_kwargs["output_nn_hidden_sizes"]
        self.module = model_kwargs["module"]
        self.learn_discount_factors = model_kwargs["learn_discount_factors"]
        self.learn_initial_hidden_states = model_kwargs["learn_initial_hidden_states"]
        self.predict_power = model_kwargs['predict_power']

        if (self.module not in ["TemperaturePowerTogether", "SeparateTemperaturePower"]) and self.verbose > 0:
            assert (base_indices is not None) & (effect_indices is not None), "These modules require indices"
            print("\nBase prediction columns:", [np.array(self.dataset.X_columns)[base_indice]
                                                 for base_indice in base_indices])
            print("Effect prediction columns:", [np.array(self.dataset.X_columns)[effect_indice]
                                                 for effect_indice in effect_indices])

        # Define indices of interest
        self.case_column = self.get_column("Case")
        self.out_column = self.get_column("outside")
        # Sometimes different
        if self.out_column == []:
            self.out_column = self.get_column("out")
        if data_kwargs['unit'] == 'UMAR':
            self.neigh_column = self.get_column("1")
            self.neigh_column = []
            for room in ['272', '273', '274', '275', '276']:
                if (room not in self.rooms) and (f"Thermal temperature measurement {room}" in self.dataset.X_columns):
                    self.neigh_column.append(self.get_column(f"temperature measurement {room}"))
        else:
            self.neigh_column = self.get_column([x for x in self.dataset.X_columns if
                                                 ("temperature measurement" in x) & (x not in self.dataset.Y_columns)])

        if self.topology is not None:
            self.rooms = self.topology['Rooms']
            if data_kwargs['unit'] == 'UMAR':
                self.valve_column = [self.get_column(f"Thermal valve {room}") for room in self.topology['Rooms']]
            else:
                self.valve_column = [self.get_column(f"Thermal valves {room}") for room in self.topology['Rooms']]
            self.temperature_column = [self.get_column(f'Thermal temperature measurement {room}') for room in
                                       self.topology['Rooms']]
            self.power_column = [self.get_column(f"Power room {room}") for room in self.topology['Rooms']]
        else:
            self.valve_column = [self.get_column("valve")]
            self.temperature_column = [self.get_column([x for x in self.dataset.Y_columns if "temperature" in x])]
            self.power_column = [self.get_column("Power room")]

        # Sanity check
        if self.verbose > 0:
            print('\nCheck:\n', [(w, [self.dataset.X_columns[i] for i in x]) for w, x in zip(['Case', 'Valve', 'Room temp',
                                                                                    'Room power', 'Out temp',
                                                                                    'Neigh temp'],
                                                                                   [[self.case_column], self.valve_column,
                                                                                    self.temperature_column,
                                                                                    self.power_column,
                                                                                    [self.out_column],
                                                                                    self.neigh_column])])

        if len(self.neigh_column) == 1:
            self.neigh_column = self.neigh_column[0]

        # Compute the scaled zero power points and the division factors to use in ResNet-like
        # modules
        self.zero_power = self.compute_zero_power()
        if self.predict_power:
            self.total_zero_power = (0 - self.dataset.min_['Thermal total power']) \
                                    / (self.dataset.max_ - self.dataset.min_)['Thermal total power'] * 0.8 + 0.1
        self.base_division_factor, self.cooling_division_factor, \
        self.heating_division_factor = self.create_division_factors()

        self.normalization_variables = self.get_normalization_variables()
        self.parameter_scalings = self.create_scalings()

        if (not self.predict_power) & (data_kwargs['unit'] == 'DFAB'):
            self.dataset.Y = self.dataset.Y[:,:-1]

        # Prepare the torch module
        if self.module == "TemperaturePowerTogether":
            self.model = TemperaturePowerTogether(
                device=self.device,
                input_size=self.X.shape[1],
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                zero_power=self.zero_power,
            )

        elif self.module == "SeparateTemperaturePower":
            self.model = SeparateTemperaturePower(
                device=self.device,
                input_size=self.X.shape[1],
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                zero_power=self.zero_power,
            )

        elif self.module == "SeparateInputs":
            self.model = SeparateInputs(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                zero_power=self.zero_power,
            )

        elif self.module == "DiscountedModules":
            self.model = DiscountedModules(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                zero_power=self.zero_power,
            )

        elif self.module == "DiscountedModulesBase":
            self.model = DiscountedModulesBase(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                zero_power=self.zero_power,
            )

        elif self.module == "ResDiscountedModulesBase":
            self.model = ResDiscountedModulesBase(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                cooling_division_factor=self.cooling_division_factor,
                heating_division_factor=self.heating_division_factor,
                predict_power=self.predict_power
            )

        elif self.module == "ResDiscountedModulesBaseTrueEnergy":
            self.model = ResDiscountedModulesBaseTrueEnergy(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                cooling_division_factor=self.cooling_division_factor,
                heating_division_factor=self.heating_division_factor,
                predict_power=self.predict_power
            )


        elif self.module == "SimpleLSTMBase":
            self.model = SimpleLSTMBase(
                device=self.device,
                base_indices=self.base_indices,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                base_division_factor=self.base_division_factor
            )


        elif self.module == "PhysicsInspiredModuleBase":
            self.model = PhysicsInspiredModuleBase(
                device=self.device,
                effect_indices=self.effect_indices,
                base_indices=self.base_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                out_column=self.out_column,
                neigh_column=self.neigh_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                normalization_variables=self.normalization_variables,
                parameter_scalings=self.parameter_scalings,
                predict_power=self.predict_power
            )

        elif self.module == "PCNN_mul":
            self.model = PCNN_mul(
                device=self.device,
                base_indices=self.base_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                out_column=self.out_column,
                neigh_column=self.neigh_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                normalization_variables=self.normalization_variables,
                parameter_scalings=self.parameter_scalings,
            )


        elif self.module == "PCNN_full_shared_mul":
            self.model = PCNN_full_shared_mul(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                out_column=self.out_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                normalization_variables=self.normalization_variables,
                parameter_scalings=self.parameter_scalings,
                topology=self.topology,
                predict_power=self.predict_power,
            )

        elif self.module == "PCNNTestQuantiles":
            self.model = PCNNTestQuantiles(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                out_column=self.out_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                normalization_variables=self.normalization_variables,
                parameter_scalings=self.parameter_scalings,
                topology=self.topology,
                predict_power=self.predict_power,
            )

        elif self.module == "PCNN_full_shared":
            self.model = PCNN_full_shared(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                out_column=self.out_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                normalization_variables=self.normalization_variables,
                parameter_scalings=self.parameter_scalings,
                topology=self.topology,
                predict_power=self.predict_power,
            )

        elif self.module == "PCNN_full":
            self.model = PCNN_full(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                out_column=self.out_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                normalization_variables=self.normalization_variables,
                parameter_scalings=self.parameter_scalings,
                topology=self.topology,
                predict_power=self.predict_power
            )

        elif self.module == "PCNN":
            self.model = PCNN(
                device=self.device,
                base_indices=self.base_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                out_column=self.out_column,
                neigh_column=self.neigh_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                normalization_variables=self.normalization_variables,
                parameter_scalings=self.parameter_scalings,
            )

        elif self.module == "ResDiscountedModulesBaseMany":
            self.model = ResDiscountedModulesBaseMany(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                cooling_division_factor=self.cooling_division_factor,
                heating_division_factor=self.heating_division_factor
            )

        elif self.module == "ResDiscountedModulesBaseManySep":
            self.model = ResDiscountedModulesBaseManySep(
                device=self.device,
                base_indices=self.base_indices,
                effect_indices=self.effect_indices,
                learn_discount_factors=self.learn_discount_factors,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                feed_output_through_nn=self.feed_output_through_nn,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                valve_column=self.valve_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                zero_power=self.zero_power,
                base_division_factor=self.base_division_factor,
                cooling_division_factor=self.cooling_division_factor,
                heating_division_factor=self.heating_division_factor
            )

        self.optimizer = optim.Adam(self.model.parameters(), lr=model_kwargs["learning_rate"])
        if False:
            self.optimizer = optim.Adam([
            {'params': [p[1] for p in self.model.named_parameters() if (('a.' not in p[0]) & ('b.' not in p[0]) & ('c.' not in p[0]) & ('d.' not in p[0]))]},
            {'params': [p[1] for p in self.model.named_parameters() if (('a.' in p[0]) | ('b.' in p[0]) | ('c.' in p[0]) | ('d.' in p[0]))], 'lr': model_kwargs["learning_rate"]*5}
            ], lr=model_kwargs["learning_rate"])
        self.loss = F.mse_loss
        self.quantiles = [0.1, 0.5, 0.9]

        # Load the model if it exists
        if load:
            self.load(load_last=load_last)

        # if the model doesn't exist, the sequences were not loaded
        if self.train_sequences is None:
            self.heating_sequences, self.cooling_sequences = self.get_sequences()
            self.train_test_validation_separation(validation_percentage=self.validation_percentage,
                                                  test_percentage=self.test_percentage)

        self.model = self.model.to(self.device)

    def compute_zero_power(self):
        """
        Small helper function to compute the scaled value of zero power
        """

        # Scale the zero
        if self.dataset.is_normalized:
            min_ = self.dataset.min_[self.power_column]
            max_ = self.dataset.max_[self.power_column]
            zero = 0.8 * (0.0 - min_) / (max_ - min_) + 0.1

        elif self.dataset.is_standardized:
            mean = self.dataset.mean[self.power_column]
            std = self.dataset.std[self.power_column]
            zero = (0.0 - mean) / std

        else:
            zero = np.array([0.0] * len(self.rooms))

        return np.array(zero)

    def create_division_factors(self):
        """
        Function to define scaling factors of the neural networks: it looks at the observed differences from
        one timestep to another and creates a factor for each room to make sure the predictions of the network
        doesn't create unobserved differences (e.g. predicting a difference of 3 degrees over one step).

        The factors are computed at the 0.01 and 0.99 quantiles of the data and divided by 2 to reflect
        the fact that the network's output is a sum over 2 predictions (base +/- heating/cooling effect)

        Returns:
            division factors for the base, heating and cooling modules
        """

        # Look at the largest negative difference (1% quantile) to cap the cooling effect
        cooling_division_factor = - 2 / np.nanquantile(np.diff(self.X[:, self.temperature_column], axis=0), q=0.01,
                                                       axis=0)
        # Look at the largest positive difference (99% quantile) to cap the cooling effect
        heating_division_factor = 2 / np.nanquantile(np.diff(self.X[:, self.temperature_column], axis=0), q=0.99,
                                                     axis=0)

        # Create the base division factor as the biggest one between the heating and cooling one
        if len(self.rooms) > 1:
            base_division_factor = np.array(
                [max(cool, heat) for cool, heat in zip(cooling_division_factor, heating_division_factor)])

            # return list(base_division_factor), list(cooling_division_factor), list(heating_division_factor)
            return [10.], [10.], [10.]

        else:
            base_division_factor = max(cooling_division_factor, heating_division_factor)

            return [10.], [10.], [10.]

            # return [base_division_factor], [cooling_division_factor], [heating_division_factor]

    def create_scalings(self):
        """
        Function to initialize good parameters for a, b, c and d, the key parameters of the structure.
        Intuition:
          - The room loses 1.5 degrees in 6h when the outside temperature is 25 degrees lower than
              the inside one (and some for losses to the neighboring room)
          - The room gains 2 degrees in 4h of heating

        Returns:
            The scaling parameters according to the data
        """

        parameter_scalings = {}

        if self.dataset.unit == 'DFAB':
            # DFAB power is in kW
            parameter_scalings['a'] = [
                1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                     / (self.dataset.max_[self.power_column[i]] / (self.dataset.max_ - self.dataset.min_)[
                            self.power_column[i]] * 0.8)  # With average power of 1500 W
                     / (4 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # In 4h
            parameter_scalings['d'] = [
                1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                     / (- self.dataset.min_[self.power_column[i]] / (self.dataset.max_ - self.dataset.min_)[
                            self.power_column[i]] * 0.8)  # With average power of 1000 W
                     / (4 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # In 4h

            parameter_scalings['b'] = 1 / (1.5 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                           * 0.8 / 25 / 6 / 60 * self.dataset.interval).mean()
            parameter_scalings['c'] = 1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                           * 0.8 / 25 / 6 / 60 * self.dataset.interval).mean()

            parameter_scalings['a'] = [
                1 / (1 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                     / (self.dataset.max_[self.power_column[i]] / (self.dataset.max_ - self.dataset.min_)[
                            self.power_column[i]] * 0.8)  # With average power of 1500 W
                     / (10 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # In 4h
            parameter_scalings['d'] = [
                1 / (1 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                     / (- self.dataset.min_[self.power_column[i]] / (self.dataset.max_ - self.dataset.min_)[
                            self.power_column[i]] * 0.8)  # With average power of 1000 W
                     / (10 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # In 4h

            parameter_scalings['b'] = 1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                           * 0.8 / 25 / 6 / 60 * self.dataset.interval).mean()
            parameter_scalings['c'] = 1 / (2.5 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                           * 0.8 / 25 / 6 / 60 * self.dataset.interval).mean()

            parameter_scalings['a'] = [
                1 / (1 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                     / (self.dataset.max_['Thermal total power'] / (self.dataset.max_ - self.dataset.min_)[
                            'Thermal total power'] * 0.8)  # With average power of 1500 W
                     / (10 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # In 4h
            parameter_scalings['d'] = [
                1 / (1 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                     / (- self.dataset.min_['Thermal total power'] / (self.dataset.max_ - self.dataset.min_)[
                            'Thermal total power'] * 0.8)  # With average power of 1000 W
                     / (10 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # In 4h

        else:
            parameter_scalings['b'] = 1 / (1.5 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                           * 0.8 / 25 / 6 / 60 * self.dataset.interval).mean()
            parameter_scalings['c'] = 1 / (1.5 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                           * 0.8 / 25 / 6 / 60 * self.dataset.interval).mean()

            # needed condition to make sure to deal with data in Watts and kWs
            if (self.dataset.max_ - self.dataset.min_)[self.power_column[0]] > 100:
                parameter_scalings['a'] = [
                    1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                         / (1000 / (self.dataset.max_ - self.dataset.min_)[
                                self.power_column[i]] * 0.8)  # With average power of 1.5 kW
                         / (4 * 60 / self.dataset.interval)) for i in range(len(self.rooms))] # in 4h
                parameter_scalings['d'] = [
                    1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                         / (1000 / (self.dataset.max_ - self.dataset.min_)[
                                self.power_column[i]] * 0.8)  # With average power of 1.5 kW
                         / (4 * 60 / self.dataset.interval)) for i in range(len(self.rooms))] # in 4h
            else:
                parameter_scalings['a'] = [
                    1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                         / (self.dataset.max_[self.power_column[i]] * 3/2 / (self.dataset.max_ - self.dataset.min_)[
                                self.power_column[i]] * 0.8)  # With average power of 1.5 kW
                         / (4 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # in 4h
                parameter_scalings['d'] = [
                    1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                         / (self.dataset.max_[self.power_column[i]] * 3/2 / (self.dataset.max_ - self.dataset.min_)[
                                self.power_column[i]] * 0.8)  # With average power of 1.5 kW
                         / (4 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # in 4h

            # PCNN paper
            if False:
                parameter_scalings['a'] = 1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                               * 0.8 / 0.25 / 4 / 60 * self.dataset.interval)
                parameter_scalings['d'] = 1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                               * 0.8 / 0.25 / 4 / 60 * self.dataset.interval)

        return parameter_scalings

    def get_normalization_variables(self):
        """
        Function to get the minimum and the amplitude of some variables in the data. In particular, we need
        that for the room temperature, the outside temperature and the neighboring room temperature.
        This is used by the physics-inspired network to unnormalize the predictions.
        """
        normalization_variables = {}
        normalization_variables['Room'] = [self.dataset.min_[self.temperature_column],
                                           (self.dataset.max_ - self.dataset.min_)[self.temperature_column]]
        normalization_variables['Out'] = [self.dataset.min_[self.out_column],
                                          (self.dataset.max_ - self.dataset.min_)[self.out_column]]
        normalization_variables['Neigh'] = [self.dataset.min_[self.neigh_column],
                                            (self.dataset.max_ - self.dataset.min_)[self.neigh_column]]
        normalization_variables['High'] = [self.dataset.min_[self.get_column('high')],
                                            (self.dataset.max_ - self.dataset.min_)[self.get_column('high')]]
        normalization_variables['Low'] = [self.dataset.min_[self.get_column(' low')],
                                            (self.dataset.max_ - self.dataset.min_)[self.get_column(' low')]]
        return normalization_variables

    def batch_iterator(self, iterator_type: str = "train", batch_size: int = None, shuffle: bool = True) -> None:
        """
        Function to create batches of the data with the wanted size, either for training,
        validation, or testing

        Args:
            iterator_type:  To know if this should handle training, validation or testing data
            batch_size:     Size of the batches
            shuffle:        Flag to shuffle the data before the batches creation

        Returns:
            Nothing, yields the batches to be then used in an iterator
        """

        # Firstly control that the training sequences exist - create them otherwise
        if self.train_sequences is None:
            self.train_test_validation_separation()
            print("The Data was not separated in train, validation and test --> the default 70%-20%-10% was used")

        # If no batch size is given, define it as the default one
        if batch_size is None:
            batch_size = self.batch_size

        # Some print
        if self.verbose > 0:
            print(f"Creating the {iterator_type} batch iterator...")

        # Copy the indices of the correct type (without first letter in case of caps)
        if "rain" in iterator_type:
            sequences = self.train_sequences
        elif "alidation" in iterator_type:
            sequences = self.validation_sequences
        elif "est" in iterator_type:
            sequences = self.test_sequences
        else:
            raise ValueError(f"Unknown type of batch creation {iterator_type}")

        # Shuffle them if wanted
        if shuffle:
            np.random.shuffle(sequences)

        # Define the right number of batches according to the wanted batch_size - taking care of the
        # special case where the indicies ae exactly divisible by the batch size, which can induce
        # an additional empty batch breaking the simulation down the line
        n_batches = int(np.ceil(len(sequences) / batch_size))

        # Iterate to yield the right batches with the wanted size
        for batch in range(n_batches):
            yield sequences[batch * batch_size: (batch + 1) * batch_size]

    def build_input_output_from_sequences(self, sequences: list):
        """
        Input and output generator from given sequences of indices corresponding to a batch.

        Args:
            sequences: sequences of the batch to prepare

        Returns:
            batch_x:    Batch input of the model
            batch_y:    Targets of the model, the temperature and the power
        """

        # Ensure the given sequences are a list of list, not only one list
        if type(sequences) == tuple:
            sequences = [sequences]

        # Iterate over the sequences to build the input in the right form
        input_tensor_list = [torch.FloatTensor(self.X[sequence[0]: sequence[1], :].copy()) for sequence in sequences]

        # Prepare the output for the temperature and power consumption
        if self.predict_differences:
            output_tensor_list = [torch.FloatTensor(self.differences_Y[sequence[0]: sequence[1], :].copy()) for sequence
                                  in sequences]
        else:
            output_tensor_list = [torch.FloatTensor(self.Y[sequence[0]: sequence[1], :].copy()) for sequence in
                                      sequences]


        # Build the final results by taking care of the batch_size=1 case
        if len(sequences) > 1:
            batch_x = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
            if not self.predict_power:
                batch_y = pad_sequence(output_tensor_list, batch_first=True, padding_value=0).reshape(len(sequences),
                                                                                                  batch_x.shape[1], 2,
                                                                                                  len(self.rooms))
            else:
                batch_y = pad_sequence(output_tensor_list, batch_first=True, padding_value=0)
        else:
            batch_x = input_tensor_list[0].view(1, input_tensor_list[0].shape[0], -1)
            if not self.predict_power:
                batch_y = output_tensor_list[0].view(1, output_tensor_list[0].shape[0], 2, len(self.rooms))
            else:
                batch_y = output_tensor_list[0].view(1, output_tensor_list[0].shape[0], -1)

        # Return everything
        return batch_x.to(self.device), batch_y.to(self.device)

    def predict(self, sequences: Union[list, int] = None, data: torch.FloatTensor = None, mpc_mode: bool = False):
        """
        Function to predict batches of "sequences", i.e. it creates batches of input and output of the
        given sequences you want to predict and forwards them through the network

        Args:
            sequences:  Sequences of the data to predict
            data:       Alternatively, data to predict, a tuple of tensors with the X and Y (if there is no
                          Y just put a vector of zeros with the right output size)
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predictions and the true output
        """

        if sequences is not None:
            # Ensure the given sequences are a list of list, not only one list
            if type(sequences) == tuple:
                sequences = [sequences]

            # Build the input and output
            batch_x, batch_y = self.build_input_output_from_sequences(sequences=sequences)

        elif data is not None:
            #batch_x = data[0].reshape(1, data[0].shape[0], -1)
            #batch_y = data[1].reshape(1, data[0].shape[0], -1)
            batch_x = data[0].reshape(data[0].shape[0], data[0].shape[1], -1)
            batch_y = data[1].reshape(data[0].shape[0], data[0].shape[1],  2, len(self.rooms))

        else:
            raise ValueError("Either sequences or data must be provided to the `predict` function")

        # Initialize the needed quantities
        if self.module == 'PCNNTestQuantiles':
            predictions = torch.zeros((batch_y.shape[0], batch_y.shape[1],
                                       2, len(self.rooms), len(self.quantiles))).to(self.device)
            states = None

            # Warm start step to get the initial hidden and cell states
            for i in range(self.warm_start_length):
                # Predict the next output and store it
                if ("Base" not in self.module) and ("PCNN" not in self.module):
                    pred, states = self.model(batch_x[:, i, :], states)
                else:
                    pred, states = self.model(batch_x[:, i, :], states, warm_start=True, mpc_mode=mpc_mode)
                predictions[:, i, :, :, :] = pred

            # Iterate through the sequences of data to predict each step, replacing the true power and temperature
            # values with the predicted ones each time
            for i in range(self.warm_start_length, batch_x.shape[1]):
                # Predict the next output and store it
                if "PCNN" not in self.module:
                    pred, states = self.model(batch_x[:, i, :], states)
                else:
                    pred, states = self.model(batch_x[:, i, :], states, mpc_mode=mpc_mode)
                predictions[:, i, :, :, :] = pred

                # Modify the input according to the prediction when required
                if (i < batch_x.shape[1] - 1) and ("Base" not in self.module) and ("PCNN" not in self.module):
                    batch_x[:, i + 1, self.temperature_column] = pred[:, 0, 0, :].clone()
                    batch_x[:, i + 1, self.power_column] = pred[:, 0, 1, :].clone()
        else:
            predictions = torch.zeros((batch_y.shape[0], batch_y.shape[1],
                                   2, len(self.rooms))).to(self.device)
            states = None

            # Warm start step to get the initial hidden and cell states
            for i in range(self.warm_start_length):
                # Predict the next output and store it
                if ("Base" not in self.module) and ("PCNN" not in self.module):
                    pred, states = self.model(batch_x[:, i, :], states)
                else:
                    pred, states = self.model(batch_x[:, i, :], states, warm_start=True, mpc_mode=mpc_mode)
                predictions[:, i, :, :] = pred

            # Iterate through the sequences of data to predict each step, replacing the true power and temperature
            # values with the predicted ones each time
            for i in range(self.warm_start_length, batch_x.shape[1]):
                # Predict the next output and store it
                if "PCNN" not in self.module:
                    pred, states = self.model(batch_x[:, i, :], states)
                else:
                    pred, states = self.model(batch_x[:, i, :], states, mpc_mode=mpc_mode)
                predictions[:, i, :, :] = pred

                # Modify the input according to the prediction when required
                if (i < batch_x.shape[1] - 1) and ("Base" not in self.module) and ("PCNN" not in self.module):
                    batch_x[:, i + 1, self.temperature_column] = pred[:, 0, 0, :].clone()
                    batch_x[:, i + 1, self.power_column] = pred[:, 0, 1, :].clone()

        return predictions, batch_y

    def scale_back_predictions(self, sequences: Union[list, int] = None, data: torch.FloatTensor = None):
        """
        Function preparing the data for analyses: it predicts the wanted sequences and returns the scaled
        predictions and true_data

        Args:
            sequences:  Sequences to predict
            data:       Alternatively, data to predict, a tuple of tensors with the X and Y (if there is no
                          Y just put a vector of zeros with the right output size)

        Returns:
            The predictions and the true data
        """

        # Compute the predictions and get the true data out of the GPU
        predictions, true_data = self.predict(sequences=sequences, data=data)
        predictions = predictions.cpu().detach().numpy()
        true_data = true_data.cpu().detach().numpy()
        unnorm_pred = predictions[:, :, 1, :].copy()

        # Reshape the data for consistency with the next part of the code if only one sequence is given
        if sequences is not None:
            # Reshape when only 1 sequence given
            if type(sequences) == tuple:
                sequences = [sequences]

        elif data is not None:
            sequences = [0]

        else:
            raise ValueError("Either sequences or data must be provided to the `scale_back_predictions` function")

        if len(predictions.shape) == 3:
            predictions = predictions.reshape(1, predictions.shape[0], predictions.shape[1], -1)
            true_data = true_data.reshape(1, true_data.shape[0], true_data.shape[1], -1)

        # Scale the data back

        cols = self.dataset.Y_columns[:-1] if self.predict_power else self.dataset.Y_columns[:-2]
        truth = true_data.reshape(true_data.shape[0], true_data.shape[1], -1)\
            if not self.predict_power else true_data[:,:,:-1]
        true = np.zeros_like(predictions)

        if self.dataset.is_normalized:
            for i, sequence in enumerate(sequences):
                predictions[i, :, :, :] = inverse_normalize(data=predictions[i, :, :, :].reshape(predictions.shape[1], -1),
                                                         min_=self.dataset.min_[cols],
                                                         max_=self.dataset.max_[cols]).reshape(predictions.shape[1], 2, -1)
                true[i, :, :, :] = inverse_normalize(data=truth[i, :, :].reshape(true_data.shape[1], -1),
                                                       min_=self.dataset.min_[cols],
                                                       max_=self.dataset.max_[cols]).reshape(true_data.shape[1], 2, -1)
        elif self.dataset.is_standardized:
            for i, sequence in enumerate(sequences):
                predictions[i, :, :, :] = inverse_standardize(data=predictions[i, :, :, :].reshape(predictions.shape[1], -1),
                                                           mean=self.dataset.mean[cols],
                                                           std=self.dataset.std[cols]).reshape(predictions.shape[1], 2, -1)
                true[i, :, :, :] = inverse_standardize(data=truth[i, :, :].reshape(true_data.shape[1], -1),
                                                         mean=self.dataset.mean[cols],
                                                         std=self.dataset.std[cols]).reshape(true_data.shape[1], 2, -1)

        if not self.predict_power:
            return predictions, true
        else:
            return (predictions, unnorm_pred), (true, true_data[:, :, -1])

    def plot_predictions(self, sequences: Union[list, int] = None, data: torch.FloatTensor = None, **kwargs) -> None:
        """
        Function to plot predictions and true data from certain sequences for visual assessement.

        Args:
            sequences:  Sequences to predict and plot
            data:       Alternatively, data to predict and plot, a tuple of tensors with the X and Y (if there is no
                          Y just put a vector of zeros with the right output size)
            **kwargs:   Kwargs of the plots, see util.plots
        """

        # Prepare the data
        if not self.predict_power:
            predictions, true_data = self.scale_back_predictions(sequences=sequences, data=data)
        else:
            (predictions, unnorm_pred), (true_data, total) = self.scale_back_predictions(sequences=sequences, data=data)

        if data is not None:
            # For the enumerate below to work
            sequences = [0]

        else:
            if type(sequences) == tuple:
                sequences = [sequences]

        for i, sequence in enumerate(sequences):
            # Design the plot with custom helpers
            fig, ax = _plot_helpers(subplots=[len(self.rooms) if not self.predict_power else len(self.rooms) + 1, 2],
                                    sharex=True, sharey=False, figsize=(20, 5 * len(self.rooms)), **kwargs)

            if len(ax.shape) == 1:
                ax = ax.reshape(1, -1)

            if data is None:
                index = self.dataset.data.index[sequence[0] + self.warm_start_length:sequence[1]]

                # Plot both informations
                for j in range(len(self.rooms)):
                    ax[j, 0].plot(index, true_data[i, :sequence[1] - sequence[0] - self.warm_start_length, 0, j],
                                  label="Observations")
                    ax[j, 0].plot(index, predictions[i, :sequence[1] - sequence[0] - self.warm_start_length, 0, j],
                                  label="Prediction")
                    ax[j, 1].plot(index, true_data[i, :sequence[1] - sequence[0] - self.warm_start_length, 1, j],
                                  label="Observations")
                    if self.predict_power:
                        ax[j, 1].plot(index, (unnorm_pred[i, :sequence[1] - sequence[0] - self.warm_start_length, j]
                                              + self.total_zero_power - 0.1) / 0.8
                                      * (self.dataset.max_ - self.dataset.min_)['Thermal total power']
                                      + self.dataset.min_['Thermal total power'], label="Predictions")
                    else:
                        ax[j, 1].plot(index, predictions[i, :sequence[1] - sequence[0] - self.warm_start_length, 1, j],
                                  label="Prediction")
                    ax[j, 0].set_title(f"Temperature room {self.rooms[j]}", size=22)
                    ax[j, 1].set_title(f"Power room {self.rooms[j]}", size=22)

                if self.predict_power:
                    for j in range(2):
                        ax[-1, j].plot(index, (total[i, :sequence[1] - sequence[0] - self.warm_start_length] - 0.1)
                                       / 0.8 * (self.dataset.max_ - self.dataset.min_)['Thermal total power']
                                       + self.dataset.min_['Thermal total power'], label="Observations")
                        ax[-1, j].plot(index, (unnorm_pred[i, :sequence[1] - sequence[0] - self.warm_start_length, :].sum(axis=-1)
                                               + self.total_zero_power - 0.1) / 0.8
                                       * (self.dataset.max_ - self.dataset.min_)['Thermal total power']
                                       + self.dataset.min_['Thermal total power'], label="Predictions")
                        ax[-1, j].set_title(f"Total power", size=22)

            else:
                # Plot both informations
                for j in range(len(self.rooms)):
                    ax[j, 0].plot(true_data[i, :, 0, j], label="Observations")
                    ax[j, 0].plot(predictions[i, :, 0, j], label="Prediction")
                    ax[j, 1].plot(true_data[i, :, 1, j], label="Observations")
                    if self.predict_power:
                        ax[j, 1].plot(index, (unnorm_pred[i, :, j] + self.total_zero_power - 0.1) / 0.8
                                      * (self.dataset.max_ - self.dataset.min_)['Thermal total power']
                                      + self.dataset.min_['Thermal total power'], label="Predictions")
                    else:
                        ax[j, 1].plot(predictions[i, :, 1, j], label="Prediction")

                    ax[j, 0].set_title(f"Temperature room {self.rooms[j]}", size=22)
                    ax[j, 1].set_title(f"Power room {self.rooms[j]}", size=22)

                if self.predict_power:
                    for j in range(2):
                        ax[-1, j].plot((total[i, :] - 0.1)
                                       / 0.8 * (self.dataset.max_ - self.dataset.min_)['Thermal total power']
                                       + self.dataset.min_['Thermal total power'], label="Observations")
                        ax[-1, j].plot((unnorm_pred[i, :, :].sum(axis=-1) + self.total_zero_power - 0.1) / 0.8 *
                                       (self.dataset.max_ - self.dataset.min_)['Thermal total power']
                                       + self.dataset.min_['Thermal total power'], label="Predictions")
                        ax[-1, j].set_title(f"Total power", size=22)

            ax[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%h-%d %H:%M"))
            ax[-1, 0].set_xlabel("Time", size=20)
            ax[-1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%h-%d %H:%M"))
            ax[-1, 1].set_xlabel("Time", size=20)

            fig.autofmt_xdate()
            # Save or show the plot
            _save_or_show(legend=True, **kwargs)

    def predictions_analysis(self, sequences: list, horizon: int = 3 * 96, number: int = 500, return_: bool = False,
                             save: bool = False, showfliers: int = False, abs_: bool = True,
                             old_version: bool = False, **kwargs) -> None:
        """
        Function analyzing the prediction performances of the model in terms of (absolute) error and
        percentage errors. It predicts both errors, possibly in absolute terms, over the given sequences.
        It currently plots histograms of errors and boxplots over a certain horizon with hourly time steps.

        Args:
            sequences:      Sequences to analyze
            horizon:        Prediction horizon to analyze
            number:         Number of sequences to analyze
            return_:        Flag to set to True if you want to return the analysis instead of plotting it directly
            save:           Flag whether to save the plots
            showfliers:     Whether to show the outliers for the boxplots
            abs_:           Flag to consider absolute error instead of classical error
            old_version:    Flag to plot old statistics, i.e. all the errors, and the mean +/- std stats

        Returns:
            The absolute errors of predictions or plot them
        """

        # Print the start
        print("Warning: Boxplots are consistent for 15 minutes interval, what about the rest?")

        # Get a copy of the validation indices and shuffle it
        seqs = sequences.copy()
        np.random.shuffle(seqs)

        # Take long enough sequences to have interesting predictions
        seqs = [sequence for sequence in seqs if
                sequence[1] - sequence[0] - self.warm_start_length >= horizon]
        seqs = seqs[:number]
        print(f"Analyzing {len(seqs)} predictions...")

        # Prepare the data and compute the errors
        predictions, true_data = self.scale_back_predictions(sequences=seqs)
        if abs_:
            errors_list = [np.abs(predictions - true_data)[:, :horizon, :, :],
                           100 * np.abs(predictions - true_data)[:, :horizon, :, :] / np.abs(true_data)[:, :horizon, :, :]]
        else:
            errors_list = [(predictions - true_data)[:, :horizon, :, ],
                           100 * (predictions - true_data)[:, :horizon, :, ] / true_data[:, :horizon, :, :]]

        if return_:
            return errors_list

        ## PLOTS
        else:

            # Histograms for each room of the temperature and power errors
            # First on the (absolute) errors, then on the (absolute) percentage errors
            for k, errors in enumerate(errors_list):
                fig, ax = _plot_helpers(subplots=[len(self.rooms), 2], sharex=False, sharey=False,
                                        figsize=(20, 5 * len(self.rooms)), **kwargs)

                if len(ax.shape) == 1:
                    ax = ax.reshape(1, -1)

                # Loop over the rooms
                for j in range(len(self.rooms)):
                    # Print the quantiles of the errors for the temperature only
                    if k == 0:
                        if j == 0:
                            print("\nAbsolute errors:" if abs_ else "\nErrors:", end="")
                        print(f"\nQuantiles {self.rooms[j]} ", end="")
                    else:
                        if j == 0:
                            print(f"\nAbsolute Percentage errors:" if abs_ else "\nPercentage errors:", end="")
                        print(f"\nQuantiles {self.rooms[j]} ", end="")
                    if abs_:
                        for i, q in enumerate([0.5, 0.75, 0.9, 0.95, 0.975]):
                            print(f" | {q}: {np.quantile(errors[:, :, 0, j].flatten(), q=q):.3f}", end="")
                            ax[j, 0].axvline(np.quantile(errors[:, :, 0, j].flatten(), q=q), color="black",
                                             linewidth=5 - i)
                    else:
                        for i, q in enumerate([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
                            print(f"{q}: {np.quantile(errors[:, :, 0, j].flatten(), q=q):.3f} | ", end="")
                            ax[j, 0].axvline(np.quantile(errors[:, :, 0, j].flatten(), q=q), color="black",
                                             linewidth=i + 1 if i < 4 else 7 - i)

                    # Histograms
                    ax[j, 0].hist(errors[:, :, 0, j].flatten(), bins=1000)
                    # Need to be careful, power is often zero, and so the errors are often zero as
                    # well - just plot when it's not the case
                    power_errors = errors[:, :, 1, j]
                    ax[j, 1].hist(power_errors[np.where(power_errors > 1e-6)].flatten(), bins=1000)
                    ax[j, 0].set_ylabel("Occurances", size=20)
                    ax[j, 0].set_title(f"Temperature errors {self.rooms[j]}", size=22)
                    ax[j, 1].set_title(f"Power errors {self.rooms[j]}", size=22)

                ax[-1, 0].set_xlabel("Error", size=20)
                ax[-1, 1].set_xlabel("Error", size=20)

                _save_or_show(save=save, save_name=f"{self.name}_Error_histograms", **kwargs)

            # Boxplot of the temperature errors at each hour ahead
            fig, ax = _plot_helpers(subplots=[len(self.rooms), 2], sharex=True, sharey=False,
                                    figsize=(20, 5 * len(self.rooms)), **kwargs)

            if len(ax.shape) == 1:
                ax = ax.reshape(1, -1)

            for j in range(len(self.rooms)):
                ax[j, 0].boxplot([errors_list[0][:, i:i + 4, 0, j].flatten() for i in range(0, horizon, 4)], notch=True,
                                 whis=[5, 95], showfliers=showfliers)
                ax[j, 1].boxplot([errors_list[1][:, i:i + 4, 0, j].flatten() for i in range(0, horizon, 4)], notch=False,
                                 whis=[5, 95], showfliers=showfliers)
                ax[j, 0].set_ylabel("AE", size=20)
                ax[j, 1].set_ylabel("APE", size=20)
                ax[j, 0].set_title(f"Temperature errors {self.rooms[j]}", size=22)
                ax[j, 1].set_title(f"Temperature errors {self.rooms[j]}", size=22)

            ax[-1, 0].set_xlabel("Time step ahead", size=20)
            ax[-1, 1].set_xlabel("Time step ahead", size=20)

            _save_or_show(save=save, save_name=f"{self.name}_Error_boxplots_temp", **kwargs)

            # Boxplots of the power errors at each hour ahead
            fig, ax = _plot_helpers(subplots=[len(self.rooms), 2], sharex=True, sharey=False,
                                    figsize=(20, 5 * len(self.rooms)), **kwargs)

            if len(ax.shape) == 1:
                ax = ax.reshape(1, -1)

            for j in range(len(self.rooms)):
                ax[j, 0].boxplot([errors_list[0][:, i:i + 4, 1, j][
                                      np.where(errors_list[0][:, i:i + 4, 1, j] > 1e-6)].flatten()
                                  for i in range(0, horizon, 4)], notch=True, whis=[5, 95], showfliers=showfliers)
                ax[j, 1].boxplot([errors_list[1][:, i:i + 4, 1, j][
                                      np.where(errors_list[1][:, i:i + 4, 1, j] > 1e-6)].flatten()
                                  for i in range(0, horizon, 4)], notch=False, whis=[5, 95], showfliers=showfliers)
                ax[j, 0].set_ylabel("AE", size=20)
                ax[j, 1].set_ylabel("APE", size=20)
                ax[j, 0].set_title(f"Power errors {self.rooms[j]}", size=22)
                ax[j, 1].set_title(f"Power errors {self.rooms[j]}", size=22)

            ax[-1, 0].set_xlabel("Time step ahead", size=20)
            ax[-1, 1].set_xlabel("Time step ahead", size=20)

            _save_or_show(save=save, save_name=f"{self.name}_Error_boxplots_power", **kwargs)

            if old_version:

                fig, ax = _plot_helpers(subplots=[2, 1], sharex=False, sharey=False, **kwargs)

                for i in range(errors.shape[0]):
                    ax[0].plot(np.arange(horizon), errors[i, :, 0], color="blue", alpha=0.1)
                    ax[1].plot(np.arange(horizon), errors[i, :, 1], color="blue", alpha=0.1)
                ax[0].plot(np.arange(horizon), np.mean(errors[:, :, 0], axis=0), color="black", alpha=1, linewidth=2)
                ax[1].plot(np.arange(horizon), np.mean(errors[:, :, 1], axis=0), color="black", alpha=1, linewidth=2)
                ax[0].set_title("Temperature errors", size=22)
                ax[0].set_ylabel("Degrees", size=20)
                ax[1].set_title("Power errors", size=22)
                ax[1].set_ylabel("Power", size=20)

                _save_or_show(save=save, save_name=f"{self.name}_Error_trajectories", **kwargs)

                fig, ax = _plot_helpers(subplots=[2, 1], sharex=False, sharey=False, **kwargs)

                ax[0].fill_between(np.arange(horizon),
                                   np.mean(errors[:, :, 0], axis=0) - 2 * np.std(errors[:, :, 0], axis=0),
                                   np.mean(errors[:, :, 0], axis=0) + 2 * np.std(errors[:, :, 0], axis=0),
                                   color="blue", alpha=0.1)
                ax[1].fill_between(np.arange(horizon),
                                   np.mean(errors[:, :, 1], axis=0) - 2 * np.std(errors[:, :, 1], axis=0),
                                   np.mean(errors[:, :, 1], axis=0) + 2 * np.std(errors[:, :, 1], axis=0),
                                   color="blue", alpha=0.1)
                ax[0].fill_between(np.arange(horizon),
                                   np.mean(errors[:, :, 0], axis=0) - np.std(errors[:, :, 0], axis=0),
                                   np.mean(errors[:, :, 0], axis=0) + np.std(errors[:, :, 0], axis=0),
                                   color="blue", alpha=0.1)
                ax[1].fill_between(np.arange(horizon),
                                   np.mean(errors[:, :, 1], axis=0) - np.std(errors[:, :, 1], axis=0),
                                   np.mean(errors[:, :, 1], axis=0) + np.std(errors[:, :, 1], axis=0),
                                   color="blue", alpha=0.1)
                ax[0].plot(np.arange(horizon), np.mean(errors[:, :, 0], axis=0), color="blue", alpha=1, linewidth=2)
                ax[1].plot(np.arange(horizon), np.mean(errors[:, :, 1], axis=0), color="blue", alpha=1, linewidth=2)
                ax[0].set_title("Temperature errors", size=22)
                ax[0].set_ylabel("Degrees", size=20)
                ax[1].set_title("Power errors", size=22)
                ax[1].set_ylabel("Power", size=20)

                _save_or_show(save=save, save_name=f"{self.name}_Error_Gaussian", **kwargs)

    def plot_consistency(self, sequences, to_open: list = None, **kwargs):
        """
        Function to analyze the physical consistency of the model: plots:
         - the predicted temperatures and power given the observed patterns of valves opening and closing
         - the predicted values in the case where the valves are kept open along the prediction horizon
         - the predicted values in the case where the valves are kept closed along the prediction horizon

         Args:
             sequences:  the sequences of inputs to analyze
             to_open:    Which vavles to open, i.e. which room to heat/cool
        """

        # Ensure the given sequences are a list of list, not only one list
        if type(sequences) == tuple:
            sequences = [sequences]

        # If none is given, just open all the valves
        if to_open is None:
            to_open = np.arange(len(self.rooms))

        # Recall te true opening and closing pattern
        temp = self.X[:, self.valve_column[to_open]].copy()
        close = np.nanmin(self.X[:, self.valve_column[to_open]])
        open_ = np.nanmax(self.X[:, self.valve_column[to_open]])

        # Get predictions in the 3 cases
        original_predictions, _ = self.scale_back_predictions(sequences)

        self.X[:, self.valve_column[to_open]] = close
        close_predictions, _ = self.scale_back_predictions(sequences)

        self.X[:, self.valve_column[to_open]] = open_
        open_predictions, _ = self.scale_back_predictions(sequences)

        # Put the original valve pattern back
        self.X[:, self.valve_column] = temp

        for i, sequence in enumerate(sequences):

            # Design the plot with custom helpers
            fig, ax = _plot_helpers(subplots=[len(self.rooms), 2], sharex=True, sharey=False, **kwargs)
            index = self.dataset.data.index[sequence[0] + self.warm_start_length:sequence[1]]

            # Plot both informations
            for j in range(len(self.rooms)):
                # Plot both informations
                ax[j, 0].plot(index, close_predictions[i, :sequence[1] - sequence[0] - self.warm_start_length, j],
                              label="Close prediction")
                ax[j, 0].plot(index, open_predictions[i, :sequence[1] - sequence[0] - self.warm_start_length, j],
                              label="Open prediction")
                ax[j, 0].plot(index, original_predictions[i, :sequence[1] - sequence[0] - self.warm_start_length, j],
                              label="Original prediction")

                ax[j, 1].plot(index, close_predictions[i, :sequence[1] - sequence[0] - self.warm_start_length,
                                     len(self.rooms) + j],
                              label="Close prediction")
                ax[j, 1].plot(index, open_predictions[i, :sequence[1] - sequence[0] - self.warm_start_length,
                                     len(self.rooms) + j],
                              label="Open prediction")
                ax[j, 1].plot(index, original_predictions[i, :sequence[1] - sequence[0] - self.warm_start_length,
                                     len(self.rooms) + j],
                              label="Original prediction")

                ax[j, 0].set_title(f"Temperature room {self.rooms[j]}", size=22)
                ax[j, 1].set_title(f"Power room {self.rooms[j]}", size=22)

            ax[-1, 0].set_xlabel("Time", size=20)
            ax[-1, 1].set_xlabel("Time", size=20)

            fig.autofmt_xdate()
            _save_or_show(legend=True, **kwargs)

    def compute_loss(self, sequences: list):
        """
        Custom function to compute the loss of a batch of sequences.

        Args:
            sequences: The sequences in the batch

        Returns:
            The loss
        """

        predictions, batch_y = self.predict(sequences=sequences)

        if not self.predict_power:
            if self.module == "PCNNTestQuantiles":
                err = 0
                for i in range(len(self.quantiles)):
                    e = predictions[:, :, 0, :, i] - batch_y[:, :, 0, :]
                    err += torch.max(self.quantiles[i] * e, (self.quantiles[i] - 1) * e).mean()
                return err / len(self.quantiles)

            else:
                return self.loss(predictions, batch_y)
        else:
            pred = torch.zeros((predictions.shape[0], predictions.shape[1], len(self.rooms) + 1)).to(self.device)
            y = torch.zeros((predictions.shape[0], predictions.shape[1], len(self.rooms) + 1)).to(self.device)
            pred[:, :, :-1] = predictions[:, :, 0, :]
            pred[:, :, -1] = torch.sum(predictions[:, :, 1, :], axis=-1)
            y[:, :, :-1] = batch_y[:, :, :len(self.rooms)]
            y[:, :, -1] = (batch_y[:, :, -1] - torch.from_numpy(np.array([self.total_zero_power])).type(
                    torch.FloatTensor).to(self.device))

            return self.loss(pred[:, :, -1], y[:, :, -1])
            #return self.loss(pred, y)

    def save(self, name_to_add: str = None, verbose: int = 0):
        """
        Function to save a PyTorch model: Save the state of all parameters, as well as the one of the
        optimizer. We also recall the losses for ease of analysis.

        Args:
            name_to_add:    Something to save a unique model

        Returns
            Nothing, everything is done in place and stored in the parameters
        """

        if verbose > 0:
            print(f"\nSaving the {name_to_add} model...")

        if name_to_add is not None:
            save_name = os.path.join(self.save_name, f"{name_to_add}_model.pt")
        else:
            save_name = os.path.join(self.save_name, "model.pt")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_sequences": self.train_sequences,
                "validation_sequences": self.validation_sequences,
                "test_sequences": self.test_sequences,
                "train_losses": self.train_losses,
                "validation_losses": self.validation_losses,
                "_validation_losses": self._validation_losses,
                "test_losses": self.test_losses,
                "times": self.times,
                "discount_factors_heating": self.discount_factors_heating,
                "discount_factors_cooling": self.discount_factors_cooling,
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "d": self.d,
                "predict_differences": self.predict_differences,
                "warm_start_length": self.warm_start_length,
                "maximum_sequence_length": self.maximum_sequence_length,
                "feed_input_through_nn": self.feed_input_through_nn,
                "input_nn_hidden_sizes": self.input_nn_hidden_sizes,
                "lstm_hidden_size": self.lstm_hidden_size,
                "lstm_num_layers": self.lstm_num_layers,
                "feed_output_through_nn": self.feed_output_through_nn,
                "output_nn_hidden_sizes": self.output_nn_hidden_sizes,
            },
            save_name,
        )

    def load(self, load_last: bool = False):
        """
        Function trying to load an existing model, by default the best one if it exists. But for training purposes,
        it is possible to load the last state of the model instead.

        Args:
            load_last:  Flag to set to True if the last model checkpoint is wanted, instead of the best one

        Returns:
             Nothing, everything is done in place and stored in the parameters.
        """

        if load_last:
            save_name = os.path.join(self.save_name, "last_model.pt")
        else:
            save_name = os.path.join(self.save_name, "best_model.pt")

        print("\nTrying to load a trained model...")
        try:
            # Build the full path to the model and check its existence

            assert os.path.exists(save_name), f"The file {save_name} doesn't exist."

            # Load the checkpoint
            checkpoint = torch.load(save_name, map_location=lambda storage, loc: storage)

            # Put it into the model
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if torch.cuda.is_available():
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            self.train_sequences = checkpoint["train_sequences"]
            self.validation_sequences = checkpoint["validation_sequences"]
            self.test_sequences = checkpoint["test_sequences"]
            self.train_losses = checkpoint["train_losses"]
            self.validation_losses = checkpoint["validation_losses"]
            #self._validation_losses = checkpoint["_validation_losses"]
            self.test_losses = checkpoint["test_losses"]
            self.times = checkpoint["times"]
            self.discount_factors_heating = checkpoint["discount_factors_heating"]
            self.discount_factors_cooling = checkpoint["discount_factors_cooling"]
            self.a = checkpoint["a"]
            self.b = checkpoint["b"]
            self.c = checkpoint["c"]
            self.d = checkpoint["d"]
            self.predict_differences = checkpoint["predict_differences"]
            self.warm_start_length = checkpoint["warm_start_length"]
            self.maximum_sequence_length = checkpoint["maximum_sequence_length"]
            self.feed_input_through_nn = checkpoint["feed_input_through_nn"]
            self.input_nn_hidden_sizes = checkpoint["input_nn_hidden_sizes"]
            self.lstm_hidden_size = checkpoint["lstm_hidden_size"]
            self.lstm_num_layers = checkpoint["lstm_num_layers"]
            self.feed_output_through_nn = checkpoint["feed_output_through_nn"]
            self.output_nn_hidden_sizes = checkpoint["output_nn_hidden_sizes"]

            # Print the current status of the found model
            if self.verbose > 0:
                print(f"Found!\nThe model has been fitted for {len(self.train_losses)} epochs already, "
                      f"with loss {np.min(self.validation_losses): .5f}.")
                print(f"It contains {len(self.train_sequences)} training sequences and "
                      f"{len(self.validation_sequences)} validation sequences.\n")

            # Plot the losses if wanted
            if self.verbose > 1:
                self.plot_losses()

        # Otherwise, keep an empty model, return nothing
        except AssertionError:
            print("No existing model was found!")


class PiNN(Model):
    def __init__(self, data_kwargs: dict, model_kwargs: dict, Y_columns: list, X_columns: list = None,
                 base_indices: list = None, effect_indices: list = None, topology: dict = None, load_last: bool = False,
                 load: bool = True, _lambda: float = 100):
        super().__init__(data_kwargs=data_kwargs, model_kwargs=model_kwargs, Y_columns=Y_columns, X_columns=X_columns,
                         base_indices=base_indices, effect_indices=effect_indices, topology=topology,
                         load_last=load_last, load=load)
        self._lambda = _lambda

    def compute_loss(self, sequences: list):

        data = self.build_input_output_from_sequences(sequences)
        data[0][:, :, self.valve_column] = 0.9
        data[0].requires_grad = True
        predictions, _ = self.predict(data=data)
        grads = torch.autograd.grad(predictions[:, -1, 0, :], data[0].to(self.device),
                                    grad_outputs=torch.ones((predictions.shape[0], predictions.shape[-1])).to(self.device),
                                    create_graph=False, retain_graph=False, allow_unused=True)[0]

        predictions, batch_y = self.predict(sequences=sequences)

        return self.loss(predictions, batch_y) + self._lambda * torch.relu(
            -grads[:, self.warm_start_length:, self.power_column + [self.out_column]]).mean()

    def _compute_loss(self, sequences: list):
        predictions, batch_y = self.predict(sequences=sequences)
        return self.loss(predictions, batch_y)

    def load(self, load_last: bool = False):
        """
        Function trying to load an existing model, by default the best one if it exists. But for training purposes,
        it is possible to load the last state of the model instead.

        Args:
            load_last:  Flag to set to True if the last model checkpoint is wanted, instead of the best one

        Returns:
             Nothing, everything is done in place and stored in the parameters.
        """

        if load_last:
            save_name = os.path.join(self.save_name, "last_model.pt")
        else:
            save_name = os.path.join(self.save_name, "best_model.pt")

        print("\nTrying to load a trained model...")
        try:
            # Build the full path to the model and check its existence

            assert os.path.exists(save_name), f"The file {save_name} doesn't exist."

            # Load the checkpoint
            checkpoint = torch.load(save_name, map_location=lambda storage, loc: storage)

            # Put it into the model
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if torch.cuda.is_available():
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            self.train_sequences = checkpoint["train_sequences"]
            self.validation_sequences = checkpoint["validation_sequences"]
            self.test_sequences = checkpoint["test_sequences"]
            self.train_losses = checkpoint["train_losses"]
            self.validation_losses = checkpoint["validation_losses"]
            #self._validation_losses = checkpoint["_validation_losses"]
            self.test_losses = checkpoint["test_losses"]
            self.times = checkpoint["times"]
            self.discount_factors_heating = checkpoint["discount_factors_heating"]
            self.discount_factors_cooling = checkpoint["discount_factors_cooling"]
            self.a = checkpoint["a"]
            self.b = checkpoint["b"]
            self.c = checkpoint["c"]
            self.d = checkpoint["d"]
            self.predict_differences = checkpoint["predict_differences"]
            self.warm_start_length = checkpoint["warm_start_length"]
            self.maximum_sequence_length = checkpoint["maximum_sequence_length"]
            self.feed_input_through_nn = checkpoint["feed_input_through_nn"]
            self.input_nn_hidden_sizes = checkpoint["input_nn_hidden_sizes"]
            self.lstm_hidden_size = checkpoint["lstm_hidden_size"]
            self.lstm_num_layers = checkpoint["lstm_num_layers"]
            self.feed_output_through_nn = checkpoint["feed_output_through_nn"]
            self.output_nn_hidden_sizes = checkpoint["output_nn_hidden_sizes"]

            # Print the current status of the found model
            if self.verbose > 0:
                print(f"Found!\nThe model has been fitted for {len(self.train_losses)} epochs already, "
                      f"with loss {np.min(self.validation_losses): .5f}.")
                print(f"It contains {len(self.train_sequences)} training sequences and "
                      f"{len(self.validation_sequences)} validation sequences.\n")

            # Plot the losses if wanted
            if self.verbose > 1:
                self.plot_losses()

        # Otherwise, keep an empty model, return nothing
        except AssertionError:
            print("No existing model was found!")


class ResidualModel(Model):
    def __init__(self, rc_model, data_kwargs: dict, model_kwargs: dict, Y_columns: list, X_columns: list = None,
                 base_indices: list = None, effect_indices: list = None, topology: dict = None, load_last: bool = False,
                 load: bool = True):
        super().__init__(data_kwargs=data_kwargs, model_kwargs=model_kwargs, Y_columns=Y_columns, X_columns=X_columns,
                         base_indices=base_indices, effect_indices=effect_indices, topology=topology,
                         load_last=load_last, load=load)

        self.rc_model = rc_model

        # RC model predictions
        try:
            with open(os.path.join('..', 'saves', 'Models', self.name+'Residuals.pkl'), 'rb') as f:
                residual_parameters = pickle.load(f)
                self.residual_min = residual_parameters[0]
                self.residual_max = residual_parameters[1]
                self.residual_zero = residual_parameters[2]

        except FileNotFoundError:
            starts = [seq[0] + self.warm_start_length for seq in self.train_sequences + self.validation_sequences]
            ends = [seq[1] + 1 for seq in self.train_sequences + self.validation_sequences]
            rc_predictions, rc_ys = rc_model.predict(starts, ends)

            # Prepare the rescaling coefficients
            residuals = rc_ys[:, :,
                        [i for i, x in enumerate(rc_model.data.columns) if ('measurement' in x)]] - rc_predictions
            self.residual_min = torch.FloatTensor(residuals.min(axis=0).min(axis=0))
            self.residual_max = torch.FloatTensor(residuals.max(axis=0).max(axis=0))
            self.residual_zero = (- self.residual_min) / (self.residual_max - self.residual_min) * 0.8 + 0.1

            with open(os.path.join('..', 'saves', 'Models', self.name+'Residuals.pkl'), 'wb') as f:
                pickle.dump([self.residual_min, self.residual_max, self.residual_zero], f)

        self.rc_predictions = None

    def build_input_output_from_sequences(self, sequences: list):
        """
        Input and output generator from given sequences of indices corresponding to a batch.

        Args:
            sequences: sequences of the batch to prepare

        Returns:
            batch_x:    Batch input of the model
            batch_y:    Targets of the model, the temperature and the power
        """

        # Ensure the given sequences are a list of list, not only one list
        if type(sequences) == tuple:
            sequences = [sequences]

        # Iterate over the sequences to build the input in the right form
        input_tensor_list = [torch.FloatTensor(self.X[sequence[0]: sequence[1], :].copy()) for sequence in sequences]

        # Prepare the output for the temperature and power consumption
        if self.predict_differences:
            output_tensor_list = [torch.FloatTensor(self.differences_Y[sequence[0]: sequence[1], :].copy()) for sequence
                                  in sequences]
        else:
            output_tensor_list = [torch.FloatTensor(self.Y[sequence[0]: sequence[1], :].copy()) for sequence in
                                  sequences]

        # Build the final results by taking care of the batch_size=1 case
        if len(sequences) > 1:
            batch_x = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
            if not self.predict_power:
                batch_y = pad_sequence(output_tensor_list, batch_first=True, padding_value=0).reshape(len(sequences),
                                                                                                      batch_x.shape[1],
                                                                                                      2,
                                                                                                      len(self.rooms))
            else:
                batch_y = pad_sequence(output_tensor_list, batch_first=True, padding_value=0)
        else:
            batch_x = input_tensor_list[0].view(1, input_tensor_list[0].shape[0], -1)
            if not self.predict_power:
                batch_y = output_tensor_list[0].view(1, output_tensor_list[0].shape[0], 2, len(self.rooms))
            else:
                batch_y = output_tensor_list[0].view(1, output_tensor_list[0].shape[0], -1)

        # RC model predictions
        starts = [seq[0] + self.warm_start_length for seq in sequences]
        ends = [seq[1] + 1 for seq in sequences]
        rc_predictions, rc_ys = self.rc_model.predict(starts, ends)

        # Keep the RC predictions in memory to rebuild them later in full_scale_back_predictions
        self.rc_predictions = rc_predictions.copy()
        self.true_y = batch_y.clone()

        # Modify to predict residuals with the network
        residuals = torch.FloatTensor(
            rc_ys[:, :, [i for i, x in enumerate(self.rc_model.data.columns) if ('measurement' in x)]] - rc_predictions)
        batch_y[:, :self.warm_start_length, 0, :] = self.residual_zero
        batch_y[:, self.warm_start_length:, 0, :] = (residuals - self.residual_min) / (
                    self.residual_max - self.residual_min) * 0.8 + 0.1

        # Modify x to ensure the warm start works (must predict zero in the first steps)
        batch_x[:, :self.warm_start_length, self.temperature_column] = self.residual_zero

        # Return everything
        return batch_x.to(self.device), batch_y.to(self.device)

    def full_scale_back_predictions(self, sequences: Union[list, int] = None, data: torch.FloatTensor = None):

        predictions, batch_y = self.predict(sequences=sequences, data=data, mpc_mode=False)
        residuals = (predictions[:, self.warm_start_length:, 0, :] - .1) / 0.8 * (
                    self.residual_max - self.residual_min) + self.residual_min
        predictions[:, self.warm_start_length:, 0, :] = torch.FloatTensor(self.rc_predictions) + residuals

        self.true_y[:, :, 0, 0] = (self.true_y[:, :, 0, 0] - 0.1) / 0.8 * (self.dataset.max_ - self.dataset.min_)[
            'Thermal temperature measurement 272'] \
                                  + self.dataset.min_['Thermal temperature measurement 272']
        self.true_y[:, :, 0, 1] = (self.true_y[:, :, 0, 1] - 0.1) / 0.8 * (self.dataset.max_ - self.dataset.min_)[
            'Thermal temperature measurement 273'] \
                                  + self.dataset.min_['Thermal temperature measurement 273']
        self.true_y[:, :, 0, 2] = (self.true_y[:, :, 0, 2] - 0.1) / 0.8 * (self.dataset.max_ - self.dataset.min_)[
            'Thermal temperature measurement 274'] \
                                  + self.dataset.min_['Thermal temperature measurement 274']

        return predictions[:, self.warm_start_length:, 0, :].detach().numpy(), self.true_y[:, self.warm_start_length:,
                                                                               0,
                                                                               :].detach().numpy(), self.rc_predictions

