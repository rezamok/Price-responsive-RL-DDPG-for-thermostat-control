"""
File containing PyTorch modules used by the models
"""
import torch
from torch import nn
import numpy as np


class PositiveLinear(nn.Module):
    """
    https://discuss.pytorch.org/t/positive-weights/19701/7
    """
    def __init__(self, in_features, out_features, require_bias=True):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.require_bias = require_bias
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.require_bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.log_weight, 0.)

    def forward(self, input):
        if self.require_bias:
            return nn.functional.linear(input, self.log_weight.exp()) + self.bias
        else:
            return nn.functional.linear(input, self.log_weight.exp())


class PCNNTestQuantiles(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            effect_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
            topology: dict,
            predict_power: bool,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the unforce dynamics (D)
            effect_indices:             Indices for the energy computation in E
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
            topology:                   Topology of the building to model
            predict_power:              Whether to use provided power measurements or infer them from data
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.topology = topology
        self.predict_power = predict_power

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]
        self.high_min = normalization_variables['High'][0]
        self.high_diff = normalization_variables['High'][1]
        self.low_min = normalization_variables['Low'][0]
        self.low_diff = normalization_variables['Low'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a']
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d']

        self.autoregression = 2
        self.quantiles = [0.1, 0.5, 0.9]
        self.quantile_factors = [-0.1, 0.0, .1]

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.ModuleList([nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))]) for i in range(len(self.quantiles))])
            self.b = nn.ModuleList([nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Outside']))]) for i in range(len(self.quantiles))])
            self.c = nn.ModuleList([nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Neighbors']))]) for i in range(len(self.quantiles))])
            self.d = nn.ModuleList([nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))]) for i in range(len(self.quantiles))])

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0]))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0]))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.base_indices)] + self.input_nn_hidden_sizes[0]
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[0][-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size[0],
                                 num_layers=self.lstm_num_layers[0], batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size[0])

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size[0]] + self.output_nn_hidden_sizes[0] + [len(self.topology['Rooms']) * len(self.quantiles)]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())
                                                 for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                if 'log_weight' in name:
                    nn.init.normal_(param, -3.0, std=0.5)
                for x in ['a', 'b', 'c', 'd']:
                    for q in range(len(self.quantiles)):
                        if f'{x}.{q}' in name:
                            nn.init.constant_(param, self.quantile_factors[q])

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False,
                mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device)

            heating_power_h = None
            heating_power_c = None
            cooling_power_h = None
            cooling_power_c = None

            self.init_temp = x[:, 0, self.temperature_column]

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros((x.shape[0], len(self.topology['Rooms']), len(self.quantiles))).to(self.device)  ## D
            self.last_effect = torch.zeros((x.shape[0], len(self.topology['Rooms']), len(self.quantiles))).to(self.device)  ## E

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            temperatures = self.last_base
        else:
            temperatures = torch.zeros_like(self.last_effect).to(self.device)
            for j in range(len(self.quantiles)):
                temperatures[:, :, j] = x[:, -1, self.temperature_column].clone() - self.last_effect[:, :, j]
                

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, len(self.topology['Rooms']), len(self.quantiles)).to(self.device)  ## D
        effect = torch.zeros(x.shape[0], 2, len(self.topology['Rooms']), len(self.quantiles)).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        for i in range(len(self.quantiles)):
            base[:, 1, :, i] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        effect[:, 0, :, :] = self.last_effect.clone()

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[0][-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))

        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0, :, :] = temp.reshape(-1, base.shape[2], base.shape[3]).squeeze() / self.base_division_factor[0] + temperatures
        else:
            # Store the outputs
            base[:, 0, :, :] = lstm_output[:, -1, :].reshape(-1, base.shape[2], base.shape[3]) / self.base_division_factor[0] + temperatures

        ## Heat losses computation in 'E'
        # Loss to the outside is b*(T_k-T^out_k)
        for j, q in enumerate(self.b):
            for i, room in enumerate(self.topology['Outside']):
                effect[:, 0, room, j] = effect[:, 0, room, j].clone() - q[i](
                    (((temperatures[:, room, j].squeeze()
                       + self.last_effect[:, room, j].clone() - 0.1) / 0.8
                      * self.room_diff[room] + self.room_min[room])
                     - ((x[:, -1, self.out_column] - 0.1) / 0.8
                        * self.out_diff + self.out_min)).
                        reshape(-1, 1)).squeeze() / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for j, q in enumerate(self.c):
            for i, (rooma, roomb) in enumerate(self.topology['Neighbors']):
                for room1, room2 in zip([rooma, roomb], [roomb, rooma]):
                    effect[:, 0, room1, j] = effect[:, 0, room1, j].clone() - q[i](
                        (((temperatures[:, room1, j].squeeze()
                           + self.last_effect[:, room1, j].clone() - 0.1) / 0.8
                          * self.room_diff[room1] + self.room_min[room1])
                         - ((temperatures[:, room2, j]
                             + self.last_effect[:, room2, j].clone() - 0.1) / 0.8
                            * self.room_diff[room2] + self.room_min[room2])).
                            reshape(-1, 1)).squeeze() / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column].mean(axis=-1), axis=1).values > 0.1001)[0]
        if len(mask) > 0:

            # Find heating and cooling sequences
            heating = x[mask, 0, self.case_column] > 0.5
            cooling = x[mask, 0, self.case_column] < 0.5

            temp = x[mask, -1, :].clone()
            # Substract the 'zero power' to get negative values for cooling power
            for j in range(len(self.quantiles)):
                effect[mask, 1, :, j] = temp[:, self.power_column] - torch.from_numpy(self.zero_power).type(
                    torch.FloatTensor).to(self.device)

            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                for j, q in enumerate(self.a):
                    for i in range(len(self.topology['Rooms'])):
                        effect[mask[heating], 0, i, j] = effect[mask[heating], 0, i, j].clone() \
                                                      + q[i](
                            effect[mask[heating], 1, i, j].clone().reshape(-1, 1)).squeeze() \
                                                      / self.a_scaling[i]

            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                for j, q in enumerate(self.d):
                    for i in range(len(self.topology['Rooms'])):
                        effect[mask[cooling], 0, i, j] = effect[mask[cooling], 0, i, j].clone() \
                                                      + q[i](
                            effect[mask[cooling], 1, i, j].clone().reshape(-1, 1)).squeeze() \
                                                      / self.d_scaling[i]

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, :, :].clone()
        self.last_effect = effect[:, 0, :, :].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        else:
            output = torch.zeros(x.shape[0], 2, effect.shape[-1])
            output[:, 0, :] = base[:, 0, :]
            output[:, 1, :] = effect[:, 0, :]

        # Return the predictions and states of the model
        return output, (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c)


class PCNN_full_shared_mul(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            effect_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
            topology: dict,
            predict_power: bool,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the unforce dynamics (D)
            effect_indices:             Indices for the energy computation in E
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
            topology:                   Topology of the building to model
            predict_power:              Whether to use provided power measurements or infer them from data
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.topology = topology
        self.predict_power = predict_power

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]
        self.high_min = normalization_variables['High'][0]
        self.high_diff = normalization_variables['High'][1]
        self.low_min = normalization_variables['Low'][0]
        self.low_diff = normalization_variables['Low'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a']
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d']

        self.autoregression = 2

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))])
            self.b = nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Outside']))])
            self.c = nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Neighbors']))])
            self.d = nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))])

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0]))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0]))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.base_indices)] + self.input_nn_hidden_sizes[0]
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[0][-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size[0],
                                 num_layers=self.lstm_num_layers[0], batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size[0])

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size[0]] + self.output_nn_hidden_sizes[0] + [len(self.topology['Rooms'])]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())
                                                 for i in range(0, len(sizes) - 1)])

        ## Predict the power
        if self.predict_power:
            # Process the input by a NN if wanted
            size = [self.autoregression + 3] + self.input_nn_hidden_sizes[1]
            self.heating_power_input_nn = nn.ModuleList(nn.ModuleList([nn.Sequential(PositiveLinear(size[i], size[i + 1]), nn.ReLU())
                                                         for i in range(0, len(size) - 1)]) for _ in range(len(self.topology['Rooms'])))
            self.cooling_power_input_nn = nn.ModuleList(nn.ModuleList([nn.Sequential(PositiveLinear(size[i], size[i + 1]), nn.ReLU())
                                                         for i in range(0, len(size) - 1)]) for _ in range(len(self.topology['Rooms'])))

            # Create the NNs to process the output of the LSTMs for each modules if wanted
            sizes = [self.input_nn_hidden_sizes[1][-1]] + self.output_nn_hidden_sizes[1] + [1]
            self.heating_power_output_nn = nn.ModuleList(nn.ModuleList(
                [nn.Sequential(PositiveLinear(sizes[i], sizes[i + 1]), nn.ReLU())
                 for i in range(0, len(sizes) - 1)]) for _ in range(len(self.topology['Rooms'])))
            self.cooling_power_output_nn = nn.ModuleList(nn.ModuleList(
                [nn.Sequential(PositiveLinear(sizes[i], sizes[i + 1]), nn.ReLU())
                 for i in range(0, len(sizes) - 1)]) for _ in range(len(self.topology['Rooms'])))

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                if 'log_weight' in name:
                    nn.init.normal_(param, -3.0, std=0.5)
                if 'a.' in name:
                    nn.init.constant_(param, 0.0)
                if 'b.' in name:
                    nn.init.constant_(param, 0.0)
                if 'd.' in name:
                    nn.init.constant_(param, 0.0)
                if 'c.' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False,
                mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device)

            heating_power_h = None
            heating_power_c = None
            cooling_power_h = None
            cooling_power_c = None

            self.past_inputs = torch.zeros((x.shape[0], len(self.topology['Rooms']), self.autoregression+1)).to(
                self.device)
            self.init_temp = x[:, 0, self.temperature_column]

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## D
            self.last_effect = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## E

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_effect
        self.past_inputs = torch.cat([self.past_inputs[:, :, 1:],
                                      x[:, -1, self.valve_column].unsqueeze(2)], dim=2)

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## D
        effect = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        if not self.predict_power:
            base[:, 1, :] = torch.from_numpy(self.zero_power).to(self.device)
            base = base.to(self.device)
        effect[:, 0, :] = self.last_effect.clone()

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[0][-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))

        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0, :] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]
        else:
            # Store the outputs
            base[:, 0, :] = lstm_output[:, -1, :] / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        ## Heat losses computation in 'E'
        # Loss to the outside is b*(T_k-T^out_k)
        for i, room in enumerate(self.topology['Outside']):
            effect[:, 0, room] = effect[:, 0, room].clone() - self.b[i](
                (((x[:, -1, self.temperature_column[room]].squeeze()
                   + self.last_effect[:, room].clone() - 0.1) / 0.8
                  * self.room_diff[room] + self.room_min[room])
                 - ((x[:, -1, self.out_column] - 0.1) / 0.8
                    * self.out_diff + self.out_min)).
                    reshape(-1, 1)).squeeze() / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for i, (rooma, roomb) in enumerate(self.topology['Neighbors']):
            for room1, room2 in zip([rooma, roomb], [roomb, rooma]):
                effect[:, 0, room1] = effect[:, 0, room1].clone() - self.c[i](
                    (((x[:, -1, self.temperature_column[room1]].squeeze()
                       + self.last_effect[:, room1].clone() - 0.1) / 0.8
                      * self.room_diff[room1] + self.room_min[room1])
                     - ((x[:, -1, self.temperature_column[room2]]
                         + self.last_effect[:, room2].clone() - 0.1) / 0.8
                        * self.room_diff[room2] + self.room_min[room2])).
                        reshape(-1, 1)).squeeze() / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column].mean(axis=-1), axis=1).values > 0.1001)[0]
        if len(mask) > 0:

            # Find heating and cooling sequences
            heating = x[mask, 0, self.case_column] > 0.5
            cooling = x[mask, 0, self.case_column] < 0.5

            if not self.predict_power:
                temp = x[mask, -1, :].clone()
                # Substract the 'zero power' to get negative values for cooling power
                effect[mask, 1, :] = temp[:, self.power_column] - torch.from_numpy(self.zero_power).type(
                    torch.FloatTensor).to(self.device)

            else:
                # Input embedding when wanted
                if sum(heating) > 0:
                    heating_power_embedding = torch.zeros(len(mask[heating]), len(self.topology['Rooms']),
                                                          self.input_nn_hidden_sizes[1][-1]).to(self.device)
                    masked = self.past_inputs[mask[heating], :, :].clone()
                    for i, heating_power_input_nn in enumerate(self.heating_power_input_nn):
                        # temp = masked[:, :-len(self.topology['Rooms'])]
                        temp = torch.cat([masked[:, i, :], 0.9 - self.init_temp[mask[heating], i].unsqueeze(1),
                                          x[mask[heating], -1, self.effect_indices[0]].unsqueeze(1)], dim=1)
                        for layer in heating_power_input_nn:
                            temp = layer(temp)
                        heating_power_embedding[:, i, :] = temp.clone()

                if sum(cooling) > 0:
                    cooling_power_embedding = torch.zeros(len(mask[heating]), len(self.topology['Rooms']),
                                                          self.input_nn_hidden_sizes[1][-1]).to(self.device)
                    masked = self.past_inputs[mask[cooling], :, :].clone()
                    # temp = masked[:, :-len(self.topology['Rooms'])]
                    for i, cooling_power_input_nn in enumerate(self.cooling_power_input_nn):
                        temp = torch.cat([masked[:, i, :], self.init_temp[mask[cooling], i].unsqueeze(1),
                                          0.9 - x[mask[cooling], -1, self.effect_indices[1]].unsqueeze(1)], dim=1)
                        for layer in cooling_power_input_nn:
                            temp = layer(temp)
                        cooling_power_embedding[:, i, :] = temp.clone()

                # Some manipulations are needed to feed the output through the neural network if wanted
                if sum(heating) > 0:
                    # Go through the output layer of the NN
                    for i, heating_power_output_nn in enumerate(self.heating_power_output_nn):
                        temp = heating_power_embedding[:, i, :].clone()
                        for layer in heating_power_output_nn:
                            temp = layer(temp)
                        heat = x[mask[heating], :, :].clone()
                        effect[mask[heating], 1, i] = temp.squeeze() * (heat[:, -1, self.valve_column[i]] - 0.1) / len(
                            self.topology['Rooms'])  # / 0.8

                if sum(cooling) > 0:
                    for i, cooling_power_output_nn in enumerate(self.cooling_power_output_nn):
                        temp = cooling_power_embedding[:, i, :].clone()
                        for layer in cooling_power_output_nn:
                            temp = layer(temp)
                        cool = x[mask[cooling], :, :].clone()
                        effect[mask[cooling], 1, :] = - temp.squeeze() * (cool[:, -1, self.valve_column[i]] - 0.1) / len(
                            self.topology['Rooms'])  # / 0.8

            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[heating], 0, i] = effect[mask[heating], 0, i].clone() \
                                                  + self.a[i](
                        effect[mask[heating], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.a_scaling[i]

            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[cooling], 0, i] = effect[mask[cooling], 0, i].clone() \
                                                  + self.d[i](
                        effect[mask[cooling], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.d_scaling[i]

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, :].clone()
        self.last_effect = effect[:, 0, :].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        else:
            output = torch.zeros(x.shape[0], 2, effect.shape[-1])
            output[:, 0, :] = base[:, 0, :]
            output[:, 1, :] = effect[:, 0, :]

        # Return the predictions and states of the model
        return output, (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c)


class PCNN_full(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            effect_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
            topology: dict,
            predict_power: bool,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the unforce dynamics (D)
            effect_indices:             Indices for the energy computation in E
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
            topology:                   Topology of the building to model
            predict_power:              Whether to use provided power measurements or infer them from data
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.topology = topology
        self.predict_power = predict_power

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a']
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d']

        self.autoregression = 2

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))])
            self.b = nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Outside']))])
            self.c = nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Neighbors']))])
            self.d = nn.ModuleList(
                [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))])

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.ParameterList(
                [nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0])) for _ in
                 range(len(self.topology['Rooms']))])
            self.initial_base_c = nn.ParameterList(
                [nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0])) for _ in
                 range(len(self.topology['Rooms']))])

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            sizes = [[len(base_indices)] + self.input_nn_hidden_sizes[0] for base_indices in self.base_indices]
            self.base_input_nn = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                               for i in range(0, len(size) - 1)]) for size in sizes])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = [self.input_nn_hidden_sizes[0][-1] if self.feed_input_through_nn else len(base_indices) for
                           base_indices in self.base_indices]
        self.base_lstm = nn.ModuleList([nn.LSTM(input_size=lstm_input_size[i], hidden_size=self.lstm_hidden_size[0],
                                                num_layers=self.lstm_num_layers[0], batch_first=True) for i in
                                        range(len(self.topology['Rooms']))])

        if self.layer_norm:
            self.base_norm = nn.ModuleList(
                [nn.LayerNorm(normalized_shape=self.lstm_hidden_size[0]) for _ in range(len(self.topology['Rooms']))])

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size[0]] + self.output_nn_hidden_sizes[0] + [1]
            self.base_output_nn = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())
                                                                for i in range(0, len(sizes) - 1)]) for _ in
                                                 range(len(self.topology['Rooms']))])

        ## Predict the power
        if self.predict_power:
            # Process the input by a NN if wanted
            size = [self.autoregression + 3] + self.input_nn_hidden_sizes[1]
            self.heating_power_input_nn = nn.ModuleList(nn.ModuleList([nn.Sequential(PositiveLinear(size[i], size[i + 1]), nn.ReLU())
                                                         for i in range(0, len(size) - 1)]) for _ in range(len(self.topology['Rooms'])))
            self.cooling_power_input_nn = nn.ModuleList(nn.ModuleList([nn.Sequential(PositiveLinear(size[i], size[i + 1]), nn.ReLU())
                                                         for i in range(0, len(size) - 1)]) for _ in range(len(self.topology['Rooms'])))

            # Create the NNs to process the output of the LSTMs for each modules if wanted
            sizes = [self.input_nn_hidden_sizes[1][-1]] + self.output_nn_hidden_sizes[1] + [1]
            self.heating_power_output_nn = nn.ModuleList(nn.ModuleList(
                [nn.Sequential(PositiveLinear(sizes[i], sizes[i + 1]), nn.ReLU())
                 for i in range(0, len(sizes) - 1)]) for _ in range(len(self.topology['Rooms'])))
            self.cooling_power_output_nn = nn.ModuleList(nn.ModuleList(
                [nn.Sequential(PositiveLinear(sizes[i], sizes[i + 1]), nn.ReLU())
                 for i in range(0, len(sizes) - 1)]) for _ in range(len(self.topology['Rooms'])))

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                if 'log_weight' in name:
                    nn.init.normal_(param, -3.0, std=0.5)
                if 'a.' in name:
                    nn.init.constant_(param, 0.0)
                if 'b.' in name:
                    nn.init.constant_(param, 0.0)
                if 'd.' in name:
                    nn.init.constant_(param, 0.0)
                if 'c.' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False,
                mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = [torch.stack([initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device) for
                          initial_base_h in self.initial_base_h]
                base_c = [torch.stack([initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device) for
                          initial_base_c in self.initial_base_c]
            else:
                base_h = [torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device) for _
                          in range(len(self.topology['Rooms']))]
                base_c = [torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device) for _
                          in range(len(self.topology['Rooms']))]

            heating_power_h = None
            heating_power_c = None
            cooling_power_h = None
            cooling_power_c = None

            self.past_inputs = torch.zeros((x.shape[0], len(self.topology['Rooms']), self.autoregression+1)).to(
                self.device)
            self.init_temp = x[:, 0, self.temperature_column]

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## D
            self.last_effect = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## E

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_effect
        self.past_inputs = torch.cat([self.past_inputs[:, :, 1:],
                                      x[:, -1, self.valve_column].unsqueeze(2)], dim=2)

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## D
        effect = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        if not self.predict_power:
            base[:, 1, :] = torch.from_numpy(self.zero_power).to(self.device)
            base = base.to(self.device)
        effect[:, 0, :] = self.last_effect.clone()

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            embeddings = []
            for i, base_input_nn in enumerate(self.base_input_nn):
                base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[0][-1]).to(self.device)
                for time_step in range(x.shape[1]):
                    temp = x[:, time_step, self.base_indices[i]]
                    for layer in base_input_nn:
                        temp = layer(temp)
                    base_embedding[:, time_step, :] = temp
                embeddings.append(base_embedding)
        else:
            base_embedding = [x[:, :, base_indices] for base_indices in self.base_indices]

        # LSTM prediction for the base temperature
        lstm_outputs = []
        for i, lstm in enumerate(self.base_lstm):
            lstm_output, (base_h[i], base_c[i]) = lstm(embeddings[i], (base_h[i], base_c[i]))
            lstm_outputs.append(lstm_output)

        if self.layer_norm:
            for i in range(len(self.base_norm)):
                lstm_outputs[i] = self.base_norm[i](lstm_outputs[i])

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            for j, base_output_nn in enumerate(self.base_output_nn):
                # Put the data is the form needed for the neural net
                temp = lstm_outputs[j][:, -1, :]
                # Go through the input layer of the NN
                for layer in base_output_nn:
                    temp = layer(temp)
                base[:, 0, j] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column[j]]
        else:
            # Store the outputs
            for j in range(len(lstm_outputs)):
                base[:, 0, j] = lstm_outputs[j][:, -1, 0] / self.base_division_factor[0] + x[:, -1,
                                                                                           self.temperature_column[j]]

        ## Heat losses computation in 'E'
        # Loss to the outside is b*(T_k-T^out_k)
        for i, room in enumerate(self.topology['Outside']):
            effect[:, 0, room] = effect[:, 0, room].clone() - self.b[i](
                (((x[:, -1, self.temperature_column[room]].squeeze()
                   + self.last_effect[:, room].clone() - 0.1) / 0.8
                  * self.room_diff[room] + self.room_min[room])
                 - ((x[:, -1, self.out_column] - 0.1) / 0.8
                    * self.out_diff + self.out_min)).
                    reshape(-1, 1)).squeeze() / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for i, (rooma, roomb) in enumerate(self.topology['Neighbors']):
            for room1, room2 in zip([rooma, roomb], [roomb, rooma]):
                effect[:, 0, room1] = effect[:, 0, room1].clone() - self.c[i](
                    (((x[:, -1, self.temperature_column[room1]].squeeze()
                       + self.last_effect[:, room1].clone() - 0.1) / 0.8
                      * self.room_diff[room1] + self.room_min[room1])
                     - ((x[:, -1, self.temperature_column[room2]]
                         + self.last_effect[:, room2].clone() - 0.1) / 0.8
                        * self.room_diff[room2] + self.room_min[room2])).
                        reshape(-1, 1)).squeeze() / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column].mean(axis=-1), axis=1).values > 0.1001)[0]
        if len(mask) > 0:

            # Find heating and cooling sequences
            heating = x[mask, 0, self.case_column] > 0.5
            cooling = x[mask, 0, self.case_column] < 0.5

            if not self.predict_power:
                temp = x[mask, -1, :].clone()
                # Substract the 'zero power' to get negative values for cooling power
                effect[mask, 1, :] = temp[:, self.power_column] - torch.from_numpy(self.zero_power).type(
                    torch.FloatTensor).to(self.device)

            else:
                # Input embedding when wanted
                if sum(heating) > 0:
                    heating_power_embedding = torch.zeros(len(mask[heating]), len(self.topology['Rooms']),
                                                          self.input_nn_hidden_sizes[1][-1]).to(self.device)
                    masked = self.past_inputs[mask[heating], :, :].clone()
                    for i, heating_power_input_nn in enumerate(self.heating_power_input_nn):
                        # temp = masked[:, :-len(self.topology['Rooms'])]
                        temp = torch.cat([masked[:, i, :], 0.9 - self.init_temp[mask[heating], i].unsqueeze(1),
                                          x[mask[heating], -1, self.effect_indices[0]].unsqueeze(1)], dim=1)
                        for layer in heating_power_input_nn:
                            temp = layer(temp)
                        heating_power_embedding[:, i, :] = temp.clone()

                if sum(cooling) > 0:
                    cooling_power_embedding = torch.zeros(len(mask[heating]), len(self.topology['Rooms']),
                                                          self.input_nn_hidden_sizes[1][-1]).to(self.device)
                    masked = self.past_inputs[mask[cooling], :, :].clone()
                    # temp = masked[:, :-len(self.topology['Rooms'])]
                    for i, cooling_power_input_nn in enumerate(self.cooling_power_input_nn):
                        temp = torch.cat([masked[:, i, :], self.init_temp[mask[cooling], i].unsqueeze(1),
                                          0.9 - x[mask[cooling], -1, self.effect_indices[1]].unsqueeze(1)], dim=1)
                        for layer in cooling_power_input_nn:
                            temp = layer(temp)
                        cooling_power_embedding[:, i, :] = temp.clone()

                # Some manipulations are needed to feed the output through the neural network if wanted
                if sum(heating) > 0:
                    # Go through the output layer of the NN
                    for i, heating_power_output_nn in enumerate(self.heating_power_output_nn):
                        temp = heating_power_embedding[:, i, :].clone()
                        for layer in heating_power_output_nn:
                            temp = layer(temp)
                        heat = x[mask[heating], :, :].clone()
                        effect[mask[heating], 1, i] = temp.squeeze() * (heat[:, -1, self.valve_column[i]] - 0.1) / len(
                            self.topology['Rooms'])  # / 0.8

                if sum(cooling) > 0:
                    for i, cooling_power_output_nn in enumerate(self.cooling_power_output_nn):
                        temp = cooling_power_embedding[:, i, :].clone()
                        for layer in cooling_power_output_nn:
                            temp = layer(temp)
                        cool = x[mask[cooling], :, :].clone()
                        effect[mask[cooling], 1, :] = - temp.squeeze() * (cool[:, -1, self.valve_column[i]] - 0.1) / len(
                            self.topology['Rooms'])  # / 0.8

            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[heating], 0, i] = effect[mask[heating], 0, i].clone() \
                                                  + self.a[i](
                        effect[mask[heating], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.a_scaling[i]

            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[cooling], 0, i] = effect[mask[cooling], 0, i].clone() \
                                                  + self.d[i](
                        effect[mask[cooling], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.d_scaling[i]

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, :].clone()
        self.last_effect = effect[:, 0, :].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        else:
            output = torch.zeros(x.shape[0], 2, effect.shape[-1])
            output[:, 0, :] = base[:, 0, :]
            output[:, 1, :] = effect[:, 0, :]

        # Return the predictions and states of the model
        return output, (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c)


class PCNN_full_shared(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            effect_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
            topology: dict,
            predict_power: bool,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the unforce dynamics (D)
            effect_indices:             Indices for the energy computation in E
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
            topology:                   Topology of the building to model
            predict_power:              Whether to use provided power measurements or infer them from data
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.topology = topology
        self.predict_power = predict_power

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]
        self.high_min = normalization_variables['High'][0]
        self.high_diff = normalization_variables['High'][1]
        self.low_min = normalization_variables['Low'][0]
        self.low_diff = normalization_variables['Low'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a']
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d']

        self.autoregression = len(self.topology['Rooms']) #* 4

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.ModuleList([PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))])
            self.b = nn.ModuleList([PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Outside']))])
            self.c = nn.ModuleList([PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Neighbors']))])
            self.d = nn.ModuleList([PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))])

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0]))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0]))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.base_indices)] + self.input_nn_hidden_sizes[0]
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[0][-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size[0],
                                 num_layers=self.lstm_num_layers[0], batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size[0])

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size[0]] + self.output_nn_hidden_sizes[0] + [len(self.topology['Rooms'])]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())
                                                 for i in range(0, len(sizes) - 1)])

        ## Predict the power
        if self.predict_power:

            # Process the input by a NN if wanted
            size = [self.autoregression + 1] + self.input_nn_hidden_sizes[1]
            self.heating_power_input_nn = nn.ModuleList([nn.Sequential(PositiveLinear(size[i], size[i + 1]), nn.ReLU())
                                                         for i in range(0, len(size) - 1)])
            self.cooling_power_input_nn = nn.ModuleList([nn.Sequential(PositiveLinear(size[i], size[i + 1]), nn.ReLU())
                                                         for i in range(0, len(size) - 1)])

            # Create the NNs to process the output of the LSTMs for each modules if wanted
            sizes = [self.input_nn_hidden_sizes[1][-1]] + self.output_nn_hidden_sizes[1] + [len(self.topology['Rooms'])]
            self.heating_power_output_nn = nn.ModuleList(
                [nn.Sequential(PositiveLinear(sizes[i], sizes[i + 1]), nn.ReLU())
                 for i in range(0, len(sizes) - 1)])
            self.cooling_power_output_nn = nn.ModuleList(
                [nn.Sequential(PositiveLinear(sizes[i], sizes[i + 1]), nn.ReLU())
                 for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                if 'log_weight' in name:
                    nn.init.normal_(param, -3.0, std=0.5)
                if 'a.' in name:
                    nn.init.constant_(param, 0.0)
                if 'b.' in name:
                    nn.init.constant_(param, 0.0)
                if 'd.' in name:
                    nn.init.constant_(param, 0.0)
                if 'c.' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False,
                mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device)

            heating_power_h = None
            heating_power_c = None
            cooling_power_h = None
            cooling_power_c = None

            self.past_inputs = torch.zeros((x.shape[0], self.autoregression + len(self.topology['Rooms']))).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## D
            self.last_effect = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## E

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_effect
        self.past_inputs = torch.cat([self.past_inputs[:, len(self.topology['Rooms']):],
                                      x[:, -1, self.valve_column]], dim=1)

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## D
        effect = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        if not self.predict_power:
            base[:, 1, :] = torch.from_numpy(self.zero_power).to(self.device)
            base = base.to(self.device)
        effect[:, 0, :] = self.last_effect.clone()

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[0][-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))

        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0, :] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]
        else:
            # Store the outputs
            base[:, 0, :] = lstm_output[:, -1, :] / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        ## Heat losses computation in 'E'
        # Loss to the outside is b*(T_k-T^out_k)
        for i, room in enumerate(self.topology['Outside']):
            effect[:, 0, room] = effect[:, 0, room].clone() - self.b[i](
                (((x[:, -1, self.temperature_column[room]].squeeze()
                   + self.last_effect[:, room].clone() - 0.1) / 0.8
                  * self.room_diff[room] + self.room_min[room])
                 - ((x[:, -1, self.out_column] - 0.1) / 0.8
                    * self.out_diff + self.out_min)).
                    reshape(-1, 1)).squeeze() / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for i, (rooma, roomb) in enumerate(self.topology['Neighbors']):
            for room1, room2 in zip([rooma, roomb], [roomb, rooma]):
                effect[:, 0, room1] = effect[:, 0, room1].clone() - self.c[i](
                    (((x[:, -1, self.temperature_column[room1]].squeeze()
                       + self.last_effect[:, room1].clone() - 0.1) / 0.8
                      * self.room_diff[room1] + self.room_min[room1])
                     - ((x[:, -1, self.temperature_column[room2]]
                         + self.last_effect[:, room2].clone() - 0.1) / 0.8
                        * self.room_diff[room2] + self.room_min[room2])).
                        reshape(-1, 1)).squeeze() / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column].mean(axis=-1), axis=1).values > 0.1001)[0]
        if len(mask) > 0:

            # Find heating and cooling sequences
            heating = x[mask, 0, self.case_column] > 0.5
            cooling = x[mask, 0, self.case_column] < 0.5

            if not self.predict_power:
                temp = x[mask, -1, :].clone()
                # Substract the 'zero power' to get negative values for cooling power
                effect[mask, 1, :] = temp[:, self.power_column] - torch.from_numpy(self.zero_power).type(
                    torch.FloatTensor).to(self.device)

            else:
                # Input embedding when wanted
                if sum(heating) > 0:
                    masked = self.past_inputs[mask[heating], :].clone()
                    #temp = masked[:, :-len(self.topology['Rooms'])]
                    temp = torch.cat([masked[:, :-len(self.topology['Rooms'])],
                               x[mask[heating], -1, self.effect_indices[0]].unsqueeze(1)], dim=1)
                    for layer in self.heating_power_input_nn:
                        temp = layer(temp)
                    heating_power_embedding = temp.clone()

                if sum(cooling) > 0:
                    masked = self.past_inputs[mask[cooling], :].clone()
                    #temp = masked[:, :-len(self.topology['Rooms'])]
                    temp = torch.cat([masked[:, :-len(self.topology['Rooms'])],
                                      - x[mask[cooling], -1, self.effect_indices[1]].unsqueeze(1)], dim=1)
                    for layer in self.cooling_power_input_nn:
                        temp = layer(temp)
                    cooling_power_embedding = temp.clone()

                # Some manipulations are needed to feed the output through the neural network if wanted
                if sum(heating) > 0:
                    # Put the data is the form needed for the neural net
                    temp = heating_power_embedding
                    # Go through the output layer of the NN
                    for layer in self.heating_power_output_nn:
                        temp = layer(temp)
                    heat = x[mask[heating], :, :].clone()
                    effect[mask[heating], 1, :] = temp * (heat[:, -1, self.valve_column] - 0.1) / len(self.topology['Rooms']) # / 0.8
                    #effect[mask[heating], 1, :] = temp * (heat[:, -1, self.valve_column] - 0.1) / 0.8 *\
                    #                              torch.relu(((heat[:, -1, self.effect_indices[0]].unsqueeze(1) - 0.1) / 0.8
                    #                                  * self.high_diff + self.high_min) -
                    #                               ((heat[:, -1, self.temperature_column]
                    #                                 + self.last_effect[mask[heating], :].clone() - 0.1) / 0.8
                    #                                * torch.FloatTensor(self.room_diff.values).to(self.device) + torch.FloatTensor(self.room_min.values).to(self.device)))
                if sum(cooling) > 0:
                    # Put the data is the form needed for the neural net
                    temp = cooling_power_embedding
                    # Go through the output layer of the NN
                    for layer in self.cooling_power_output_nn:
                        temp = layer(temp)
                    cool = x[mask[cooling], :, :].clone()
                    effect[mask[cooling], 1, :] =  - temp * (cool[:, -1, self.valve_column] - 0.1) / len(self.topology['Rooms']) # / 0.8
                    #effect[mask[cooling], 1, :] = - temp * (cool[:, -1, self.valve_column] - 0.1) / 0.8 *\
                    #                              torch.relu(((cool[:, -1, self.temperature_column]
                    #                                 + self.last_effect[mask[cooling], :].clone() - 0.1) / 0.8
                    #                                * torch.FloatTensor(self.room_diff.values).to(self.device) + torch.FloatTensor(self.room_min.values).to(self.device))
                    #                                         - ((cool[:, -1, self.effect_indices[0]].unsqueeze(1) - 0.1) / 0.8
                    #                                  * self.low_diff + self.low_min))

            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[heating], 0, i] = effect[mask[heating], 0, i].clone() \
                                                  + self.a[i](
                        effect[mask[heating], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.a_scaling[i]

            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[cooling], 0, i] = effect[mask[cooling], 0, i].clone() \
                                                  + self.d[i](
                        effect[mask[cooling], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.d_scaling[i]

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, :].clone()
        self.last_effect = effect[:, 0, :].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        else:
            output = torch.zeros(x.shape[0], 2, effect.shape[-1])
            output[:, 0, :] = base[:, 0, :]
            output[:, 1, :] = effect[:, 0, :]

        # Return the predictions and states of the model
        return output, (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c)


class PCNN_full_shared_third(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            effect_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
            topology: dict,
            predict_power: bool,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the unforce dynamics (D)
            effect_indices:             Indices for the energy computation in E
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
            topology:                   Topology of the building to model
            predict_power:              Whether to use provided power measurements or infer them from data
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.topology = topology
        self.predict_power = predict_power

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a']
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d']

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.ModuleList([PositiveLinear(1, 1) for _ in range(len(self.topology['Rooms']))])
            self.b = nn.ModuleList([PositiveLinear(1, 1) for _ in range(len(self.topology['Outside']))])
            self.c = nn.ModuleList([PositiveLinear(1, 1) for _ in range(len(self.topology['Neighbors']))])
            self.d = nn.ModuleList([PositiveLinear(1, 1) for _ in range(len(self.topology['Rooms']))])

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0]))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0]))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.base_indices)] + self.input_nn_hidden_sizes[0]
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[0][-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size[0],
                                 num_layers=self.lstm_num_layers[0], batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size[0])

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size[0]] + self.output_nn_hidden_sizes[0] + [len(self.topology['Rooms'])]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())
                                                 for i in range(0, len(sizes) - 1)])

        ## Predict the power
        if self.predict_power:

            # Hidden and cell state initialization
            if self.learn_initial_hidden_states:
                self.initial_heating_power_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers[1], self.lstm_hidden_size[1]))
                self.initial_heating_power_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers[1], self.lstm_hidden_size[1]))
                self.initial_cooling_power_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers[1], self.lstm_hidden_size[1]))
                self.initial_cooling_power_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers[1], self.lstm_hidden_size[1]))

            # Process the input by a NN if wanted
            if self.feed_input_through_nn:
                size = [len(self.effect_indices)] + self.input_nn_hidden_sizes[1]
                self.heating_power_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                    for i in range(0, len(size) - 1)])
                self.cooling_power_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                     for i in range(0, len(size) - 1)])

            # Create the LSTMs at the core of `D`, with normalization layers
            lstm_input_size = self.input_nn_hidden_sizes[1][-1] if self.feed_input_through_nn else len(self.effect_indices)
            self.heating_power_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size[1],
                                     num_layers=self.lstm_num_layers[1], batch_first=True)
            self.cooling_power_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size[1],
                                      num_layers=self.lstm_num_layers[1], batch_first=True)

            if self.layer_norm:
                self.heating_power_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size[1])
                self.cooling_power_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size[1])

            # Create the NNs to process the output of the LSTMs for each modules if wanted
            if self.feed_output_through_nn:
                sizes = [self.lstm_hidden_size[1]] + self.output_nn_hidden_sizes[1] + [len(self.topology['Rooms'])]
                self.heating_power_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Sigmoid())
                                                     for i in range(0, len(sizes) - 1)])
                self.cooling_power_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Sigmoid())
                                                      for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                if 'a.' in name:
                    nn.init.constant_(param, 0.0)
                if 'b.' in name:
                    nn.init.constant_(param, 0.0)
                if 'd.' in name:
                    nn.init.constant_(param, 0.0)
                if 'c.' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False,
                mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device)

            if self.predict_power:
                if self.learn_initial_hidden_states:
                    heating_power_h = torch.stack([self.initial_heating_power_h.clone() for _ in range(x.shape[0])], dim=1).to(
                        self.device)
                    heating_power_c = torch.stack([self.initial_heating_power_c.clone() for _ in range(x.shape[0])], dim=1).to(
                        self.device)
                    cooling_power_h = torch.stack([self.initial_cooling_power_h.clone() for _ in range(x.shape[0])], dim=1).to(
                        self.device)
                    cooling_power_c = torch.stack([self.initial_cooling_power_c.clone() for _ in range(x.shape[0])], dim=1).to(
                        self.device)
                else:
                    heating_power_h = torch.zeros((self.lstm_num_layers[1], x.shape[0], self.lstm_hidden_size[1])).to(
                        self.device)
                    heating_power_c = torch.zeros((self.lstm_num_layers[1], x.shape[0], self.lstm_hidden_size[1])).to(
                        self.device)
                    cooling_power_h = torch.zeros((self.lstm_num_layers[1], x.shape[0], self.lstm_hidden_size[1])).to(
                        self.device)
                    cooling_power_c = torch.zeros((self.lstm_num_layers[1], x.shape[0], self.lstm_hidden_size[1])).to(
                        self.device)
            else:
                heating_power_h = None
                heating_power_c = None
                cooling_power_h = None
                cooling_power_c = None

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## D
            self.last_effect = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## E

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_effect

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## D
        effect = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        if not self.predict_power:
            base[:, 1, :] = torch.from_numpy(self.zero_power).to(self.device)
            base = base.to(self.device)
        effect[:, 0, :] = self.last_effect.clone()

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[0][-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))

        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0, :] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]
        else:
            # Store the outputs
            base[:, 0, :] = lstm_output[:, -1, :] / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        ## Heat losses computation in 'E'
        # Loss to the outside is b*(T_k-T^out_k)
        for i, room in enumerate(self.topology['Outside']):
            effect[:, 0, room] = effect[:, 0, room].clone() - self.b[i](
                (((x[:, -1, self.temperature_column[room]].squeeze()
                   + self.last_effect[:, room].clone() - 0.1) / 0.8
                  * self.room_diff[room] + self.room_min[room])
                 - ((x[:, -1, self.out_column] - 0.1) / 0.8
                    * self.out_diff + self.out_min)).
                    reshape(-1, 1)).squeeze() / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for i, (rooma, roomb) in enumerate(self.topology['Neighbors']):
            for room1, room2 in zip([rooma, roomb], [roomb, rooma]):
                effect[:, 0, room1] = effect[:, 0, room1].clone() - self.c[i](
                    (((x[:, -1, self.temperature_column[room1]].squeeze()
                       + self.last_effect[:, room1].clone() - 0.1) / 0.8
                      * self.room_diff[room1] + self.room_min[room1])
                     - ((x[:, -1, self.temperature_column[room2]]
                         + self.last_effect[:, room2].clone() - 0.1) / 0.8
                        * self.room_diff[room2] + self.room_min[room2])).
                        reshape(-1, 1)).squeeze() / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column].mean(axis=-1), axis=1).values > 0.1001)[0]
        if len(mask) > 0:

            # Find heating and cooling sequences
            heating = x[mask, 0, self.case_column] > 0.5
            cooling = x[mask, 0, self.case_column] < 0.5

            if not self.predict_power:
                temp = x[mask, -1, :].clone()
                # Substract the 'zero power' to get negative values for cooling power
                effect[mask, 1, :] = temp[:, self.power_column] - torch.from_numpy(self.zero_power).type(
                    torch.FloatTensor).to(self.device)

            else:
                # Input embedding when wanted
                if self.feed_input_through_nn:
                    if sum(heating) > 0:
                        heating_power_embedding = torch.zeros(len(mask[heating]), x.shape[1], self.input_nn_hidden_sizes[1][-1]).to(
                            self.device)
                        masked = x[mask[heating], :, :].clone()
                        for time_step in range(x.shape[1]):
                            temp = masked[:, time_step, self.effect_indices]
                            for layer in self.heating_power_input_nn:
                                temp = layer(temp)
                            heating_power_embedding[:, time_step, :] = temp

                    if sum(cooling) > 0:
                        cooling_power_embedding = torch.zeros(len(mask[cooling]), x.shape[1], self.input_nn_hidden_sizes[1][-1]).to(
                            self.device)
                        masked = x[mask[cooling], :, :].clone()
                        for time_step in range(x.shape[1]):
                            temp = masked[:, time_step, self.effect_indices]
                            for layer in self.cooling_power_input_nn:
                                temp = layer(temp)
                            cooling_power_embedding[:, time_step, :] = temp
                else:
                    masked = x[mask, :, :].clone()
                    if sum(heating) > 0:
                        heating_power_embedding = masked[heating, :, self.effect_indices]
                    if sum(cooling) > 0:
                        cooling_power_embedding = masked[cooling, :, self.effect_indices]

                # LSTM prediction for the heating/cooling
                if sum(heating) > 0:
                    heating_lstm_output, (heating_power_h[:, mask[heating], :], heating_power_c[:, mask[heating], :]) = \
                        self.heating_power_lstm(heating_power_embedding, (heating_power_h[:, mask[heating], :],
                                                                          heating_power_c[:, mask[heating], :]))
                if sum(cooling) > 0:
                    cooling_lstm_output, (cooling_power_h[:, mask[cooling], :], cooling_power_c[:, mask[cooling], :]) = \
                        self.cooling_power_lstm(cooling_power_embedding, (cooling_power_h[:, mask[cooling], :],
                                                                          cooling_power_c[:, mask[cooling], :]))

                if self.layer_norm:
                    if sum(heating) > 0:
                        heating_lstm_output = self.heating_power_norm(heating_lstm_output)
                    if sum(cooling) > 0:
                        cooling_lstm_output = self.cooling_power_norm(cooling_lstm_output)

                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    if sum(heating) > 0:
                        # Put the data is the form needed for the neural net
                        temp = heating_lstm_output[:, -1, :]
                        # Go through the output layer of the NN
                        for layer in self.heating_power_output_nn:
                            temp = layer(temp)
                        heat = x[mask[heating], :, :].clone()
                        effect[mask[heating], 1, :] = temp.squeeze() * (heat[:, :, self.valve_column].squeeze() - 0.1) / 0.8 / 2
                    if sum(cooling) > 0:
                        # Put the data is the form needed for the neural net
                        temp = cooling_lstm_output[:, -1, :]
                        # Go through the output layer of the NN
                        for layer in self.cooling_power_output_nn:
                            temp = layer(temp)
                        cool = x[mask[cooling], :, :].clone()
                        effect[mask[cooling], 1, :] = - temp.squeeze() * (cool[:, :, self.valve_column].squeeze() - 0.1) / 0.8 / 2

                else:
                    # Store the outputs
                    effect[mask, 1, :] = lstm_output[:, -1, :]

            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[heating], 0, i] = effect[mask[heating], 0, i].clone() \
                                                  + self.a[i](
                        effect[mask[heating], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.a_scaling[i]

            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[cooling], 0, i] = effect[mask[cooling], 0, i].clone() \
                                                  + self.d[i](
                        effect[mask[cooling], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.d_scaling[i]

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, :].clone()
        self.last_effect = effect[:, 0, :].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        else:
            output = torch.zeros(x.shape[0], 2, effect.shape[-1])
            output[:, 0, :] = base[:, 0, :]
            output[:, 1, :] = effect[:, 0, :]

        # Return the predictions and states of the model
        return output, (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c)


class PCNN_full_shared_bis(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            effect_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
            topology: dict,
            predict_power: bool,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the unforce dynamics (D)
            effect_indices:             Indices for the energy computation in E
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
            topology:                   Topology of the building to model
            predict_power:              Whether to use provided power measurements or infer them from data
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.topology = topology
        self.predict_power = predict_power

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a']
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d']

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Rooms']))])
            self.b = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Outside']))])
            self.c = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Neighbors']))])
            self.d = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Rooms']))])

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0]))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers[0], self.lstm_hidden_size[0]))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.base_indices)] + self.input_nn_hidden_sizes[0]
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[0][-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size[0],
                                 num_layers=self.lstm_num_layers[0], batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size[0])

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size[0]] + self.output_nn_hidden_sizes[0] + [len(self.topology['Rooms'])]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())
                                                 for i in range(0, len(sizes) - 1)])

        ## Predict the power
        if self.predict_power:

            # Hidden and cell state initialization
            if self.learn_initial_hidden_states:
                self.initial_heating_power_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers[1], self.lstm_hidden_size[1]))
                self.initial_heating_power_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers[1], self.lstm_hidden_size[1]))
                self.initial_cooling_power_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers[1], self.lstm_hidden_size[1]))
                self.initial_cooling_power_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers[1], self.lstm_hidden_size[1]))

            # Process the input by a NN if wanted
            if self.feed_input_through_nn:
                size = [len(self.effect_indices)] + self.input_nn_hidden_sizes[1]
                self.heating_power_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                    for i in range(0, len(size) - 1)])
                self.cooling_power_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                     for i in range(0, len(size) - 1)])

            # Create the LSTMs at the core of `D`, with normalization layers
            lstm_input_size = self.input_nn_hidden_sizes[1][-1] if self.feed_input_through_nn else len(self.effect_indices)
            self.heating_power_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size[1],
                                     num_layers=self.lstm_num_layers[1], batch_first=True)
            self.cooling_power_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size[1],
                                      num_layers=self.lstm_num_layers[1], batch_first=True)

            if self.layer_norm:
                self.heating_power_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size[1])
                self.cooling_power_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size[1])

            # Create the NNs to process the output of the LSTMs for each modules if wanted
            if self.feed_output_through_nn:
                sizes = [self.lstm_hidden_size[1]] + self.output_nn_hidden_sizes[1] + [len(self.topology['Rooms'])]
                self.heating_power_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Sigmoid())
                                                     for i in range(0, len(sizes) - 1)])
                self.cooling_power_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Sigmoid())
                                                      for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
            if 'a.' in name:
                nn.init.constant_(param, 1.)
            if 'b.' in name:
                nn.init.constant_(param, 1.)
            if 'd.' in name:
                nn.init.constant_(param, 1.)
            if 'c.' in name:
                nn.init.constant_(param, 1.)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False,
                mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers[0], x.shape[0], self.lstm_hidden_size[0])).to(self.device)

            if self.predict_power:
                if self.learn_initial_hidden_states:
                    heating_power_h = torch.stack([self.initial_heating_power_h.clone() for _ in range(x.shape[0])], dim=1).to(
                        self.device)
                    heating_power_c = torch.stack([self.initial_heating_power_c.clone() for _ in range(x.shape[0])], dim=1).to(
                        self.device)
                    cooling_power_h = torch.stack([self.initial_cooling_power_h.clone() for _ in range(x.shape[0])], dim=1).to(
                        self.device)
                    cooling_power_c = torch.stack([self.initial_cooling_power_c.clone() for _ in range(x.shape[0])], dim=1).to(
                        self.device)
                else:
                    heating_power_h = torch.zeros((self.lstm_num_layers[1], x.shape[0], self.lstm_hidden_size[1])).to(
                        self.device)
                    heating_power_c = torch.zeros((self.lstm_num_layers[1], x.shape[0], self.lstm_hidden_size[1])).to(
                        self.device)
                    cooling_power_h = torch.zeros((self.lstm_num_layers[1], x.shape[0], self.lstm_hidden_size[1])).to(
                        self.device)
                    cooling_power_c = torch.zeros((self.lstm_num_layers[1], x.shape[0], self.lstm_hidden_size[1])).to(
                        self.device)
            else:
                heating_power_h = None
                heating_power_c = None
                cooling_power_h = None
                cooling_power_c = None

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## D
            self.last_effect = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## E

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_effect

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## D
        effect = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        if not self.predict_power:
            base[:, 1, :] = torch.from_numpy(self.zero_power).to(self.device)
            base = base.to(self.device)
        effect[:, 0, :] = self.last_effect.clone()

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[0][-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))

        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0, :] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]
        else:
            # Store the outputs
            base[:, 0, :] = lstm_output[:, -1, :] / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        ## Heat losses computation in 'E'
        # Loss to the outside is b*(T_k-T^out_k)
        for i, room in enumerate(self.topology['Outside']):
            effect[:, 0, room] = effect[:, 0, room].clone() - self.b[i](
                (((x[:, -1, self.temperature_column[room]].squeeze()
                   + self.last_effect[:, room].clone() - 0.1) / 0.8
                  * self.room_diff[room] + self.room_min[room])
                 - ((x[:, -1, self.out_column] - 0.1) / 0.8
                    * self.out_diff + self.out_min)).
                    reshape(-1, 1)).squeeze() / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for i, (rooma, roomb) in enumerate(self.topology['Neighbors']):
            for room1, room2 in zip([rooma, roomb], [roomb, rooma]):
                effect[:, 0, room1] = effect[:, 0, room1].clone() - self.c[i](
                    (((x[:, -1, self.temperature_column[room1]].squeeze()
                       + self.last_effect[:, room1].clone() - 0.1) / 0.8
                      * self.room_diff[room1] + self.room_min[room1])
                     - ((x[:, -1, self.temperature_column[room2]]
                         + self.last_effect[:, room2].clone() - 0.1) / 0.8
                        * self.room_diff[room2] + self.room_min[room2])).
                        reshape(-1, 1)).squeeze() / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column].mean(axis=-1), axis=1).values > 0.1001)[0]
        if len(mask) > 0:

            # Find heating and cooling sequences
            heating = x[mask, 0, self.case_column] > 0.5
            cooling = x[mask, 0, self.case_column] < 0.5

            if not self.predict_power:
                temp = x[mask, -1, :].clone()
                # Substract the 'zero power' to get negative values for cooling power
                effect[mask, 1, :] = temp[:, self.power_column] - torch.from_numpy(self.zero_power).type(
                    torch.FloatTensor).to(self.device)

            else:
                # Input embedding when wanted
                if self.feed_input_through_nn:
                    if sum(heating) > 0:
                        heating_power_embedding = torch.zeros(len(mask[heating]), x.shape[1], self.input_nn_hidden_sizes[1][-1]).to(
                            self.device)
                        masked = x[mask[heating], :, :].clone()
                        for time_step in range(x.shape[1]):
                            temp = masked[:, time_step, self.effect_indices]
                            for layer in self.heating_power_input_nn:
                                temp = layer(temp)
                            heating_power_embedding[:, time_step, :] = temp

                    if sum(cooling) > 0:
                        cooling_power_embedding = torch.zeros(len(mask[cooling]), x.shape[1], self.input_nn_hidden_sizes[1][-1]).to(
                            self.device)
                        masked = x[mask[cooling], :, :].clone()
                        for time_step in range(x.shape[1]):
                            temp = masked[:, time_step, self.effect_indices]
                            for layer in self.cooling_power_input_nn:
                                temp = layer(temp)
                            cooling_power_embedding[:, time_step, :] = temp
                else:
                    masked = x[mask, :, :].clone()
                    if sum(heating) > 0:
                        heating_power_embedding = masked[heating, :, self.effect_indices]
                    if sum(cooling) > 0:
                        cooling_power_embedding = masked[cooling, :, self.effect_indices]

                # LSTM prediction for the heating/cooling
                if sum(heating) > 0:
                    heating_lstm_output, (heating_power_h[:, mask[heating], :], heating_power_c[:, mask[heating], :]) = \
                        self.heating_power_lstm(heating_power_embedding, (heating_power_h[:, mask[heating], :],
                                                                          heating_power_c[:, mask[heating], :]))
                if sum(cooling) > 0:
                    cooling_lstm_output, (cooling_power_h[:, mask[cooling], :], cooling_power_c[:, mask[cooling], :]) = \
                        self.cooling_power_lstm(cooling_power_embedding, (cooling_power_h[:, mask[cooling], :],
                                                                          cooling_power_c[:, mask[cooling], :]))

                if self.layer_norm:
                    if sum(heating) > 0:
                        heating_lstm_output = self.heating_power_norm(heating_lstm_output)
                    if sum(cooling) > 0:
                        cooling_lstm_output = self.cooling_power_norm(cooling_lstm_output)

                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    if sum(heating) > 0:
                        # Put the data is the form needed for the neural net
                        temp = heating_lstm_output[:, -1, :]
                        # Go through the output layer of the NN
                        for layer in self.heating_power_output_nn:
                            temp = layer(temp)
                        heat = x[mask[heating], :, :].clone()
                        effect[mask[heating], 1, :] = temp.squeeze() * (heat[:, :, self.valve_column].squeeze() - 0.1) / 0.8 / 2
                    if sum(cooling) > 0:
                        # Put the data is the form needed for the neural net
                        temp = cooling_lstm_output[:, -1, :]
                        # Go through the output layer of the NN
                        for layer in self.cooling_power_output_nn:
                            temp = layer(temp)
                        cool = x[mask[cooling], :, :].clone()
                        effect[mask[cooling], 1, :] = - temp.squeeze() * (cool[:, :, self.valve_column].squeeze() - 0.1) / 0.8 / 2

                else:
                    # Store the outputs
                    effect[mask, 1, :] = lstm_output[:, -1, :]

            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[heating], 0, i] = effect[mask[heating], 0, i].clone() \
                                                  + self.a[i](
                        effect[mask[heating], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.a_scaling[i]

            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[cooling], 0, i] = effect[mask[cooling], 0, i].clone() \
                                                  + self.d[i](
                        effect[mask[cooling], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.d_scaling[i]

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, :].clone()
        self.last_effect = effect[:, 0, :].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        else:
            output = torch.zeros(x.shape[0], 2, effect.shape[-1])
            output[:, 0, :] = base[:, 0, :]
            output[:, 1, :] = effect[:, 0, :]

        # Return the predictions and states of the model
        return output, (base_h, base_c, heating_power_h, heating_power_c, cooling_power_h, cooling_power_c)


class PCNN_full_shared_old(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            effect_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
            topology: dict,
            use_energy_prediction: bool,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the unforce dynamics (D)
            effect_indices:             Indices for the energy computation in E
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
            topology:                   Topology of the building to model
            use_energy_prediction:      Whether to use provided power measurements or infer them from data
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.topology = topology
        self.use_energy_prediction = use_energy_prediction

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a']
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d']

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative
        f = nn.Tanh()

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Rooms']))])
            self.b = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Outside']))])
            self.c = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Neighbors']))])
            self.d = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Rooms']))])

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), f)
                                                               for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                                num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [len(self.topology['Rooms'])]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                                for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
            if 'a.' in name:
                nn.init.constant_(param, 1.)
            if 'b.' in name:
                nn.init.constant_(param, 1.)
            if 'd.' in name:
                nn.init.constant_(param, 1.)
            if 'c.' in name:
                nn.init.constant_(param, 1.)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False,
                mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## D
            self.last_effect = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## E

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_effect

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## D
        effect = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        base[:, 1, :] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        effect[:, 0, :] = self.last_effect.clone()

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))

        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0, :] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]
        else:
            # Store the outputs
            base[:, 0, :] = lstm_output[:, -1, :] / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        ## Heat losses computation in 'E'
        # Loss to the outside is b*(T_k-T^out_k)
        for i, room in enumerate(self.topology['Outside']):
            effect[:, 0, room] = effect[:, 0, room].clone() - self.b[i](
                (((x[:, -1, self.temperature_column[room]].squeeze()
                   + self.last_effect[:, room].clone() - 0.1) / 0.8
                  * self.room_diff[room] + self.room_min[room])
                 - ((x[:, -1, self.out_column] - 0.1) / 0.8
                    * self.out_diff + self.out_min)).
                reshape(-1, 1)).squeeze() / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for i, (rooma, roomb) in enumerate(self.topology['Neighbors']):
            for room1, room2 in zip([rooma, roomb], [roomb, rooma]):
                effect[:, 0, room1] = effect[:, 0, room1].clone() - self.c[i](
                    (((x[:, -1, self.temperature_column[room1]].squeeze()
                       + self.last_effect[:, room1].clone() - 0.1) / 0.8
                      * self.room_diff[room1] + self.room_min[room1])
                     - ((x[:, -1, self.temperature_column[room2]]
                         + self.last_effect[:, room2].clone() - 0.1) / 0.8
                        * self.room_diff[room2] + self.room_min[room2])).
                    reshape(-1, 1)).squeeze() / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column].mean(axis=-1), axis=1).values > 0.1001)[0]
        if len(mask) > 0:
            temp = x[mask, -1, :].clone()
            # Substract the 'zero power' to get negative values for cooling power
            effect[mask, 1, :] = temp[:, self.power_column] - torch.from_numpy(self.zero_power).type(
                torch.FloatTensor).to(self.device)

            # Find the sequences in the batch that are heating
            heating = x[mask, 0, self.case_column] > 0.5
            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[heating], 0, i] = effect[mask[heating], 0, i].clone() \
                                                  + self.a[i](
                        effect[mask[heating], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.a_scaling[i]

            cooling = x[mask, 0, self.case_column] < 0.5
            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[cooling], 0, i] = effect[mask[cooling], 0, i].clone() \
                                                  + self.d[i](
                        effect[mask[cooling], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.d_scaling[i]

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, :].clone()
        self.last_effect = effect[:, 0, :].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        else:
            output = torch.zeros(x.shape[0], 2, effect.shape[-1])
            output[:, 0, :] = base[:, 0, :]
            output[:, 1, :] = effect[:, 0, :]

        # Return the predictions and states of the model
        return output, (base_h, base_c)

class PCNN_full_old(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
            topology: dict,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.topology = topology

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a']
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d']

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative
        f = nn.Tanh()

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Rooms']))])
            self.b = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Outside']))])
            self.c = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Neighbors']))])
            self.d = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.topology['Rooms']))])

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.ParameterList(
                [nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size)) for _ in
                 range(len(self.topology['Rooms']))])
            self.initial_base_c = nn.ParameterList(
                [nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size)) for _ in
                 range(len(self.topology['Rooms']))])

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            sizes = [[len(base_indices)] + self.input_nn_hidden_sizes for base_indices in self.base_indices]
            self.base_input_nn = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), f)
                                                               for i in range(0, len(size) - 1)]) for size in sizes])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = [self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(base_indices) for
                           base_indices in self.base_indices]
        self.base_lstm = nn.ModuleList([nn.LSTM(input_size=lstm_input_size[i], hidden_size=self.lstm_hidden_size,
                                                num_layers=self.lstm_num_layers, batch_first=True) for i in
                                        range(len(self.topology['Rooms']))])
        if self.layer_norm:
            self.base_norm = nn.ModuleList(
                [nn.LayerNorm(normalized_shape=self.lstm_hidden_size) for _ in range(len(self.topology['Rooms']))])

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                                for i in range(0, len(sizes) - 1)]) for _ in
                                                 range(len(self.topology['Rooms']))])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
            if 'a.' in name:
                nn.init.constant_(param, 1.)
            if 'b.' in name:
                nn.init.constant_(param, 1.)
            if 'd.' in name:
                nn.init.constant_(param, 1.)
            if 'c.' in name:
                nn.init.constant_(param, 1.)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False,
                mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = [torch.stack([initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device) for
                          initial_base_h in self.initial_base_h]
                base_c = [torch.stack([initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device) for
                          initial_base_c in self.initial_base_c]
            else:
                base_h = [torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device) for _
                          in range(len(self.topology['Rooms']))]
                base_c = [torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device) for _
                          in range(len(self.topology['Rooms']))]

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## D
            self.last_effect = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## E

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, [self.temperature_column]] = self.last_base.unsqueeze(1)
        else:
            x[:, -1, [self.temperature_column]] = x[:, -1,
                                                  [self.temperature_column]].clone() - self.last_effect.unsqueeze(1)

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, len(self.topology['Rooms']))  ## D
        effect = torch.zeros(x.shape[0], 2, len(self.topology['Rooms'])).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        base[:, 1, :] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        effect[:, 0, :] = self.last_effect.clone()

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            embeddings = []
            for i, base_input_nn in enumerate(self.base_input_nn):
                base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
                for time_step in range(x.shape[1]):
                    temp = x[:, time_step, self.base_indices[i]]
                    for layer in base_input_nn:
                        temp = layer(temp)
                    base_embedding[:, time_step, :] = temp
                embeddings.append(base_embedding)
        else:
            base_embedding = [x[:, :, base_indices] for base_indices in self.base_indices]

        # LSTM prediction for the base temperature
        lstm_outputs = []
        for i, lstm in enumerate(self.base_lstm):
            lstm_output, (base_h[i], base_c[i]) = lstm(embeddings[i], (base_h[i], base_c[i]))
            lstm_outputs.append(lstm_output)

        if self.layer_norm:
            for i in range(len(self.base_norm)):
                lstm_outputs[i] = self.base_norm[i](lstm_outputs[i])

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            for j, base_output_nn in enumerate(self.base_output_nn):
                # Put the data is the form needed for the neural net
                temp = lstm_outputs[j][:, -1, :]
                # Go through the input layer of the NN
                for layer in base_output_nn:
                    temp = layer(temp)
                base[:, 0, j] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column[j]]
        else:
            # Store the outputs
            for j in range(len(lstm_outputs)):
                base[:, 0, j] = lstm_outputs[j][:, -1, 0] / self.base_division_factor[0] + x[:, -1,
                                                                                           self.temperature_column[j]]

        ## Heat losses computation in 'E'
        # Loss to the outside is b*(T_k-T^out_k)
        for i, room in enumerate(self.topology['Outside']):
            effect[:, 0, room] = effect[:, 0, room].clone() - self.b[i](
                (((x[:, -1, self.temperature_column[room]].squeeze()
                   + self.last_effect[:, room].clone() - 0.1) / 0.8
                  * self.room_diff[room] + self.room_min[room])
                 - ((x[:, -1, self.out_column] - 0.1) / 0.8
                    * self.out_diff + self.out_min)).
                reshape(-1, 1)).squeeze() / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for i, (rooma, roomb) in enumerate(self.topology['Neighbors']):
            for room1, room2 in zip([rooma, roomb], [roomb, rooma]):
                effect[:, 0, room1] = effect[:, 0, room1].clone() - self.c[i](
                    (((x[:, -1, self.temperature_column[room1]].squeeze()
                       + self.last_effect[:, room1].clone() - 0.1) / 0.8
                      * self.room_diff[room1] + self.room_min[room1])
                     - ((x[:, -1, self.temperature_column[room2]]
                         + self.last_effect[:, room2].clone() - 0.1) / 0.8
                        * self.room_diff[room2] + self.room_min[room2])).
                    reshape(-1, 1)).squeeze() / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column].mean(axis=-1), axis=1).values > 0.1001)[0]
        if len(mask) > 0:
            temp = x[mask, -1, :].clone()
            # Substract the 'zero power' to get negative values for cooling power
            effect[mask, 1, :] = temp[:, self.power_column] - torch.from_numpy(self.zero_power).type(
                torch.FloatTensor).to(self.device)

            # Find the sequences in the batch that are heating
            heating = x[mask, 0, self.case_column] > 0.5
            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[heating], 0, i] = effect[mask[heating], 0, i].clone() \
                                                  + self.a[i](
                        effect[mask[heating], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.a_scaling[i]

            cooling = x[mask, 0, self.case_column] < 0.5
            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                for i in range(len(self.topology['Rooms'])):
                    effect[mask[cooling], 0, i] = effect[mask[cooling], 0, i].clone() \
                                                  + self.a[i](
                        effect[mask[cooling], 1, i].clone().reshape(-1, 1)).squeeze() \
                                                  / self.a_scaling[i]

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, :].clone()
        self.last_effect = effect[:, 0, :].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        else:
            output = torch.zeros(x.shape[0], 2, effect.shape[-1])
            output[:, 0, :] = base[:, 0, :]
            output[:, 1, :] = effect[:, 0, :]

        # Return the predictions and states of the model
        return output, (base_h, base_c)


class PCNN_mul(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            neigh_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.neigh_column = neigh_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0].values.astype(np.float32)[0]
        self.room_diff = normalization_variables['Room'][1].values.astype(np.float32)[0]
        self.out_min = normalization_variables['Out'][0].astype(np.float32)
        self.out_diff = normalization_variables['Out'][1].astype(np.float32)
        self.neigh_min = normalization_variables['Neigh'][0].values.astype(np.float32)
        self.neigh_diff = normalization_variables['Neigh'][1].values.astype(np.float32)

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a'][0]
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d'][0]

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative
        f = nn.Tanh()

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = PositiveLinear(1, 1, require_bias=False)
            self.b = PositiveLinear(1, 1, require_bias=False)
            self.c = nn.ModuleList([PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.neigh_column))])
            self.d = PositiveLinear(1, 1, require_bias=False)

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                for i in range(0, len(sizes) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                if 'a.' in name:
                    nn.init.constant_(param, 0.0)
                if 'b.' in name:
                    nn.init.constant_(param, 0.0)
                if 'd.' in name:
                    nn.init.constant_(param, 0.0)
                if 'c.' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False, mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros(x.shape[0]).to(self.device)  ## D
            self.last_effect = torch.zeros(x.shape[0]).to(self.device)  ## E

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base.unsqueeze(1)
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_effect.unsqueeze(1)

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, 1)  ## D
        effect = torch.zeros(x.shape[0], 2, 1).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        base[:, 1, 0] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        effect[:, 0, 0] = self.last_effect.clone()

        ## Heat losses computation in 'E'
        if self.learn_discount_factors:
            # Loss to the outside is b*(T_k-T^out_k)
            effect[:, 0, 0] = effect[:, 0, 0].clone() - self.b((((x[:, -1, self.temperature_column].squeeze()
                                                                      + self.last_effect.clone() - 0.1) / 0.8
                                                                     * self.room_diff + self.room_min)
                                                                    - ((x[:, -1, self.out_column] - 0.1) / 0.8
                                                                       * self.out_diff + self.out_min)).
                                                                   reshape(-1, 1)).squeeze() / self.b_scaling
            # Loss to the neighboring zone is c*(T_k-T^neigh_k)
            for i in range(len(self.neigh_column)):
                effect[:, 0, 0] = effect[:, 0, 0].clone() - self.c[i]((((x[:, -1, self.temperature_column].squeeze()
                                                                          + self.last_effect.clone() - 0.1) / 0.8
                                                                         * self.room_diff + self.room_min)
                                                                        - ((x[:, -1, self.neigh_column[i]] - 0.1) / 0.8
                                                                           * self.neigh_diff[i] + self.neigh_min[i])).
                                                                       reshape(-1, 1)).squeeze() / self.c_scaling

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0, 0] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column].squeeze()
        else:
            # Store the outputs
            base[:, 0, 0] = lstm_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1, self.temperature_column].squeeze()

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column], axis=1).values > 0.1001)[0]
        if len(mask) > 0:
            # Substract the 'zero power' to get negative values for cooling power
            effect[mask, 1, 0] = x[mask, -1, self.power_column].clone() \
                                   - torch.from_numpy(self.zero_power).type(torch.FloatTensor).to(self.device)

            # Find the sequences in the batch that are heating
            heating = x[mask, 0, self.case_column] > 0.5
            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                effect[mask[heating], 0, 0] = effect[mask[heating], 0, 0].clone() \
                                                + self.a(effect[mask[heating], 1, 0].clone().reshape(-1, 1)).squeeze() \
                                                / self.a_scaling

            cooling = x[mask, 0, self.case_column] < 0.5
            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                effect[mask[cooling], 0, 0] = effect[mask[cooling], 0, 0].clone() \
                                                + self.d(effect[mask[cooling], 1, 0].clone().reshape(-1, 1)).squeeze() \
                                                / self.d_scaling

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, 0].clone()
        self.last_effect = effect[:, 0, 0].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        else:
            output =  torch.zeros(x.shape[0], 2, 1)
            output[:, 0, 0] = base[:, 0, 0]
            output[:, 1, 0] = effect[:, 0, 0]

        # Return the predictions and states of the model
        return output, (base_h, base_c)


class PCNN_mul_old(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            neigh_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.neigh_column = neigh_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0].values.astype(np.float32)[0]
        self.room_diff = normalization_variables['Room'][1].values.astype(np.float32)[0]
        self.out_min = normalization_variables['Out'][0].astype(np.float32)
        self.out_diff = normalization_variables['Out'][1].astype(np.float32)
        self.neigh_min = normalization_variables['Neigh'][0].values.astype(np.float32)
        self.neigh_diff = normalization_variables['Neigh'][1].values.astype(np.float32)

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a'][0]
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d'][0]

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative
        f = nn.Tanh()

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.Linear(1, 1, bias=False)
            self.b = nn.Linear(1, 1, bias=False)
            self.c = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(len(self.neigh_column))])
            self.d = nn.Linear(1, 1, bias=False)

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                for i in range(0, len(sizes) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
            if 'a.weight' == name:
                nn.init.constant_(param, 1.)
            if 'b.weight' == name:
                nn.init.constant_(param, 1.)
            if 'd.weight' == name:
                nn.init.constant_(param, 1.)
            if 'c.' in name:
                nn.init.constant_(param, 1.)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False, mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros(x.shape[0]).to(self.device)  ## D
            self.last_effect = torch.zeros(x.shape[0]).to(self.device)  ## E

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base.unsqueeze(1)
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_effect.unsqueeze(1)

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, 1)  ## D
        effect = torch.zeros(x.shape[0], 2, 1).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        base[:, 1, 0] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        effect[:, 0, 0] = self.last_effect.clone()

        ## Heat losses computation in 'E'
        if self.learn_discount_factors:
            # Loss to the outside is b*(T_k-T^out_k)
            effect[:, 0, 0] = effect[:, 0, 0].clone() - self.b((((x[:, -1, self.temperature_column].squeeze()
                                                                      + self.last_effect.clone() - 0.1) / 0.8
                                                                     * self.room_diff + self.room_min)
                                                                    - ((x[:, -1, self.out_column] - 0.1) / 0.8
                                                                       * self.out_diff + self.out_min)).
                                                                   reshape(-1, 1)).squeeze() / self.b_scaling
            # Loss to the neighboring zone is c*(T_k-T^neigh_k)
            for i in range(len(self.neigh_column)):
                effect[:, 0, 0] = effect[:, 0, 0].clone() - self.c[i]((((x[:, -1, self.temperature_column].squeeze()
                                                                          + self.last_effect.clone() - 0.1) / 0.8
                                                                         * self.room_diff + self.room_min)
                                                                        - ((x[:, -1, self.neigh_column[i]] - 0.1) / 0.8
                                                                           * self.neigh_diff[i] + self.neigh_min[i])).
                                                                       reshape(-1, 1)).squeeze() / self.c_scaling

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0, 0] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column].squeeze()
        else:
            # Store the outputs
            base[:, 0, 0] = lstm_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1, self.temperature_column].squeeze()

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column], axis=1).values > 0.1001)[0]
        if len(mask) > 0:
            # Substract the 'zero power' to get negative values for cooling power
            effect[mask, 1, 0] = x[mask, -1, self.power_column].clone() \
                                   - torch.from_numpy(self.zero_power).type(torch.FloatTensor).to(self.device)

            # Find the sequences in the batch that are heating
            heating = x[mask, 0, self.case_column] > 0.5
            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                effect[mask[heating], 0, 0] = effect[mask[heating], 0, 0].clone() \
                                                + self.a(effect[mask[heating], 1, 0].clone().reshape(-1, 1)).squeeze() \
                                                / self.a_scaling

            cooling = x[mask, 0, self.case_column] < 0.5
            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                effect[mask[cooling], 0, 0] = effect[mask[cooling], 0, 0].clone() \
                                                + self.d(effect[mask[cooling], 1, 0].clone().reshape(-1, 1)).squeeze() \
                                                / self.d_scaling

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, 0].clone()
        self.last_effect = effect[:, 0, 0].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        else:
            output =  torch.zeros(x.shape[0], 2, 1)
            output[:, 0, 0] = base[:, 0, 0]
            output[:, 1, 0] = effect[:, 0, 0]

        # Return the predictions and states of the model
        return output, (base_h, base_c)


class PCNN(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            neigh_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.neigh_column = neigh_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0].values.astype(np.float32)[0]
        self.room_diff = normalization_variables['Room'][1].values.astype(np.float32)[0]
        self.out_min = normalization_variables['Out'][0].astype(np.float32)
        self.out_diff = normalization_variables['Out'][1].astype(np.float32)
        self.neigh_min = normalization_variables['Neigh'][0].astype(np.float32)
        self.neigh_diff = normalization_variables['Neigh'][1].astype(np.float32)

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a'][0]
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d'][0]

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative
        f = nn.Tanh()

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = PositiveLinear(1, 1, require_bias=False)
            self.b = PositiveLinear(1, 1, require_bias=False)
            self.c = PositiveLinear(1, 1, require_bias=False)
            self.d = PositiveLinear(1, 1, require_bias=False)

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                for i in range(0, len(sizes) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                if 'a.' in name:
                    nn.init.constant_(param, 0.0)
                if 'b.' in name:
                    nn.init.constant_(param, 0.0)
                if 'd.' in name:
                    nn.init.constant_(param, 0.0)
                if 'c.' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False,
                mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros(x.shape[0]).to(self.device)  ## D
            self.last_effect = torch.zeros(x.shape[0]).to(self.device)  ## E

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base.unsqueeze(1)
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_effect.unsqueeze(
                1)

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, 1)  ## D
        effect = torch.zeros(x.shape[0], 2, 1).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        base[:, 1, 0] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        effect[:, 0, 0] = self.last_effect.clone()

        ## Heat losses computation in 'E'
        if self.learn_discount_factors:
            # Loss to the outside is b*(T_k-T^out_k)
            effect[:, 0, 0] = effect[:, 0, 0].clone() - self.b((((x[:, -1, self.temperature_column].squeeze()
                                                                  + self.last_effect.clone() - 0.1) / 0.8
                                                                 * self.room_diff + self.room_min)
                                                                - ((x[:, -1, self.out_column] - 0.1) / 0.8
                                                                   * self.out_diff + self.out_min)).
                                                               reshape(-1, 1)).squeeze() / self.b_scaling
            # Loss to the neighboring zone is c*(T_k-T^neigh_k)
            effect[:, 0, 0] = effect[:, 0, 0].clone() - self.c((((x[:, -1, self.temperature_column].squeeze()
                                                                  + self.last_effect.clone() - 0.1) / 0.8
                                                                 * self.room_diff + self.room_min)
                                                                - ((x[:, -1, self.neigh_column] - 0.1) / 0.8
                                                                   * self.neigh_diff + self.neigh_min).squeeze()).
                                                               reshape(-1, 1)).squeeze() / self.c_scaling

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0, 0] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column].squeeze()
        else:
            # Store the outputs
            base[:, 0, 0] = lstm_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1,
                                                                                   self.temperature_column].squeeze()

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column], axis=1).values > 0.1001)[0]
        if len(mask) > 0:
            # Substract the 'zero power' to get negative values for cooling power
            effect[mask, 1, 0] = x[mask, -1, self.power_column].clone() \
                                 - torch.from_numpy(self.zero_power).type(torch.FloatTensor).to(self.device)

            # Find the sequences in the batch that are heating
            heating = x[mask, 0, self.case_column] > 0.5
            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                effect[mask[heating], 0, 0] = effect[mask[heating], 0, 0].clone() \
                                              + self.a(effect[mask[heating], 1, 0].clone().reshape(-1, 1)).squeeze() \
                                              / self.a_scaling

            cooling = x[mask, 0, self.case_column] < 0.5
            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                effect[mask[cooling], 0, 0] = effect[mask[cooling], 0, 0].clone() \
                                              + self.d(effect[mask[cooling], 1, 0].clone().reshape(-1, 1)).squeeze() \
                                              / self.d_scaling

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, 0].clone()
        self.last_effect = effect[:, 0, 0].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :, :] = 0.
        else:
            output = torch.zeros(x.shape[0], 2, 1)
            output[:, 0, 0] = base[:, 0, 0]
            output[:, 1, 0] = effect[:, 0, 0]

        # Return the predictions and states of the model
        return output, (base_h, base_c)


class PCNN_old(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            neigh_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.neigh_column = neigh_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)

        # Define latent variables
        self.last_base = None  ## D
        self.last_effect = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0][0]
        self.room_diff = normalization_variables['Room'][1][0]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]
        self.neigh_min = normalization_variables['Neigh'][0]
        self.neigh_diff = normalization_variables['Neigh'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a'][0]
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d'][0]

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative
        f = nn.Tanh()

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.Linear(1, 1, bias=False)
            self.b = nn.Linear(1, 1, bias=False)
            self.c = nn.Linear(1, 1, bias=False)
            self.d = nn.Linear(1, 1, bias=False)

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                for i in range(0, len(sizes) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
            if 'a.weight' == name:
                nn.init.constant_(param, 1.)
            if 'b.weight' == name:
                nn.init.constant_(param, 1.)
            if 'c.weight' == name:
                nn.init.constant_(param, 1.)
            if 'd.weight' == name:
                nn.init.constant_(param, 1.)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False, mpc_mode: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros(x.shape[0]).to(self.device)  ## D
            self.last_effect = torch.zeros(x.shape[0]).to(self.device)  ## E

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base.unsqueeze(1)
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_effect.unsqueeze(1)

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2, 1)  ## D
        effect = torch.zeros(x.shape[0], 2, 1).to(self.device)  ## E

        # We need to take the true '0' power, which is not '0' since the data is normalized
        base[:, 1, 0] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        effect[:, 0, 0] = self.last_effect.clone()

        ## Heat losses computation in 'E'
        if self.learn_discount_factors:
            # Loss to the outside is b*(T_k-T^out_k)
            effect[:, 0, 0] = effect[:, 0, 0].clone() - self.b((((x[:, -1, [self.temperature_column]].squeeze()
                                                                      + self.last_effect.clone() - 0.1) / 0.8
                                                                     * self.room_diff + self.room_min)
                                                                    - ((x[:, -1, self.out_column] - 0.1) / 0.8
                                                                       * self.out_diff + self.out_min)).
                                                                   reshape(-1, 1)).squeeze() / self.b_scaling
            # Loss to the neighboring zone is c*(T_k-T^neigh_k)
            effect[:, 0, 0] = effect[:, 0, 0].clone() - self.c((((x[:, -1, [self.temperature_column]].squeeze()
                                                                      + self.last_effect.clone() - 0.1) / 0.8
                                                                     * self.room_diff + self.room_min)
                                                                    - ((x[:, -1, self.neigh_column] - 0.1) / 0.8
                                                                       * self.neigh_diff + self.neigh_min)).
                                                                   reshape(-1, 1)).squeeze() / self.c_scaling

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0, 0] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column].squeeze()
        else:
            # Store the outputs
            base[:, 0, 0] = lstm_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1, self.temperature_column].squeeze()

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column], axis=1).values > 0.1001)[0]
        if len(mask) > 0:
            # Substract the 'zero power' to get negative values for cooling power
            effect[mask, 1, 0] = x[mask, -1, self.power_column].clone() \
                                   - torch.from_numpy(self.zero_power).type(torch.FloatTensor).to(self.device)

            # Find the sequences in the batch that are heating
            heating = x[mask, 0, self.case_column] > 0.5
            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                effect[mask[heating], 0, 0] = effect[mask[heating], 0, 0].clone() \
                                                + self.a(effect[mask[heating], 1, 0].clone().reshape(-1, 1)).squeeze() \
                                                / self.a_scaling

            cooling = x[mask, 0, self.case_column] < 0.5
            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                effect[mask[cooling], 0, 0] = effect[mask[cooling], 0, 0].clone() \
                                                + self.d(effect[mask[cooling], 1, 0].clone().reshape(-1, 1)).squeeze() \
                                                / self.d_scaling

        # Recall 'D' and 'E' for the next time step
        self.last_base = base[:, 0, 0].clone()
        self.last_effect = effect[:, 0, 0].clone()

        # Final computation of the result 'T=D+E'
        if not mpc_mode:
            output = base + effect
            # Trick needed since some sequences are padded
            output[torch.where(x[:, -1, 0] < 1e-6)[0], :, :] = 0.
        else:
            output =  torch.zeros(x.shape[0], 2, 1)
            output[:, 0, 0] = base[:, 0, 0]
            output[:, 1, 0] = effect[:, 0, 0]

        # Return the predictions and states of the model
        return output, (base_h, base_c)


class PCNN_old(nn.Module):
    """
    PCNN model, from the paper
    `Physically Consistent Neural Networks for building models: theory and analysis`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            out_column: int,
            neigh_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.neigh_column = neigh_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)

        # Define latent variables
        self.last_base = None   ## D
        self.effect = None      ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]
        self.neigh_min = normalization_variables['Neigh'][0]
        self.neigh_diff = normalization_variables['Neigh'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = parameter_scalings['a']
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d']

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # We need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict differences that can be negative
        f = nn.Tanh()

        ## Initialization of the parameters of `E`
        if self.learn_discount_factors:
            self.a = nn.Linear(1, 1, bias=False)
            self.b = nn.Linear(1, 1, bias=False)
            self.c = nn.Linear(1, 1, bias=False)
            self.d = nn.Linear(1, 1, bias=False)

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                for i in range(0, len(sizes) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
            if 'a.weight' == name:
                nn.init.constant_(param, 1.)
            if 'b.weight' == name:
                nn.init.constant_(param, 1.)
            if 'c.weight' == name:
                nn.init.constant_(param, 1.)
            if 'd.weight' == name:
                nn.init.constant_(param, 1.)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (base_h, base_c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_base = torch.zeros(x.shape[0], 2).to(self.device)     ## D
            self.effect = torch.zeros(x.shape[0], 2).to(self.device)        ## E

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, [self.temperature_column]] = self.last_base[:, 0].unsqueeze(1)

        # Define the output of 'D' and 'E' for this time step
        base = torch.zeros(x.shape[0], 2)   ## D
        # We need to take the true '0' power, which is not '0' since the data is normalized
        base[:, 1] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        self.effect[:, 1] = 0.      ## E

        ## Heat losses computation in 'E'
        if self.learn_discount_factors:
            # Loss to the outside is b*(T-T_out)
            self.effect[:, 0] = self.effect[:, 0].clone() - self.b((((x[:, -1, [self.temperature_column]].squeeze()
                                                                        + self.effect[:, 0].clone() - 0.1) / 0.8
                                                                       * self.room_diff + self.room_min)
                                                                      - ((x[:, -1, self.out_column] - 0.1) / 0.8
                                                                         * self.out_diff + self.out_min)).
                                                                     reshape(-1, 1)).squeeze() / self.b_scaling
            # Loss to the neighboring zone is c*(T-T_neigh)
            self.effect[:, 0] = self.effect[:, 0].clone() - self.c((((x[:, -1, [self.temperature_column]].squeeze()
                                                                        + self.effect[:, 0].clone() - 0.1) / 0.8
                                                                       * self.room_diff + self.room_min)
                                                                      - ((x[:, -1, self.neigh_column] - 0.1) / 0.8
                                                                         * self.neigh_diff + self.neigh_min)).
                                                                     reshape(-1, 1)).squeeze() / self.c_scaling

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]
        else:
            # Store the outputs
            base[:, 0] = lstm_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        mask = torch.where(torch.max(x[:, :, self.valve_column], axis=1).values > 0.1001)[0]
        if len(mask) > 0:
            # Substract the 'zero power' to get negative values for cooling power
            self.effect[mask, 1] = x[mask, -1, self.power_column].clone() \
                                            - torch.from_numpy(self.zero_power).to(self.device)

            # Find the sequences in the batch that are heating
            heating = x[mask, 0, self.case_column] > 0.5
            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                self.effect[mask[heating], 0] = self.effect[mask[heating], 0].clone() \
                                       + self.a(self.effect[mask[heating], 1].clone().reshape(-1, 1)).squeeze() \
                                                / self.a_scaling

            cooling = x[mask, 0, self.case_column] < 0.5
            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                self.effect[mask[cooling], 0] = self.effect[mask[cooling], 0].clone() \
                                        + self.d(self.effect[mask[cooling], 1].clone().reshape(-1, 1)).squeeze() \
                                                / self.d_scaling

        # Final computation of the result 'T=D+E'
        output = base + self.effect

        # Trick needed since some sequences are padded
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.

        # Recall 'D' and 'E' for the next time step
        self.last_base[:, 0] = base[:, 0]
        self.last_base[:, 1] = output[:, 1]

        # Return the predictions and states of the model
        return output, (base_h, base_c)


class TemperaturePowerTogether(nn.Module):
    """
    Class representing the behavior of one room. A first NN can extract features, feed it through LSTMs,
    and the output might then be postprocess by NNs again. The network outputs the temperature and power
    consumption of the room together.

    Key: the network has three branches (modules): one base, one heating and one cooling module. The base
    module predicts the expected room temperature if no heating/cooling is provided. Then, the heating
    and cooling modules add the effect of heating or cooling on the temperature. Furthermore, they compute
    the associated power consumption. The trick is that Sigmoid activation are used to ensure the heating
    effect is positive and the cooling one negative.
    """

    def __init__(
        self,
        device,
        input_size: int,
        learn_initial_hidden_states: bool,
        feed_input_through_nn: bool,
        input_nn_hidden_sizes: list,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        layer_norm: bool,
        feed_output_through_nn: bool,
        output_nn_hidden_sizes: list,
        case_column: int,
        valve_column: int,
        zero_power: float,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            input_size:                 Input size of the model
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            zero_power:                Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.input_size = input_size
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.zero_power = zero_power

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        f = nn.Sigmoid() # nn.LeakyReLU(negative_slope=1e-6)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
            self.initial_heating_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_heating_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [self.input_size] + self.input_nn_hidden_sizes
            self.input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

        # Create the LSTMs of the 3 modules
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else self.input_size
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Make sure the output is of the right size if no NN is required at the output
        lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
        self.heating_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.cooling_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)
            self.heating_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
            self.cooling_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            # Heating and cooling modules predict temperature and power
            sizes = [lstm_hidden_size] + self.output_nn_hidden_sizes + [2]
            self.heating_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.cooling_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:     Input
            states: Original hidden and cell states if known (for all LSTMs, i.e. the base, the heating and the cooling)

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        # Clone the input since it will be manipulated
        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c) = states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                heating_h = torch.stack([self.initial_heating_h.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                heating_c = torch.stack([self.initial_heating_c.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                cooling_h = torch.stack([self.initial_cooling_h.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                cooling_c = torch.stack([self.initial_cooling_c.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
                heating_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                heating_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)

        # Check the shape of x
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Define the output of the 3 modules predicting temperature and power
        base = torch.zeros(x.shape[0], 2)
        # Put the power to the real zero according to the given factor
        base[:, 1] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        heating = torch.zeros(x.shape[0], 2).to(self.device)
        cooling = torch.zeros(x.shape[0], 2).to(self.device)

        # Input embedding step when needed
        embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
        if self.feed_input_through_nn:
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, :]
                for layer in self.input_nn:
                    temp = layer(temp)
                embedding[:, time_step, :] = temp
        else:
            embedding = x

        ## BASE MODULE
        ## Predict the bae temperature
        # LSTM
        lstm_output, (base_h, base_c) = self.base_lstm(embedding, (base_h, base_c))

        # Normalization when required
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0] = temp.squeeze()

        else:
            # Store the outputs
            base[:, 0] = lstm_output[:, -1, 0]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]

        if len(heating_cases) > 0:
            # Define the inputs for the heating cases
            heating_input = embedding[heating_cases, :, :]
            # Get the sequences where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[heating_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :]) = self.heating_lstm(
                heating_input, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :])
            )

            # Normalization if wanted
            if self.layer_norm:
                lstm_output = self.heating_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Put the data is the form needed for the neural net
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn:
                        temp = layer(temp)
                    heating[heating_cases[mask], :] = temp

                else:
                    # Store the outputs
                    heating[heating_cases[mask], :] = lstm_output[mask, -1, :]

        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]

        if len(cooling_cases) > 0:
            # Define the inputs for the heating cases, and tensors that will retain the heating effect,
            # the hidden states and cell states
            cooling_input = embedding[cooling_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[cooling_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :]) = self.cooling_lstm(
                cooling_input, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :])
            )

            # Normalization when required
            if self.layer_norm:
                lstm_output = self.cooling_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Put the data is the form needed for the neural net
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn:
                        temp = layer(temp)
                    cooling[cooling_cases[mask], :] = temp

                else:
                    # Store the outputs
                    cooling[cooling_cases[mask], :] = lstm_output[mask, -1, :]

        # Final computation of the result
        output = base + heating - cooling

        # Trick needed since some sequences are padded: this ensures the error doesn't explode for nothing
        # when sequences are padded with zero (i.e. avoids the network to try to predict the zeroes that
        # don't acutally exist
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.

        # Return all the network's outputs
        return output, (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c)


class SeparateTemperaturePower(nn.Module):
    """
    Class similar to `TemperaturePowerTogether` above. The difference is that each module (base, heating,
    cooling) is branched after the LSTMs to separately predict the temperature and the power. (The base
    module doens't need to predict the power, so it only has one output branch)
    """

    def __init__(
        self,
        device,
        input_size: int,
        learn_initial_hidden_states: bool,
        feed_input_through_nn: bool,
        input_nn_hidden_sizes: list,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        layer_norm: bool,
        feed_output_through_nn: bool,
        output_nn_hidden_sizes: list,
        case_column: int,
        valve_column: int,
        zero_power: float,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            input_size:                 Input size of the model
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.input_size = input_size
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.zero_power = zero_power

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        f = nn.Sigmoid() # nn.LeakyReLU(negative_slope=1e-6)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
            self.initial_heating_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_heating_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [self.input_size] + self.input_nn_hidden_sizes
            self.input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

        # Create the LSTMs of the 3 modules
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else self.input_size
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Make sure the output is of the right size if no NN is required at the output
        lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
        self.heating_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.cooling_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wwanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)
            self.heating_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
            self.cooling_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            # Heating and cooling modules predict temperature and power
            sizes = [lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.heating_output_nn_temperature = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.heating_output_nn_power = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])
            self.cooling_output_nn_temperature = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.cooling_output_nn_power = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:      Input
            states: Original hidden and cell states if known (for all LSTMs, i.e. the base, the heating and the cooling)

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        # Clone the input before manipulations
        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c) = states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                heating_h = torch.stack([self.initial_heating_h.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                heating_c = torch.stack([self.initial_heating_c.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                cooling_h = torch.stack([self.initial_cooling_h.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                cooling_c = torch.stack([self.initial_cooling_c.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
                heating_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                heating_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)

        # Ensure consistent shape of the input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Define the output of the 3 modules predicting temperature and power, again putting the
        # power to the scaled zero
        base = torch.zeros(x.shape[0], 2)
        base[:, 1] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        heating = torch.zeros(x.shape[0], 2).to(self.device)
        cooling = torch.zeros(x.shape[0], 2).to(self.device)

        # Input embedding step when needed
        embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
        if self.feed_input_through_nn:
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, :]
                for layer in self.input_nn:
                    temp = layer(temp)
                embedding[:, time_step, :] = temp
        else:
            embedding = x

        ## BASE MODULE
        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(embedding, (base_h, base_c))

        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0] = temp.squeeze()

        else:
            # Store the outputs
            base[:, 0] = lstm_output[:, -1, 0]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]

        if len(heating_cases) > 0:
            # Define the inputs for the heating cases
            heating_input = embedding[heating_cases, :, :]
            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[heating_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :]) = self.heating_lstm(
                heating_input, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.heating_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Temperature effect
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn_temperature:
                        temp = layer(temp)
                    heating[heating_cases[mask], 0] = temp.squeeze()
                    # Power computation
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn_power:
                        temp = layer(temp)
                    heating[heating_cases[mask], 1] = temp.squeeze() * x[heating_cases[mask], -1, self.valve_column].squeeze()

                else:
                    # Store the outputs
                    heating[heating_cases[mask], :] = lstm_output[mask, -1, :]

        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]

        if len(cooling_cases) > 0:
            # Define the inputs for the heating cases, and tensors that will retain the heating effect,
            # the hidden states and cell states
            cooling_input = embedding[cooling_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[cooling_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :]) = self.cooling_lstm(
                cooling_input, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.cooling_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Temperature effect
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn_temperature:
                        temp = layer(temp)
                    cooling[cooling_cases[mask], 0] = temp.squeeze()
                    # Power computation
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn_power:
                        temp = layer(temp)
                    cooling[cooling_cases[mask], 1] = temp.squeeze() * x[cooling_cases[mask], -1, self.valve_column].squeeze()

                else:
                    # Store the outputs
                    cooling[cooling_cases[mask], :] = lstm_output[mask, -1, :]

        # Final computation of the result
        output = base + heating - cooling

        # Trick needed since some sequences are padded as before to avoid the network predicting the
        # padding zeroes that don't exist
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.

        # Return all the network's outputs
        return output, (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c)


class SeparateInputs(nn.Module):
    """
    Class similar to `SeparateTemperaturePower`, but the input of the base temperature module is different
    from the one of both the heating and cooling modules. This reflects the fact that the temperature
    might be influenced by oustide conditions for example, which don't impact the heating system. Reversely,
    the heating/cooling effect depend on the valve openings.
    SO, the main difference is that we now have two different input embeddings, one for the base and one
    for the other two modules.
    """

    def __init__(
        self,
        device,
        base_indices: list,
        effect_indices: list,
        learn_initial_hidden_states: bool,
        feed_input_through_nn: bool,
        input_nn_hidden_sizes: list,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        layer_norm: bool,
        feed_output_through_nn: bool,
        output_nn_hidden_sizes: list,
        case_column: int,
        valve_column: int,
        zero_power: float,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            effect_indices:             Input features used to predict the heating and cooling components
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.zero_power = zero_power

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        f = nn.Sigmoid() # nn.LeakyReLU(negative_slope=1e-6)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
            self.initial_heating_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_heating_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            sizes = [len(self.effect_indices)] + self.input_nn_hidden_sizes
            self.effect_input_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Create the LSTMs of the 3 modules
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Make sure the output is of the right size if no NN is required at the output
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.effect_indices)
        lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
        self.heating_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.cooling_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wwanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)
            self.heating_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
            self.cooling_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            # Heating and cooling modules predict temperature and power
            sizes = [lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            f = nn.Sigmoid()  # nn.LeakyReLU(negative_slope=1e-6)
            self.heating_output_nn_temperature = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.heating_output_nn_power = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])
            self.cooling_output_nn_temperature = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.cooling_output_nn_power = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:      Input
            states: Original hidden and cell states if known (for all LSTMs, i.e. the base, the heating and the cooling)

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c) = states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                heating_h = torch.stack([self.initial_heating_h.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                heating_c = torch.stack([self.initial_heating_c.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                cooling_h = torch.stack([self.initial_cooling_h.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                cooling_c = torch.stack([self.initial_cooling_c.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
                heating_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                heating_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Define the output of the 3 modules predicting temperature and power, putting again the power
        # to the given scaled zero
        base = torch.zeros(x.shape[0], 2)
        base[:, 1] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        heating = torch.zeros(x.shape[0], 2).to(self.device)
        cooling = torch.zeros(x.shape[0], 2).to(self.device)

        # Input embedding step when needed
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            effect_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            # Loop over all the input time steps to encode each one using the feature extracting NN, both for
            # the base and the effect modules
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
                temp = x[:, time_step, self.effect_indices]
                for layer in self.effect_input_nn:
                    temp = layer(temp)
                effect_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]
            effect_embedding = x[:, :, self.effect_indices]

        ## BASE MODULE
        # LSTM prediction for the base temperature

        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0] = temp.squeeze()

        else:
            # Store the outputs
            base[:, 0] = lstm_output[:, -1, 0]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]

        if len(heating_cases) > 0:
            # Define the inputs for the heating cases
            heating_input = effect_embedding[heating_cases, :, :]
            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[heating_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :]) = self.heating_lstm(
                heating_input, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.heating_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Temperature effect
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn_temperature:
                        temp = layer(temp)
                    heating[heating_cases[mask], 0] = temp.squeeze()
                    # Power effect
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn_power:
                        temp = layer(temp)
                    heating[heating_cases[mask], 1] = temp.squeeze() #* x[heating_cases[mask], -1, self.valve_column].squeeze()

                else:
                    # Store the outputs
                    heating[heating_cases[mask], :] = lstm_output[mask, -1, :]

        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]

        if len(cooling_cases) > 0:
            # Define the inputs for the heating cases, and tensors that will retain the heating effect,
            # the hidden states and cell states
            cooling_input = effect_embedding[cooling_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[cooling_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :]) = self.cooling_lstm(
                cooling_input, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.cooling_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Temperature effect
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn_temperature:
                        temp = layer(temp)
                    cooling[cooling_cases[mask], 0] = temp.squeeze()
                    # Power computation
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn_power:
                        temp = layer(temp)
                    cooling[cooling_cases[mask], 1] = temp.squeeze() #* x[cooling_cases[mask], -1, self.valve_column].squeeze()

                else:
                    # Store the outputs
                    cooling[cooling_cases[mask], :] = lstm_output[mask, -1, :]

        # Final computation of the result
        output = base + heating - cooling

        # Trick needed since some sequences are padded
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.

        # Return all the network's outputs
        return output, (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c)


class DiscountedModules(nn.Module):
    """
    Class similar to `SeparateInputs`, but now the heating and cooling effect on the temperature
    are kept in memory from one time step to another. This reflects the fact that heating the room
    at a given time step t will still have effects on that room in future time steps.

    In this class, the heating and cooling modules' outputs are kept in memory, and they are discounted at each
    time step by a learned factor (initialized at 0.95) and the heating/cooling module then
    adds/subtracts an additional effect coming from the current time step.
    """

    def __init__(
        self,
        device,
        base_indices: list,
        effect_indices: list,
        learn_discount_factors: bool,
        learn_initial_hidden_states: bool,
        feed_input_through_nn: bool,
        input_nn_hidden_sizes: list,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        layer_norm: bool,
        feed_output_through_nn: bool,
        output_nn_hidden_sizes: list,
        case_column: int,
        valve_column: int,
        zero_power: float,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            effect_indices:             Input features used to predict the heating and cooling components
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.zero_power = zero_power

        # To retain the heating and cooling effect
        self.heating = None
        self.cooling = None

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        f = nn.Sigmoid() #nn.LeakyReLU(negative_slope=1e-6)

        if self.learn_discount_factors:
            self.discount_factor_heating = nn.Parameter(data=torch.zeros(1) + 0.95)
            self.discount_factor_cooling = nn.Parameter(data=torch.zeros(1) + 0.95)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
            self.initial_heating_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_heating_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            sizes = [len(self.effect_indices)] + self.input_nn_hidden_sizes
            self.effect_input_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Create the LSTMs of the 3 modules
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Make sure the output is of the right size if no NN is required at the output
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.effect_indices)
        lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
        self.heating_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.cooling_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wwanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)
            self.heating_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
            self.cooling_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            # Heating and cooling modules predict temperature and power
            sizes = [lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.heating_output_nn_temperature = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.heating_output_nn_power = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])
            self.cooling_output_nn_temperature = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.cooling_output_nn_power = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:      Input
            states: Original hidden and cell states if known (for all LSTMs, i.e. the base, the heating and the cooling)

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c) = states
        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                heating_h = torch.stack([self.initial_heating_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                heating_c = torch.stack([self.initial_heating_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                cooling_h = torch.stack([self.initial_cooling_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                cooling_c = torch.stack([self.initial_cooling_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
                heating_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                heating_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)

            # initialize the effects at zero at the start of the trajectories
            self.heating = torch.zeros(x.shape[0], 2).to(self.device)
            self.cooling = torch.zeros(x.shape[0], 2).to(self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Define the output of the 3 modules predicting temperature and power, putting the latter at the
        # true scaled given zero
        base = torch.zeros(x.shape[0], 2)
        base[:, 1] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)

        # The power consumption is put to zero (it is independent from the power consumption of th
        # previous steps)
        self.heating[:, 1] = 0.
        self.cooling[:, 1] = 0.

        # Discount the effects stored from previous time steps
        if self.learn_discount_factors:
            self.heating[:, 0] = self.discount_factor_heating * self.heating[:, 0].clone()
            self.cooling[:, 0] = self.discount_factor_cooling * self.cooling[:, 0].clone()

        # Input embedding step when needed
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            effect_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)

            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
                temp = x[:, time_step, self.effect_indices]
                for layer in self.effect_input_nn:
                    temp = layer(temp)
                effect_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]
            effect_embedding = x[:, :, self.effect_indices]

        ## BASE MODULE
        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0] = temp.squeeze()

        else:
            # Store the outputs
            base[:, 0] = lstm_output[:, -1, 0]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]

        if len(heating_cases) > 0:
            # Define the inputs for the heating cases
            heating_input = effect_embedding[heating_cases, :, :]
            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[heating_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :]) = self.heating_lstm(
                heating_input, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.heating_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Add the effect of heating at the current time step to the previous one for the temperature
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn_temperature:
                        temp = layer(temp)
                    self.heating[heating_cases[mask], 0] += temp.squeeze()
                    # Compute the effect of heating at the current time step to the previous one for the power
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn_power:
                        temp = layer(temp)
                    self.heating[heating_cases[mask], 1] = temp.squeeze()

                else:
                    # Store the outputs
                    self.heating[heating_cases[mask], 0] += lstm_output[mask, -1, 0]
                    self.heating[heating_cases[mask], 1] = lstm_output[mask, -1, 1]

        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]

        if len(cooling_cases) > 0:
            # Define the inputs for the heating cases, and tensors that will retain the heating effect,
            # the hidden states and cell states
            cooling_input = effect_embedding[cooling_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[cooling_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :]) = self.cooling_lstm(
                cooling_input, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.cooling_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Temperature effect
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn_temperature:
                        temp = layer(temp)
                    self.cooling[cooling_cases[mask], 0] += temp.squeeze()
                    # Power computation
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn_power:
                        temp = layer(temp)
                    self.cooling[cooling_cases[mask], 1] = temp.squeeze()

                else:
                    # Store the outputs
                    self.cooling[cooling_cases[mask], 0] += lstm_output[mask, -1, 0]
                    self.cooling[cooling_cases[mask], 1] = lstm_output[mask, -1, 1]

        # Final computation of the result
        output = base + self.heating - self.cooling

        # Trick needed since some sequences are padded
        output[torch.where(x[:,-1,0] < 1e-6)[0],:] = 0.

        # Return all the network's outputs
        return output, (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c)


class DiscountedModulesBase(nn.Module):
    """
    Class similar to `DiscountedModules`, but this time the base temperature prediction is used as input
    to the model instead of the final one. That way, we ensure the base prediction to always be the same
    irrespective of the heating/cooling pattern, which is the case in reality.
    """

    def __init__(
        self,
        device,
        base_indices: list,
        effect_indices: list,
        learn_discount_factors: bool,
        learn_initial_hidden_states: bool,
        feed_input_through_nn: bool,
        input_nn_hidden_sizes: list,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        layer_norm: bool,
        feed_output_through_nn: bool,
        output_nn_hidden_sizes: list,
        case_column: int,
        valve_column: int,
        temperature_column: int,
        power_column: int,
        zero_power: float,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            effect_indices:             Input features used to predict the heating and cooling components
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.zero_power = zero_power

        # Last base will be used to keep track of the base temperature prediction
        self.last_base = None
        self.heating = None
        self.cooling = None

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        f = nn.Sigmoid() #nn.LeakyReLU(negative_slope=1e-6)

        if self.learn_discount_factors:
            self.discount_factor_heating = nn.Parameter(data=torch.zeros(1) + 0.95)
            self.discount_factor_cooling = nn.Parameter(data=torch.zeros(1) + 0.95)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
            self.initial_heating_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_heating_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            sizes = [len(self.effect_indices)] + self.input_nn_hidden_sizes
            self.effect_input_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Create the LSTMs of the 3 modules
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Make sure the output is of the right size if no NN is required at the output
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.effect_indices)
        lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
        self.heating_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.cooling_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wwanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)
            self.heating_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
            self.cooling_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            # Heating and cooling modules predict temperature and power
            sizes = [lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.heating_output_nn_temperature = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.heating_output_nn_power = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])
            self.cooling_output_nn_temperature = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.cooling_output_nn_power = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            wamr_start: Whether we are warm starting the model

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c) = states

        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                heating_h = torch.stack([self.initial_heating_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                heating_c = torch.stack([self.initial_heating_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                cooling_h = torch.stack([self.initial_cooling_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                cooling_c = torch.stack([self.initial_cooling_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
                heating_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                heating_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)

            # Initialize everything to zero
            self.last_base = torch.zeros(x.shape[0], 2).to(self.device)
            self.heating = torch.zeros(x.shape[0], 2).to(self.device)
            self.cooling = torch.zeros(x.shape[0], 2).to(self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # If we are not warm starting (i.e. using real temperatures and power as input), we will use
        # the ones predicted as the last time step
        if not warm_start:
            x[:, -1, [self.temperature_column, self.power_column]] = self.last_base

        # Define the output of the 3 modules predicting temperature and power, put the power
        # to the scaled given zero
        base = torch.zeros(x.shape[0], 2)
        base[:, 1] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)

        # The power is independent from the last step
        self.heating[:, 1] = 0.
        self.cooling[:, 1] = 0.

        # Discount the accumulated effect of the previous time steps
        if self.learn_discount_factors:
            self.heating[:, 0] = self.discount_factor_heating * self.heating[:, 0].clone()
            self.cooling[:, 0] = self.discount_factor_cooling * self.cooling[:, 0].clone()

        # Input embedding step when needed
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            effect_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)

            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
                temp = x[:, time_step, self.effect_indices]
                for layer in self.effect_input_nn:
                    temp = layer(temp)
                effect_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]
            effect_embedding = x[:, :, self.effect_indices]

        ## BASE MODULE
        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0] = temp.squeeze()

        else:
            # Store the outputs
            base[:, 0] = lstm_output[:, -1, 0]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]

        if len(heating_cases) > 0:
            # Define the inputs for the heating cases
            heating_input = effect_embedding[heating_cases, :, :]
            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[heating_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :]) = self.heating_lstm(
                heating_input, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.heating_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Temperature effect
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn_temperature:
                        temp = layer(temp)
                    self.heating[heating_cases[mask], 0] += temp.squeeze()
                    # Power computation
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn_power:
                        temp = layer(temp)
                    self.heating[heating_cases[mask], 1] = temp.squeeze()

                else:
                    # Store the outputs
                    self.heating[heating_cases[mask], 0] += lstm_output[mask, -1, 0]
                    self.heating[heating_cases[mask], 1] = lstm_output[mask, -1, 1]

        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]

        if len(cooling_cases) > 0:
            # Define the inputs for the heating cases, and tensors that will retain the heating effect,
            # the hidden states and cell states
            cooling_input = effect_embedding[cooling_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[cooling_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :]) = self.cooling_lstm(
                cooling_input, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.cooling_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Temperature effect
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn_temperature:
                        temp = layer(temp)
                    self.cooling[cooling_cases[mask], 0] += temp.squeeze()
                    # Power computation
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn_power:
                        temp = layer(temp)
                    self.cooling[cooling_cases[mask], 1] = temp.squeeze()

                else:
                    # Store the outputs
                    self.cooling[cooling_cases[mask], 0] += lstm_output[mask, -1, 0]
                    self.cooling[cooling_cases[mask], 1] = lstm_output[mask, -1, 1]

        # Final computation of the result
        output = base + self.heating - self.cooling

        # Trick needed since some sequences are padded
        output[torch.where(x[:,-1,0] < 1e-6)[0],:] = 0.

        # Store the base temperature prediction and the power output for the next step
        self.last_base[:, 0] = base[:, 0]
        self.last_base[:, 1] = output[:, 1]

        # Return all the network's outputs
        return output, (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c)


class ResDiscountedModulesBase(nn.Module):
    """
    Class similar to `DiscountedModulesBase`, but with a ResNet type of architecture, i.e. the NNs
    actually predict differences in temperature, which is added to the input temperature before
    the model returns it.
    Predictions for the power consumption don't change.
    """

    def __init__(
        self,
        device,
        base_indices: list,
        effect_indices: list,
        learn_discount_factors: bool,
        learn_initial_hidden_states: bool,
        feed_input_through_nn: bool,
        input_nn_hidden_sizes: list,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        layer_norm: bool,
        feed_output_through_nn: bool,
        output_nn_hidden_sizes: list,
        case_column: int,
        valve_column: int,
        temperature_column: int,
        power_column: int,
        zero_power: float,
        base_division_factor: list,
        cooling_division_factor: list,
        heating_division_factor: list,
        use_energy_prediction: bool
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            effect_indices:             Input features used to predict the heating and cooling components
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            cooling_division_factor:    Factors to scale the heating predictions, similar to `base_division_factor`
            heating_division_factor:    Factors to scale the cooling predictions, similar to `base_division_factor`
            use_energy_prediction:      Flag to set to `True` if want to use the energy prediction from the last step
                                          as input for the current prediction (`False` for physics-based models)
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.zero_power = zero_power
        self.use_energy_prediction = use_energy_prediction
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.cooling_division_factor = torch.FloatTensor(cooling_division_factor)
        self.heating_division_factor = torch.FloatTensor(heating_division_factor)

        self.last_base = None
        self.heating = None
        self.cooling = None

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        # We now need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict diffenrences that can be negative
        f = nn.Tanh()

        if self.learn_discount_factors:
            self.discount_factor_heating = nn.Parameter(data=torch.zeros(1) + 0.95)
            self.discount_factor_cooling = nn.Parameter(data=torch.zeros(1) + 0.95)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
            self.initial_heating_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_heating_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            sizes = [len(self.effect_indices)] + self.input_nn_hidden_sizes
            self.effect_input_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Create the LSTMs of the 3 modules
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Make sure the output is of the right size if no NN is required at the output
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.effect_indices)
        lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
        self.heating_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.cooling_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wwanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)
            self.heating_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
            self.cooling_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            # Heating and cooling modules predict temperature and power
            sizes = [lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            # Here we now need a non negative activation to ensure the positivity of the heating effect
            # or the negativity of the cooling one
            f = nn.Sigmoid()
            self.heating_output_nn_temperature = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.heating_output_nn_power = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])
            self.cooling_output_nn_temperature = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
            self.cooling_output_nn_power = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            wamr_start: Whether we are warm starting the model

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c) = states

        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                heating_h = torch.stack([self.initial_heating_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                heating_c = torch.stack([self.initial_heating_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                cooling_h = torch.stack([self.initial_cooling_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                cooling_c = torch.stack([self.initial_cooling_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
                heating_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                heating_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)

            self.last_base = torch.zeros(x.shape[0], 2).to(self.device)
            self.heating = torch.zeros(x.shape[0], 2).to(self.device)
            self.cooling = torch.zeros(x.shape[0], 2).to(self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Again, use the last base predictions as input when not warm starting
        if not warm_start:
            x[:, -1, [self.temperature_column]] = self.last_base[:, 0].unsqueeze(1)
            if self.use_energy_prediction:
                x[:, -1, [self.temperature_column, self.power_column]] = self.last_base

        # Define the output of the 3 modules predicting temperature and power
        base = torch.zeros(x.shape[0], 2)
        base[:, 1] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        self.heating[:, 1] = 0.
        self.cooling[:, 1] = 0.

        # Discount previous effects
        if self.learn_discount_factors:
            self.heating[:, 0] = self.discount_factor_heating * self.heating[:, 0].clone()
            self.cooling[:, 0] = self.discount_factor_cooling * self.cooling[:, 0].clone()

        # Input embedding step when needed
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            effect_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)

            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
                temp = x[:, time_step, self.effect_indices]
                for layer in self.effect_input_nn:
                    temp = layer(temp)
                effect_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]
            effect_embedding = x[:, :, self.effect_indices]

        ## BASE MODULE
        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]
        else:
            # Store the outputs
            base[:, 0] = lstm_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]

        if len(heating_cases) > 0:
            # Define the inputs for the heating cases
            heating_input = effect_embedding[heating_cases, :, :]
            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[heating_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :]) = self.heating_lstm(
                heating_input, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.heating_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Temperature effect
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn_temperature:
                        temp = layer(temp)
                    self.heating[heating_cases[mask], 0] += temp.squeeze() / self.heating_division_factor[0]
                    # Power computation
                    temp = lstm_output[mask, -1, :]
                    for layer in self.heating_output_nn_power:
                        temp = layer(temp)
                    self.heating[heating_cases[mask], 1] = temp.squeeze()

                else:
                    # Store the outputs
                    self.heating[heating_cases[mask], 0] += lstm_output[mask, -1, 0] / self.heating_division_factor[0]
                    self.heating[heating_cases[mask], 1] = lstm_output[mask, -1, 1]

        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]

        if len(cooling_cases) > 0:
            # Define the inputs for the heating cases, and tensors that will retain the heating effect,
            # the hidden states and cell states
            cooling_input = effect_embedding[cooling_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[cooling_cases, :, self.valve_column], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :]) = self.cooling_lstm(
                cooling_input, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.cooling_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.feed_output_through_nn:
                    # Temperature effect
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn_temperature:
                        temp = layer(temp)
                    self.cooling[cooling_cases[mask], 0] += temp.squeeze() / self.cooling_division_factor[0]
                    # Power computation
                    temp = lstm_output[mask, -1, :]
                    for layer in self.cooling_output_nn_power:
                        temp = layer(temp)
                    self.cooling[cooling_cases[mask], 1] = temp.squeeze()

                else:
                    # Store the outputs
                    self.cooling[cooling_cases[mask], 0] += lstm_output[mask, -1, 0] / self.cooling_division_factor[0]
                    self.cooling[cooling_cases[mask], 1] = lstm_output[mask, -1, 1]

        # Final computation of the result
        output = base + self.heating - self.cooling

        # Trick needed since some sequences are padded
        output[torch.where(x[:,-1,0] < 1e-6)[0],:] = 0.

        # Recall the current base temperature and the power predictions
        self.last_base[:, 0] = base[:, 0]
        self.last_base[:, 1] = output[:, 1]

        # Return all the network's outputs
        return output, (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c)


class ResDiscountedModulesBaseTrueEnergy(nn.Module):
    """
    Class similar to `DiscountedModulesBase`, but with a ResNet type of architecture, i.e. the NNs
    actually predict differences in temperature, which is added to the input temperature before
    the model returns it.
    Predictions for the power consumption don't change.
    """

    def __init__(
        self,
        device,
        base_indices: list,
        effect_indices: list,
        learn_discount_factors: bool,
        learn_initial_hidden_states: bool,
        feed_input_through_nn: bool,
        input_nn_hidden_sizes: list,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        layer_norm: bool,
        feed_output_through_nn: bool,
        output_nn_hidden_sizes: list,
        case_column: int,
        valve_column: int,
        temperature_column: int,
        power_column: int,
        zero_power: float,
        base_division_factor: list,
        cooling_division_factor: list,
        heating_division_factor: list,
        use_energy_prediction: bool
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            effect_indices:             Input features used to predict the heating and cooling components
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            cooling_division_factor:    Factors to scale the heating predictions, similar to `base_division_factor`
            heating_division_factor:    Factors to scale the cooling predictions, similar to `base_division_factor`
            use_energy_prediction:      Flag to set to `True` if want to use the energy prediction from the last step
                                          as input for the current prediction (`False` for physics-based models)
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        #self.base_indices = base_indices
        self.base_indices = base_indices[1:]
        print('Achtung')
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.zero_power = zero_power
        self.use_energy_prediction = use_energy_prediction
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.cooling_division_factor = torch.FloatTensor(cooling_division_factor)
        self.heating_division_factor = torch.FloatTensor(heating_division_factor)

        self.last_base = None
        self.heating = None
        self.cooling = None

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        # We now need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict diffenrences that can be negative
        f = nn.Tanh()

        if self.learn_discount_factors:
            self.discount_factor_heating = nn.Parameter(data=torch.zeros(1) + 10000)#250000
            self.discount_factor_cooling = nn.Parameter(data=torch.zeros(1) + 10000)#250000
            self.cooling_division_factor = nn.Parameter(data=torch.zeros(1) + 70)#30
            self.heating_division_factor = nn.Parameter(data=torch.zeros(1) + 70)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                for i in range(0, len(sizes) - 1)])
            if self.use_energy_prediction:
                sizes = [len(self.effect_indices)] + self.input_nn_hidden_sizes
                self.effect_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                      for i in range(0, len(sizes) - 1)])

        # Create the LSTMs of the 3 modules
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            if self.use_energy_prediction:
                # Heating and cooling modules predict temperature and power
                sizes = [self.input_nn_hidden_sizes[-1]] + self.output_nn_hidden_sizes + [1]
                # Here we now need a non negative activation to ensure the positivity of the heating effect
                # or the negativity of the cooling one
                f = nn.Sigmoid()
                self.heating_output_nn_power = nn.ModuleList(
                    [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                     range(0, len(sizes) - 1)])
                self.cooling_output_nn_power = nn.ModuleList(
                    [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                     range(0, len(sizes) - 1)])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            wamr_start: Whether we are warm starting the model

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c) = states

        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            self.last_base = torch.zeros(x.shape[0], 2).to(self.device)
            self.heating = torch.zeros(x.shape[0], 2).to(self.device)
            self.cooling = torch.zeros(x.shape[0], 2).to(self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Again, use the last base predictions as input when not warm starting
        if not warm_start:
            if self.use_energy_prediction:
                x[:, -1, [self.temperature_column, self.power_column]] = self.last_base
            else:
                x[:, -1, [self.temperature_column]] = self.last_base[:, 0].unsqueeze(1)

        # Define the output of the 3 modules predicting temperature and power
        base = torch.zeros(x.shape[0], 2)
        base[:, 1] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        self.heating[:, 1] = 0.
        self.cooling[:, 1] = 0.

        # Discount previous effects
        if self.learn_discount_factors:
            mask = (x[:, -1, 0] > 0.05) & (x[:, 0, self.case_column] > 0.5)
            #self.heating[mask, 0] = (1. - 1. / self.discount_factor_heating / x[mask, -1, 0]) \
            #                        * self.heating[mask, 0].clone()

            self.heating[mask, 0] = self.heating[mask, 0] - (1. / self.discount_factor_heating
                                                             * (((x[mask, -1, [self.temperature_column]]
                                                                + self.heating[mask, 0].clone() - 0.1) / 0.8
                                                             * 12.964820 + 18.181446) - ((x[mask, -1, 0] - 0.1)
                                                             / 0.8 * 46.631382 - 9.936853)))

            mask = (x[:, -1, 0] > 0.05) & (x[:, 0, self.case_column] < 0.5)

            self.cooling[mask, 0] = self.cooling[mask, 0] - (1. / self.discount_factor_cooling
                                                             * (((x[mask, -1, 0] - 0.1) / 0.8 * 46.631382 - 9.936853)
                                                             - ((x[mask, -1, [self.temperature_column]]
                                                                - self.cooling[mask, 0].clone() - 0.1) / 0.8
                                                             * 12.964820 + 18.181446)))

            #self.cooling[mask, 0] = (1. - 1. / self.discount_factor_cooling * x[mask, -1, 0]) \
            #                       * self.cooling[mask, 0].clone()

        # Input embedding step when needed
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            if self.use_energy_prediction:
                effect_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)

            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
                if self.use_energy_prediction:
                    temp = x[:, time_step, self.effect_indices]
                    for layer in self.effect_input_nn:
                        temp = layer(temp)
                    effect_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]
            if self.use_energy_prediction:
                effect_embedding = x[:, :, self.effect_indices]

        ## BASE MODULE
        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]
        else:
            # Store the outputs
            base[:, 0] = lstm_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]
        if len(heating_cases) > 0:
            # Define the inputs for the heating cases
            if self.use_energy_prediction:
                heating_input = effect_embedding[heating_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[heating_cases, :, self.valve_column], axis=1).values > 0.1001

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.use_energy_prediction:
                    if self.feed_output_through_nn:
                        # Power computation
                        temp = heating_input[mask, -1, :]
                        for layer in self.heating_output_nn_power:
                            temp = layer(temp)
                        self.heating[heating_cases[mask], 1] = temp.squeeze()
                    else:
                        self.heating[heating_cases[mask], 1] = heating_input[mask, -1, 1]
                else:
                    self.heating[heating_cases[mask], 1] = x[heating_cases[mask], -1, self.power_column].clone() \
                                                           - torch.from_numpy(self.zero_power).to(self.device)

                # Temperature effect
                self.heating[heating_cases[mask], 0] = self.heating[heating_cases[mask], 0].clone() \
                                                       + self.heating[heating_cases[mask], 1].clone() \
                                                       / self.heating_division_factor
        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]
        if len(cooling_cases) > 0:
            # Define the inputs for the heating cases, and tensors that will retain the heating effect,
            # the hidden states and cell states
            if self.use_energy_prediction:
                cooling_input = effect_embedding[cooling_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[cooling_cases, :, self.valve_column], axis=1).values > 0.1001

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.use_energy_prediction:
                    if self.feed_output_through_nn:
                        # Power computation
                        temp = cooling_input[mask, -1, :]
                        for layer in self.cooling_output_nn_power:
                            temp = layer(temp)
                        self.cooling[cooling_cases[mask], 1] = temp.squeeze()
                        raise NotImplementedError('Check that!')
                    else:
                        self.cooling[cooling_cases[mask], 1] = cooling_input[mask, -1, 1]
                else:
                    self.cooling[cooling_cases[mask], 1] = torch.from_numpy(self.zero_power).to(self.device) \
                                                           - x[cooling_cases[mask], -1, self.power_column].clone()

                # Temperature effect
                self.cooling[cooling_cases[mask], 0] = self.cooling[cooling_cases[mask], 0].clone() \
                                                       + self.cooling[cooling_cases[mask], 1].clone() \
                                                       / self.cooling_division_factor

        # Final computation of the result
        output = base + self.heating - self.cooling
        #print(base, self.heating, self.cooling)

        # Trick needed since some sequences are padded
        output[torch.where(x[:,-1,0] < 1e-6)[0],:] = 0.

        # Recall the current base temperature and the power predictions
        self.last_base[:, 0] = base[:, 0]
        self.last_base[:, 1] = output[:, 1]

        # Return all the network's outputs
        return output, (base_h, base_c)


class SimpleLSTMBase(nn.Module):
    """
    Class similar to `DiscountedModulesBase`, but with a ResNet type of architecture, i.e. the NNs
    actually predict differences in temperature, which is added to the input temperature before
    the model returns it.
    Predictions for the power consumption don't change.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            temperature_column: int,
            power_column: int,
            base_division_factor: list
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Indices to take in the LSTM
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            temperature_column:         Column to predict
            power_column:               Column of the power
            base_division_factor:       Basic division factor
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.base_division_factor = torch.FloatTensor(base_division_factor)

        self.last_base = None

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        # We now need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict diffenrences that can be negative
        f = nn.Tanh()

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                for i in range(0, len(sizes) - 1)])

        # Create the LSTMs
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [len(self.temperature_column)]
            self.base_output_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False, mpc_mode:bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   If it should be used for MPC

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c) = states

        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            self.last_base = torch.zeros(x.shape[0], len(self.temperature_column)).to(self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Again, use the last base predictions as input when not warm starting
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_base#.unsqueeze(1)

        # Input embedding step when needed
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)

            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]

        ## BASE MODULE
        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column].squeeze()
        else:
            # Store the outputs
            base = lstm_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1, self.temperature_column].squeeze()

        # Final computation of the result
        output = torch.zeros(x.shape[0], 2, len(self.temperature_column)).to(self.device)
        output[:, 0, :] = base
        output[:, 1, :] = x[:, -1, [self.power_column]].squeeze()

        # Trick needed since some sequences are padded
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :, :] = 0.

        # Recall the current base temperature and the power predictions
        self.last_base = output[:, 0, :]

        # Return all the network's outputs
        return output, (base_h, base_c)


class PhysicsInspiredModuleBase(nn.Module):
    """
    Class similar to `DiscountedModulesBase`, but with a ResNet type of architecture, i.e. the NNs
    actually predict differences in temperature, which is added to the input temperature before
    the model returns it.
    Predictions for the power consumption don't change.
    """

    def __init__(
        self,
        device,
        base_indices: list,
        effect_indices: list,
        learn_discount_factors: bool,
        learn_initial_hidden_states: bool,
        feed_input_through_nn: bool,
        input_nn_hidden_sizes: list,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        layer_norm: bool,
        feed_output_through_nn: bool,
        output_nn_hidden_sizes: list,
        case_column: int,
        valve_column: int,
        temperature_column: int,
        out_column: int,
        neigh_column: int,
        power_column: int,
        zero_power: float,
        base_division_factor: list,
        normalization_variables: dict,
        parameter_scalings: dict,
        use_energy_prediction: bool
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            effect_indices:             Input features used to predict the heating and cooling components
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
            use_energy_prediction:      Flag to set to `True` if want to use the energy prediction from the last step
                                          as input for the current prediction (`False` for physics-based models)
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.neigh_column = neigh_column
        self.zero_power = zero_power
        self.use_energy_prediction = use_energy_prediction
        self.base_division_factor = torch.FloatTensor(base_division_factor)

        self.last_base = None
        self.heating = None
        self.cooling = None

        self.room_min = 17.658691
        self.room_diff = 13.993608
        self.out_min = -9.936853
        self.out_diff = 46.631382
        self.neigh_min = 14.910292
        self.neigh_diff = 18.205642

        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]
        self.neigh_min = normalization_variables['Neigh'][0]
        self.neigh_diff = normalization_variables['Neigh'][1]

        self.a_scaling = 40
        self.b_scaling = 8000
        self.c_scaling = 8000
        self.d_scaling = 40

        self.a_scaling = parameter_scalings['a']
        self.b_scaling = parameter_scalings['b']
        self.c_scaling = parameter_scalings['c']
        self.d_scaling = parameter_scalings['d']

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        # We now need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict diffenrences that can be negative
        f = nn.Tanh()

        if self.learn_discount_factors:

            self.a = nn.Linear(1, 1, bias=False)
            self.b = nn.Linear(1, 1, bias=False)
            self.c = nn.Linear(1, 1, bias=False)
            self.d = nn.Linear(1, 1, bias=False)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                for i in range(0, len(sizes) - 1)])
            if self.use_energy_prediction:
                sizes = [len(self.effect_indices)] + self.input_nn_hidden_sizes
                self.effect_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                      for i in range(0, len(sizes) - 1)])

        # Create the LSTMs of the 3 modules
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            if self.use_energy_prediction:
                # Heating and cooling modules predict temperature and power
                sizes = [self.input_nn_hidden_sizes[-1]] + self.output_nn_hidden_sizes + [1]
                # Here we now need a non negative activation to ensure the positivity of the heating effect
                # or the negativity of the cooling one
                f = nn.Sigmoid()
                self.heating_output_nn_power = nn.ModuleList(
                    [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                     range(0, len(sizes) - 1)])
                self.cooling_output_nn_power = nn.ModuleList(
                    [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                     range(0, len(sizes) - 1)])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
            if 'a.weight' == name:
                nn.init.constant_(param, 1.)
            if 'b.weight' == name:
                nn.init.constant_(param, 1.)
            if 'c.weight' == name:
                nn.init.constant_(param, 1.)
            if 'd.weight' == name:
                nn.init.constant_(param, 1.)


    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            wamr_start: Whether we are warm starting the model

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c) = states

        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            self.last_base = torch.zeros(x.shape[0], 2).to(self.device)
            self.heating = torch.zeros(x.shape[0], 2).to(self.device)
            self.cooling = torch.zeros(x.shape[0], 2).to(self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Again, use the last base predictions as input when not warm starting
        if not warm_start:
            if self.use_energy_prediction:
                x[:, -1, [self.temperature_column, self.power_column]] = self.last_base
            else:
                x[:, -1, [self.temperature_column]] = self.last_base[:, 0].unsqueeze(1)

        # Define the output of the 3 modules predicting temperature and power
        base = torch.zeros(x.shape[0], 2)
        base[:, 1] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        self.heating[:, 1] = 0.
        self.cooling[:, 1] = 0.

        # Discount previous effects
        if self.learn_discount_factors:
            mask = (x[:, -1, 0] > 0.05) & (x[:, 0, self.case_column] > 0.5)
            #self.heating[mask, 0] = (1. - 1. / self.discount_factor_heating / x[mask, -1, 0]) \
            #                        * self.heating[mask, 0].clone()

            if sum(mask) > 0:
                #self.heating[mask, 0] = self.heating[mask, 0].clone() - 1. / self.b * \
                #                                                     (((x[mask, -1, [self.temperature_column]]
                #                                                        + self.heating[mask, 0].clone() - 0.1) / 0.8
                #                                                     * self.room_diff + self.room_min) - ((x[mask, -1, 0] - 0.1)
                #                                                     / 0.8 * self.out_diff + self.out_min))
                #self.heating[mask, 0] = self.heating[mask, 0].clone() - 1. / self.c * \
                #                                                     (((x[mask, -1, [self.temperature_column]]
                #                                                          + self.heating[mask, 0].clone() - 0.1) / 0.8
                #                                                         * self.room_diff + self.room_min) - ((x[mask, -1, 3] - 0.1)
                #                                                     / 0.8 * self.neigh_diff + self.neigh_min))
                self.heating[mask, 0] = self.heating[mask, 0].clone() - self.b(
                                                                 (((x[mask, -1, [self.temperature_column]]
                                                                    + self.heating[mask, 0].clone() - 0.1) / 0.8
                                                                 * self.room_diff + self.room_min) - ((x[mask, -1, self.out_column] - 0.1)
                                                                 / 0.8 * self.out_diff + self.out_min)).reshape(-1,1)).squeeze() / self.b_scaling

                self.heating[mask, 0] = self.heating[mask, 0].clone() - self.c(
                                                                 (((x[mask, -1, [self.temperature_column]]
                                                                      + self.heating[mask, 0].clone() - 0.1) / 0.8
                                                                     * self.room_diff + self.room_min) - ((x[mask, -1, self.neigh_column] - 0.1)
                                                                                                 / 0.8 * self.neigh_diff + self.neigh_min)).reshape(-1,1)).squeeze() / self.c_scaling

            mask = (x[:, -1, 0] > 0.05) & (x[:, 0, self.case_column] < 0.5)

            if sum(mask) > 0:

                #self.cooling[mask, 0] = self.cooling[mask, 0].clone() - 1. / self.b * \
                #                                                 (((x[mask, -1, 0] - 0.1) / 0.8 * self.out_diff + self.out_min)
                #                                                 - ((x[mask, -1, [self.temperature_column]]
                #                                                    - self.cooling[mask, 0].clone() - 0.1) / 0.8
                #                                                 * self.room_diff + self.room_min))
                #self.cooling[mask, 0] = self.cooling[mask, 0].clone() - 1. / self.c * \
                #                                                 (((x[mask, -1, 3] - 0.1) / 0.8 * self.neigh_diff + self.neigh_min)
                #                                                   - ((x[mask, -1, [self.temperature_column]]
                #                                                        - self.cooling[mask, 0].clone() - 0.1) / 0.8
                #                                                       * self.room_diff + self.room_min))
                self.cooling[mask, 0] = self.cooling[mask, 0].clone() - self.b(
                                                                 (((x[mask, -1, self.out_column] - 0.1) / 0.8 * self.out_diff + self.out_min)
                                                                 - ((x[mask, -1, [self.temperature_column]]
                                                                    - self.cooling[mask, 0].clone() - 0.1) / 0.8
                                                                 * self.room_diff + self.room_min)).reshape(-1,1)).squeeze() / self.b_scaling
                self.cooling[mask, 0] = self.cooling[mask, 0].clone() - self.c(
                                                                 (((x[mask, -1, self.neigh_column] - 0.1) / 0.8 * self.neigh_diff + self.neigh_min)
                                                                   - ((x[mask, -1, [self.temperature_column]]
                                                                        - self.cooling[mask, 0].clone() - 0.1) / 0.8
                                                                       * self.room_diff + self.room_min)).reshape(-1,1)).squeeze() / self.c_scaling

            #self.cooling[mask, 0] = (1. - 1. / self.discount_factor_cooling * x[mask, -1, 0]) \
            #                       * self.cooling[mask, 0].clone()


        # Input embedding step when needed
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            if self.use_energy_prediction:
                effect_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)

            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
                if self.use_energy_prediction:
                    temp = x[:, time_step, self.effect_indices]
                    for layer in self.effect_input_nn:
                        temp = layer(temp)
                    effect_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]
            if self.use_energy_prediction:
                effect_embedding = x[:, :, self.effect_indices]

        ## BASE MODULE
        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]
        else:
            # Store the outputs
            base[:, 0] = lstm_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]
        if len(heating_cases) > 0:
            # Define the inputs for the heating cases
            if self.use_energy_prediction:
                heating_input = effect_embedding[heating_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[heating_cases, :, self.valve_column], axis=1).values > 0.1001

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.use_energy_prediction:
                    if self.feed_output_through_nn:
                        # Power computation
                        temp = heating_input[mask, -1, :]
                        for layer in self.heating_output_nn_power:
                            temp = layer(temp)
                        self.heating[heating_cases[mask], 1] = temp.squeeze()
                    else:
                        self.heating[heating_cases[mask], 1] = heating_input[mask, -1, 1]
                else:
                    self.heating[heating_cases[mask], 1] = x[heating_cases[mask], -1, self.power_column].clone() \
                                                           - torch.from_numpy(self.zero_power).to(self.device)

                # Temperature effect
                self.heating[heating_cases[mask], 0] = self.heating[heating_cases[mask], 0].clone() \
                                                       + self.a(self.heating[heating_cases[mask], 1].clone().
                                                                reshape(-1,1)).squeeze() / self.a_scaling
                                                        # + self.heating[heating_cases[mask], 1].clone() / self.a

        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]
        if len(cooling_cases) > 0:
            # Define the inputs for the heating cases, and tensors that will retain the heating effect,
            # the hidden states and cell states
            if self.use_energy_prediction:
                cooling_input = effect_embedding[cooling_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[cooling_cases, :, self.valve_column], axis=1).values > 0.1001

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.use_energy_prediction:
                    if self.feed_output_through_nn:
                        # Power computation
                        temp = cooling_input[mask, -1, :]
                        for layer in self.cooling_output_nn_power:
                            temp = layer(temp)
                        self.cooling[cooling_cases[mask], 1] = temp.squeeze()
                        raise NotImplementedError('Check that!')
                    else:
                        self.cooling[cooling_cases[mask], 1] = cooling_input[mask, -1, 1]
                else:
                    self.cooling[cooling_cases[mask], 1] = torch.from_numpy(self.zero_power).to(self.device) \
                                                           - x[cooling_cases[mask], -1, self.power_column].clone()

                # Temperature effect
                self.cooling[cooling_cases[mask], 0] = self.cooling[cooling_cases[mask], 0].clone() \
                                                       + self.d(self.cooling[cooling_cases[mask], 1].clone().
                                                                reshape(-1,1)).squeeze() / self.d_scaling
                                                    # + self.cooling[cooling_cases[mask], 1].clone() / self.d

        # Final computation of the result
        output = base + self.heating - self.cooling
        #print(base, self.heating, self.cooling)

        # Trick needed since some sequences are padded
        output[torch.where(x[:,-1,0] < 1e-6)[0],:] = 0.

        # Recall the current base temperature and the power predictions
        self.last_base[:, 0] = base[:, 0]
        self.last_base[:, 1] = output[:, 1]

        # Return all the network's outputs
        return output, (base_h, base_c)


class PhysicsInspiredModuleBaseOLD(nn.Module):
    """
    Class similar to `DiscountedModulesBase`, but with a ResNet type of architecture, i.e. the NNs
    actually predict differences in temperature, which is added to the input temperature before
    the model returns it.
    Predictions for the power consumption don't change.
    """

    def __init__(
        self,
        device,
        base_indices: list,
        effect_indices: list,
        learn_discount_factors: bool,
        learn_initial_hidden_states: bool,
        feed_input_through_nn: bool,
        input_nn_hidden_sizes: list,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        layer_norm: bool,
        feed_output_through_nn: bool,
        output_nn_hidden_sizes: list,
        case_column: int,
        valve_column: int,
        temperature_column: int,
        power_column: int,
        zero_power: float,
        base_division_factor: list,
        cooling_division_factor: list,
        heating_division_factor: list,
        use_energy_prediction: bool
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            effect_indices:             Input features used to predict the heating and cooling components
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            cooling_division_factor:    Factors to scale the heating predictions, similar to `base_division_factor`
            heating_division_factor:    Factors to scale the cooling predictions, similar to `base_division_factor`
            use_energy_prediction:      Flag to set to `True` if want to use the energy prediction from the last step
                                          as input for the current prediction (`False` for physics-based models)
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        #self.base_indices = base_indices
        self.base_indices = base_indices[2:]
        print('Achtung')
        self.solar_indices = [1, 5, 6, 8, 9]
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.cooling_division_factor = torch.FloatTensor(cooling_division_factor)
        self.heating_division_factor = torch.FloatTensor(heating_division_factor)

        self.last_base = None
        self.heating = None
        self.cooling = None

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        # We now need to consider the TanH activation - or any other that allows negative outputs, since
        # we predict diffenrences that can be negative
        f = nn.Tanh()

        if self.learn_discount_factors:
            self.a = nn.Parameter(data=torch.zeros(1) + 70)#250000
            self.c = nn.Parameter(data=torch.zeros(1) + 70)#250000
            self.b = nn.Parameter(data=torch.zeros(1) + 10000)#30
            self.e = nn.Parameter(data=torch.zeros(1) + 35)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_solar_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, int(self.lstm_hidden_size/4)))
            self.initial_solar_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, int(self.lstm_hidden_size/4)))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                for i in range(0, len(sizes) - 1)])

            sizes = [len(self.solar_indices)] + [int(x/4) for x in self.input_nn_hidden_sizes]
            self.solar_input_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f)
                                                  for i in range(0, len(sizes) - 1)])

        # Create the LSTMs of the 3 modules
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        lstm_input_size = int(self.input_nn_hidden_sizes[-1]/4) if self.feed_input_through_nn else len(self.solar_indices)
        self.solar_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=int(self.lstm_hidden_size/4),
                                  num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)
            self.solar_norm = nn.LayerNorm(normalized_shape=int(self.lstm_hidden_size/4))

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            # Heating and cooling modules predict temperature and power
            sizes = [int(self.lstm_hidden_size/4)] + [int(x/4) for x in self.output_nn_hidden_sizes] + [1]
            # Here we now need a non negative activation to ensure the positivity of the heating effect
            # or the negativity of the cooling one
            f = nn.Sigmoid()
            self.solar_output_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            wamr_start: Whether we are warm starting the model

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c, solar_h, solar_c) = states

        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                solar_h = torch.stack([self.initial_solar_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                solar_c = torch.stack([self.initial_solar_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                solar_h = torch.zeros((self.lstm_num_layers, x.shape[0], int(self.lstm_hidden_size/4))).to(self.device)
                solar_c = torch.zeros((self.lstm_num_layers, x.shape[0], int(self.lstm_hidden_size/4))).to(self.device)

            self.last_base = torch.zeros(x.shape[0], 2).to(self.device)
            self.heating = torch.zeros(x.shape[0], 2).to(self.device)
            self.cooling = torch.zeros(x.shape[0], 2).to(self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Again, use the last base predictions as input when not warm starting
        if not warm_start:
            x[:, -1, [self.temperature_column]] = self.last_base[:, 0].unsqueeze(1)

        # Define the output of the 3 modules predicting temperature and power
        base = torch.zeros(x.shape[0], 2)
        base[:, 1] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        self.heating[:, 1] = 0.
        self.cooling[:, 1] = 0.

        # Input embedding step when needed
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            solar_embedding = torch.zeros(x.shape[0], x.shape[1], int(self.input_nn_hidden_sizes[-1]/4)).to(self.device)

            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp

                temp = x[:, time_step, self.solar_indices]
                for layer in self.solar_input_nn:
                    temp = layer(temp)
                solar_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]
            solar_embedding = x[:, :, self.solar_indices]

        ## BASE MODULE
        # LSTM prediction for the base temperature
        base_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            base_output = self.base_norm(base_output)

        # Solar prediction
        solar_output, (solar_h, solar_c) = self.solar_lstm(solar_embedding, (solar_h, solar_c))
        if self.layer_norm:
            solar_output = self.solar_norm(solar_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = base_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, 0] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]

            temp = solar_output[:, -1, :]
            for layer in self.solar_output_nn:
                temp = layer(temp)
            if x.shape[0] > 1:
                solar_input = temp.squeeze()
            else:
                solar_input = temp.reshape(x.shape[0], -1)

        else:
            # Store the outputs
            base[:, 0] = base_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1, self.temperature_column]
            solar_input = solar_output[:, -1, 0]

        # Discount previous effects
        if self.learn_discount_factors:
            mask = (x[:, -1, 0] > 0.05) & (x[:, 0, self.case_column] > 0.5)
            if sum(mask) > 0:

                self.heating[mask, 0] = self.heating[mask, 0] - (1. / self.b
                                                                 * (((x[mask, -1, [self.temperature_column]]
                                                                      + self.heating[mask, 0].clone() - 0.1) / 0.8
                                                                     * 12.964820 + 18.181446) - ((x[mask, -1, 0] - 0.1)
                                                                                                 / 0.8 * 46.631382 - 9.936853)))

                # Multiply by (irr - 0.1) to get zero when there is no sun
                self.heating[mask, 0] = self.heating[mask, 0] + 1. / self.e * (x[mask, -1, 1] - 0.1) * solar_input[mask]

            mask = (x[:, -1, 0] > 0.05) & (x[:, 0, self.case_column] < 0.5)

            if sum(mask) > 0:

                self.cooling[mask, 0] = self.cooling[mask, 0] - (1. / self.b
                                                                 * (((x[
                                                                          mask, -1, 0] - 0.1) / 0.8 * 46.631382 - 9.936853)
                                                                    - ((x[mask, -1, [self.temperature_column]]
                                                                        - self.cooling[mask, 0].clone() - 0.1) / 0.8
                                                                       * 12.964820 + 18.181446)))

                # Less cooling effect
                # Multiply by (irr - 0.1) to get zero when there is no sun
                self.cooling[mask, 0] = self.cooling[mask, 0] - 1. / self.e * (x[mask, -1, 1]-0.1) * solar_input[mask]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]
        if len(heating_cases) > 0:
            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[heating_cases, :, self.valve_column], axis=1).values > 0.1001

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted
                self.heating[heating_cases[mask], 1] = x[heating_cases[mask], -1, self.power_column].clone() \
                                                           - torch.from_numpy(self.zero_power).to(self.device)

                # Temperature effect
                self.heating[heating_cases[mask], 0] = self.heating[heating_cases[mask], 0].clone() \
                                                       + self.heating[heating_cases[mask], 1].clone() \
                                                       / self.a
        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]
        if len(cooling_cases) > 0:
            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.max(x[cooling_cases, :, self.valve_column], axis=1).values > 0.1001

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if sum(mask) > 0:
                # Some manipulations are needed to feed the output through the neural network if wanted

                self.cooling[cooling_cases[mask], 1] = torch.from_numpy(self.zero_power).to(self.device) \
                                                           - x[cooling_cases[mask], -1, self.power_column].clone()

                # Temperature effect
                self.cooling[cooling_cases[mask], 0] = self.cooling[cooling_cases[mask], 0].clone() \
                                                       + self.cooling[cooling_cases[mask], 1].clone() \
                                                       / self.c

        # Final computation of the result
        output = base + self.heating - self.cooling
        #print(base, self.heating, self.cooling)

        # Trick needed since some sequences are padded
        output[torch.where(x[:,-1,0] < 1e-6)[0],:] = 0.

        # Recall the current base temperature and the power predictions
        self.last_base[:, 0] = base[:, 0]
        self.last_base[:, 1] = output[:, 1]

        # Return all the network's outputs
        return output, (base_h, base_c, solar_h, solar_c)


class ResDiscountedModulesBaseMany(nn.Module):
    """
    Class similar to `ResDiscountedModulesBase`, but now capable to predict the temperature evolution
    and power consumption of several rooms at the same time.
    The base module predicts the base temperature of all the rooms together (multiple outputs) and
    then the heating and cooling modules are branched for each room, and each branch is further
    subdivided in two to predict the temperature effect and the power consumption of the room.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            effect_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            cooling_division_factor: list,
            heating_division_factor: list

    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            effect_indices:             Input features used to predict the heating and cooling components
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            cooling_division_factor:    Factors to scale the heating predictions, similar to `base_division_factor`
            heating_division_factor:    Factors to scale the cooling predictions, similar to `base_division_factor`
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.cooling_division_factor = torch.FloatTensor(cooling_division_factor)
        self.heating_division_factor = torch.FloatTensor(heating_division_factor)

        self.last_base = None
        self.heating = None
        self.cooling = None

        # Build the models
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        f = nn.Tanh()  # nn.LeakyReLU(negative_slope=1e-6)

        if self.learn_discount_factors:
            self.discount_factor_heating = nn.Parameter(data=torch.zeros(1) + 0.95)
            self.discount_factor_cooling = nn.Parameter(data=torch.zeros(1) + 0.95)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
            self.initial_heating_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_heating_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))

        # Process the input if wanted
        if self.feed_input_through_nn:
            sizes = [len(self.base_indices)] + self.input_nn_hidden_sizes
            self.base_input_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            sizes = [len(self.effect_indices)] + self.input_nn_hidden_sizes
            self.effect_input_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)])

        # Create the LSTMs of the 3 modules
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)

        # Make sure the output is of the right size if no NN is required at the output
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.effect_indices)
        lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
        self.heating_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                                    num_layers=self.lstm_num_layers, batch_first=True)
        self.cooling_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                                    num_layers=self.lstm_num_layers, batch_first=True)

        # Create the normalization layers if wanted
        if self.layer_norm:
            self.base_norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)
            self.heating_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
            self.cooling_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [len(self.temperature_column)]
            self.base_output_nn = nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])

            # Heating and cooling modules predict temperature and power
            sizes = [lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            f = nn.Sigmoid()

            # Construct one temperature effect for each room
            self.heating_output_nn_temperature = nn.ModuleList(
                [nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
                 for _ in self.temperature_column])
            self.cooling_output_nn_temperature = nn.ModuleList(
                [nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
                 for _ in self.temperature_column])

            # Construct one power effect for each room
            self.heating_output_nn_power = nn.ModuleList([nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)]) for _ in self.temperature_column])
            self.cooling_output_nn_power = nn.ModuleList([nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)]) for _ in self.temperature_column])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            wamr_start: Whether we are warm starting the model

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c) = states

        else:
            if self.learn_initial_hidden_states:
                base_h = torch.stack([self.initial_base_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                base_c = torch.stack([self.initial_base_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                heating_h = torch.stack([self.initial_heating_h.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                heating_c = torch.stack([self.initial_heating_c.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                cooling_h = torch.stack([self.initial_cooling_h.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
                cooling_c = torch.stack([self.initial_cooling_c.clone() for _ in range(x.shape[0])], dim=1).to(
                    self.device)
            else:
                base_h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                base_c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
                heating_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                heating_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_h = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)
                cooling_c = torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device)

            # Initialize tensors to recall base predictions and the heating and cooling effect
            # This now has a size larger than 2, as it needs to retain the information for all the rooms
            self.last_base = torch.zeros(x.shape[0], len(self.temperature_column) + len(self.power_column)).to(
                self.device)
            self.heating = torch.zeros(x.shape[0], len(self.temperature_column) + len(self.power_column)).to(
                self.device)
            self.cooling = torch.zeros(x.shape[0], len(self.temperature_column) + len(self.power_column)).to(
                self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Replace the temperatures by the predicted base temperatures, and the powers by the last predictions
        if not warm_start:
            x[:, -1, [self.temperature_column]] = self.last_base[:, :len(self.temperature_column)].unsqueeze(1)
            x[:, -1, [self.power_column]] = self.last_base[:, len(self.temperature_column):].unsqueeze(1)

        # Define the output of the 3 modules predicting temperature and power
        base = torch.zeros(x.shape[0], len(self.temperature_column) + len(self.power_column))
        base[:, len(self.temperature_column):] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)

        # Put the powers to zero
        self.heating[:, len(self.temperature_column):] = 0.
        self.cooling[:, len(self.temperature_column):] = 0.

        # Discount the effect of heating/cooling from previous time steps
        if self.learn_discount_factors:
            self.heating[:, :len(self.temperature_column)] = self.discount_factor_heating * self.heating[:, :len(
                self.temperature_column)].clone()
            self.cooling[:, :len(self.temperature_column)] = self.discount_factor_cooling * self.cooling[:, :len(
                self.temperature_column)].clone()

        # Input embedding step when needed
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            effect_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)

            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.base_indices]
                for layer in self.base_input_nn:
                    temp = layer(temp)
                base_embedding[:, time_step, :] = temp
                temp = x[:, time_step, self.effect_indices]
                for layer in self.effect_input_nn:
                    temp = layer(temp)
                effect_embedding[:, time_step, :] = temp
        else:
            base_embedding = x[:, :, self.base_indices]
            effect_embedding = x[:, :, self.effect_indices]

        ## BASE MODULE
        # LSTM prediction for the base temperature
        lstm_output, (base_h, base_c) = self.base_lstm(base_embedding, (base_h, base_c))
        if self.layer_norm:
            lstm_output = self.base_norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.feed_output_through_nn:
            # Put the data is the form needed for the neural net
            temp = lstm_output[:, -1, :]
            # Go through the input layer of the NN
            for layer in self.base_output_nn:
                temp = layer(temp)
            base[:, :len(self.temperature_column)] = temp.squeeze() / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        else:
            # Store the output
            raise NotImplementedError("Check")
            base[:, :len(self.temperature_column)] = lstm_output[:, -1, 0] / self.base_division_factor[0] + x[:, -1, self.temperature_column]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]

        if len(heating_cases) > 0:
            # Define the inputs for the heating cases
            heating_input = effect_embedding[heating_cases, :, :]
            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            valves = x[:, :, self.valve_column]
            mask = torch.max(valves[heating_cases, :, :], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :]) = self.heating_lstm(
                heating_input, (heating_h[:, heating_cases, :], heating_c[:, heating_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.heating_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before, looping through each room
            if self.feed_output_through_nn:
                for i, module in enumerate(self.heating_output_nn_temperature):
                    if sum(mask[:, i]) > 0:
                        temp = lstm_output[mask[:, i], -1, :]
                        for layer in module:
                            temp = layer(temp)
                        self.heating[heating_cases[mask[:, i]], i] += temp.squeeze() / self.heating_division_factor[i]

                for i, module in enumerate(self.heating_output_nn_power):
                    if sum(mask[:, i]) > 0:
                        temp = lstm_output[mask[:, i], -1, :]
                        for layer in module:
                            temp = layer(temp)
                        self.heating[heating_cases[mask[:, i]], i + len(self.temperature_column)] = temp.squeeze()

            else:
                for i in range(len(self.temperature_column)):
                    if sum(mask[:, i]) > 0:
                        self.heating[heating_cases[mask[:, i]], i] += lstm_output[mask[:, i], -1, i] / self.heating_division_factor[i]
                        self.heating[heating_cases[mask[:, i]], i + len(self.temperature_column)] = lstm_output[
                            mask[:, i], -1, i + len(self.temperature_column)]

        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]

        if len(cooling_cases) > 0:
            # Define the inputs for the heating cases, and tensors that will retain the heating effect,
            # the hidden states and cell states
            cooling_input = effect_embedding[cooling_cases, :, :]

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            valves = x[:, :, self.valve_column]
            mask = torch.max(valves[cooling_cases, :, :], axis=1).values > 0.1001

            # LSTMs prediction
            lstm_output, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :]) = self.cooling_lstm(
                cooling_input, (cooling_h[:, cooling_cases, :], cooling_c[:, cooling_cases, :])
            )

            if self.layer_norm:
                lstm_output = self.cooling_norm(lstm_output)

            # Compute the actual effects only if the valves were open (through the 'mask' defined before)
            # Note that the hidden and cell states still need to be updated, and were thus taken care of
            # before
            if self.feed_output_through_nn:

                for i, module in enumerate(self.cooling_output_nn_temperature):
                    if sum(mask[:, i]) > 0:
                        temp = lstm_output[mask[:, i], -1, :]
                        for layer in module:
                            temp = layer(temp)
                        self.cooling[cooling_cases[mask[:, i]], i] += temp.squeeze() / self.cooling_division_factor[i]

                for i, module in enumerate(self.cooling_output_nn_power):
                    if sum(mask[:, i]) > 0:
                        temp = lstm_output[mask[:, i], -1, :]
                        for layer in module:
                            temp = layer(temp)
                        self.cooling[cooling_cases[mask[:, i]], i + len(self.temperature_column)] = temp.squeeze()

            else:
                for i in range(len(self.temperature_column)):
                    if sum(mask[:, i]) > 0:
                        self.cooling[cooling_cases[mask[:, i]], i] += lstm_output[mask[:, i], -1, i] / self.cooling_division_factor[i]
                        self.cooling[cooling_cases[mask[:, i]], i + len(self.temperature_column)] = lstm_output[
                            mask[:, i], -1, i + len(self.temperature_column)]

        # Final computation of the result
        output = base + self.heating - self.cooling

        # Trick needed since some sequences are padded
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.

        self.last_base[:, :len(self.temperature_column)] = base[:, :len(self.temperature_column)]
        self.last_base[:, len(self.temperature_column):] = output[:, len(self.temperature_column):]

        # Return all the network's outputs
        return output, (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c)


class ResDiscountedModulesBaseManySep(nn.Module):
    """
    Class similar to `ResDiscountedModulesBaseMany`, but now the rooms are treated separately, with each
    room having its own input embeddings, base module, and heating/cooling modules.
    """

    def __init__(
            self,
            device,
            base_indices: list,
            effect_indices: list,
            learn_discount_factors: bool,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            feed_output_through_nn: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            valve_column: int,
            temperature_column: int,
            power_column: int,
            zero_power: float,
            base_division_factor: list,
            cooling_division_factor: list,
            heating_division_factor: list
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            base_indices:               Input features used to predict the base component
            effect_indices:             Input features used to predict the heating and cooling components
            learn_discount_factors:     Whether to add a discount factor to the added effects on the room temperature
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            feed_output_through_nn:     Flag whether or not to feed the LSTM output through a NNs
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            valve_column:               Index of the column corresponding to the valve opening
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            base_division_factor:       Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            cooling_division_factor:    Factors to scale the heating predictions, similar to `base_division_factor`
            heating_division_factor:    Factors to scale the cooling predictions, similar to `base_division_factor`
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.base_indices = base_indices
        self.effect_indices = effect_indices
        self.learn_discount_factors = learn_discount_factors
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.feed_output_through_nn = feed_output_through_nn
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.valve_column = valve_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.zero_power = zero_power
        self.base_division_factor = torch.FloatTensor(base_division_factor)
        self.cooling_division_factor = torch.FloatTensor(cooling_division_factor)
        self.heating_division_factor = torch.FloatTensor(heating_division_factor)

        self.last_base = None
        self.heating = None
        self.cooling = None

        # Build the LSTMs, one for each component
        self._build_models()

    def _build_models(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and power prediction.
        Each is composed of a LSTM possibly followed by a NN and preceeded by another one.
        """

        f = nn.Tanh()  # nn.LeakyReLU(negative_slope=1e-6)

        if self.learn_discount_factors:
            self.discount_factor_heating = nn.Parameter(data=torch.zeros(3) + 0.95)
            self.discount_factor_cooling = nn.Parameter(data=torch.zeros(3) + 0.95)

        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_base_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
            self.initial_heating_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_heating_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))
            self.initial_cooling_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, lstm_hidden_size))

        # Process the input if wanted, in a list because we want it for each room
        if self.feed_input_through_nn:
            sizes = [[len(base)] + self.input_nn_hidden_sizes for base in self.base_indices]
            self.base_input_nn = nn.ModuleList([nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[j][i], sizes[j][i + 1]), f) for i in range(0, len(sizes[j]) - 1)]) for
                                                j, room in enumerate(self.temperature_column)])

            sizes = [[len(effect)] + self.input_nn_hidden_sizes for effect in self.effect_indices]
            self.effect_input_nn = nn.ModuleList([nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[j][i], sizes[j][i + 1]), f) for i in range(0, len(sizes[j]) - 1)]) for
                                                  j, room in enumerate(self.temperature_column)])

        # Create the LSTMs of the 3 modules, in a list because we have them for each room
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.base_indices)
        self.base_lstm = nn.ModuleList([nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                                num_layers=self.lstm_num_layers, batch_first=True) for room in
                                        self.temperature_column])

        # Make sure the output is of the right size if no NN is required at the output
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.effect_indices)
        lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
        self.heating_lstm = nn.ModuleList([nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                                                   num_layers=self.lstm_num_layers, batch_first=True) for room in
                                           self.temperature_column])
        self.cooling_lstm = nn.ModuleList([nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                                                   num_layers=self.lstm_num_layers, batch_first=True) for room in
                                           self.temperature_column])

        # Create the normalization layers if wwanted
        self.base_norm = nn.ModuleList(
            [nn.LayerNorm(normalized_shape=self.lstm_hidden_size) for _ in self.temperature_column])
        self.heating_norm = nn.ModuleList(
            [nn.LayerNorm(normalized_shape=lstm_hidden_size) for _ in self.temperature_column])
        self.cooling_norm = nn.ModuleList(
            [nn.LayerNorm(normalized_shape=lstm_hidden_size) for _ in self.temperature_column])

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        if self.feed_output_through_nn:
            # The base prediction only predicts the temperature
            sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            self.base_output_nn = nn.ModuleList(
                [nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
                 for _ in self.temperature_column])

            # Heating and cooling modules predict temperature and power, as list of lists (for each
            # room a list of layers)
            sizes = [lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
            f = nn.Sigmoid()
            self.heating_output_nn_temperature = nn.ModuleList(
                [nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
                 for _ in self.temperature_column])
            self.cooling_output_nn_temperature = nn.ModuleList(
                [nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in range(0, len(sizes) - 1)])
                 for _ in self.temperature_column])

            self.heating_output_nn_power = nn.ModuleList([nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)]) for temp in self.temperature_column])
            self.cooling_output_nn_power = nn.ModuleList([nn.ModuleList(
                [nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), f) for i in
                 range(0, len(sizes) - 1)]) for temp in self.temperature_column])

        # Xavier initialization
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, x_: torch.FloatTensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            wamr_start: Whether we are warm starting the model

        Returns:
            A tuple of tuples, containing the base temperature output, the heating and the cooling effects,
             as well as the current hidden and cell states
        """

        x = x_.clone()

        # Define hidden and cell states of the 3 modules
        if states is not None:
            (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c) = states

        else:
            # Now slightly more complex, need to initialize everything for each room
            if self.learn_initial_hidden_states:
                base_h = [torch.stack([self.initial_base_h.clone() for i in range(x.shape[0])], dim=1).to(self.device)
                          for _ in self.temperature_column]
                base_c = [torch.stack([self.initial_base_c.clone() for i in range(x.shape[0])], dim=1).to(self.device)
                          for _ in self.temperature_column]
                heating_h = [
                    torch.stack([self.initial_heating_h.clone() for i in range(x.shape[0])], dim=1).to(self.device) for
                    _ in self.temperature_column]
                heating_c = [
                    torch.stack([self.initial_heating_c.clone() for i in range(x.shape[0])], dim=1).to(self.device) for
                    _ in self.temperature_column]
                cooling_h = [
                    torch.stack([self.initial_cooling_h.clone() for i in range(x.shape[0])], dim=1).to(self.device) for
                    _ in self.temperature_column]
                cooling_c = [
                    torch.stack([self.initial_cooling_c.clone() for i in range(x.shape[0])], dim=1).to(self.device) for
                    _ in self.temperature_column]
            else:
                base_h = [torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device) for _
                          in self.temperature_column]
                base_c = [torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device) for _
                          in self.temperature_column]
                lstm_hidden_size = 2 if self.lstm_hidden_size == 1 else self.lstm_hidden_size
                heating_h = [torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device) for _ in
                             self.temperature_column]
                heating_c = [torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device) for _ in
                             self.temperature_column]
                cooling_h = [torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device) for _ in
                             self.temperature_column]
                cooling_c = [torch.zeros((self.lstm_num_layers, x.shape[0], lstm_hidden_size)).to(self.device) for _ in
                             self.temperature_column]

            # Initialize tensors to keep everything in momory
            self.last_base = torch.zeros(x.shape[0], len(self.temperature_column) + len(self.power_column)).to(
                self.device)
            self.heating = torch.zeros(x.shape[0], len(self.temperature_column) + len(self.power_column)).to(
                self.device)
            self.cooling = torch.zeros(x.shape[0], len(self.temperature_column) + len(self.power_column)).to(
                self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # if not warm starting, use the last predictions as inputs (the base for the temperatures)
        if not warm_start:
            x[:, -1, [self.temperature_column]] = self.last_base[:, :len(self.temperature_column)].unsqueeze(1)
            x[:, -1, [self.power_column]] = self.last_base[:, len(self.temperature_column):].unsqueeze(1)

        # Define the output of the 3 modules predicting temperature and power
        base = torch.zeros(x.shape[0], len(self.temperature_column) + len(self.power_column))
        base[:, len(self.temperature_column):] = torch.from_numpy(self.zero_power).to(self.device)
        base = base.to(self.device)
        self.heating[:, len(self.temperature_column):] = 0.
        self.cooling[:, len(self.temperature_column):] = 0.

        # Discount the effect from previous time steps
        if self.learn_discount_factors:
            self.heating[:, :len(self.temperature_column)] = self.discount_factor_heating * self.heating[:, :len(
                self.temperature_column)].clone()
            self.cooling[:, :len(self.temperature_column)] = self.discount_factor_cooling * self.cooling[:, :len(
                self.temperature_column)].clone()

        # Input embedding step when needed
        if self.feed_input_through_nn:
            base_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1],
                                         len(self.temperature_column)).to(self.device)
            effect_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1],
                                           len(self.temperature_column)).to(self.device)

            # Always loop over the rooms to create the wanted embeddings
            for time_step in range(x.shape[1]):
                for i, module in enumerate(self.base_input_nn):
                    temp = x[:, time_step, self.base_indices[i]]
                    for layer in module:
                        temp = layer(temp)
                    base_embedding[:, time_step, :, i] = temp

                for i, module in enumerate(self.effect_input_nn):
                    temp = x[:, time_step, self.effect_indices[i]]
                    for layer in module:
                        temp = layer(temp)
                    effect_embedding[:, time_step, :, i] = temp
        else:
            for i in range(len(self.temperature_column)):
                base_embedding = x[:, :, self.base_indices[i], i]
                effect_embedding = x[:, :, self.effect_indices[i], i]

        ## BASE MODULE
        # LSTM prediction for the base temperature
        # Loop over the room, take the right embedding as inputs
        for i, lstm, norm, module in zip(np.arange(len(self.temperature_column)), self.base_lstm, self.base_norm,
                                         self.base_output_nn):
            lstm_output, (base_h[i], base_c[i]) = lstm(base_embedding[:, :, :, i], (base_h[i], base_c[i]))
            if self.layer_norm:
                lstm_output = norm(lstm_output)

            # Some manipulations are needed to feed the output through the neural network if wanted
            if self.feed_output_through_nn:
                # Put the data is the form needed for the neural net
                temp = lstm_output[:, -1, :]
                # Go through the input layer of the NN
                for layer in module:
                    temp = layer(temp)
                base[:, i] = temp.squeeze() / self.base_division_factor[i] + x[:, -1, self.temperature_column[i]]
            else:
                # Store the outputs
                base[:, i] = lstm_output[:, -1, 0, i] / self.base_division_factor[i] + x[:, -1,
                                                                                       self.temperature_column[i]]

        ## HEATING MODULE
        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, self.case_column] > 0.5)[0]

        if len(heating_cases) > 0:
            # Again, loop over the rooms to compute the heating effects
            for i, lstm, norm, module, power in zip(np.arange(len(self.temperature_column)), self.heating_lstm,
                                                     self.heating_norm, self.heating_output_nn_temperature,
                                                     self.heating_output_nn_power):
                # Define the inputs for the heating cases
                heating_input = effect_embedding[heating_cases, :, :, i]
                # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
                # the heating effect has to be zero
                mask = torch.max(x[heating_cases, :, self.valve_column[i]], axis=1).values > 0.1001

                # LSTMs prediction
                lstm_output, (heating_h[i][:, heating_cases, :], heating_c[i][:, heating_cases, :]) = lstm(
                    heating_input, (heating_h[i][:, heating_cases, :], heating_c[i][:, heating_cases, :])
                )

                if self.layer_norm:
                    lstm_output = norm(lstm_output)

                # Compute the actual effects only if the valves were open (through the 'mask' defined before)
                # Note that the hidden and cell states still need to be updated, and were thus taken care of
                # before

                if self.feed_output_through_nn:
                    if sum(mask) > 0:
                        temp = lstm_output[mask, -1, :]
                        for layer in module:
                            temp = layer(temp)
                        self.heating[heating_cases[mask], i] += temp.squeeze() / self.heating_division_factor[i]
                        temp = lstm_output[mask, -1, :]
                        for layer in power:
                            temp = layer(temp)
                        self.heating[heating_cases[mask], i + len(self.temperature_column)] = temp.squeeze()

                else:
                    if sum(mask) > 0:
                        self.heating[heating_cases[mask], i] += lstm_output[mask, -1, i] / self.heating_division_factor[
                            i]
                        self.heating[heating_cases[mask], i + len(self.temperature_column)] = lstm_output[
                            mask, -1, i + len(self.temperature_column)]

        # COOLING MODULE
        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, self.case_column] < 0.5)[0]

        if len(cooling_cases) > 0:
            # Again, loop over the rooms to compute the cooling effects
            for i, lstm, norm, module, power in zip(np.arange(len(self.temperature_column)), self.cooling_lstm,
                                                     self.cooling_norm, self.cooling_output_nn_temperature,
                                                     self.cooling_output_nn_power):
                # Define the inputs for the heating cases, and tensors that will retain the heating effect,
                # the hidden states and cell states
                cooling_input = effect_embedding[cooling_cases, :, :, i]

                # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
                # the heating effect has to be zero
                mask = torch.max(x[cooling_cases, :, self.valve_column[i]], axis=1).values > 0.1001

                # LSTMs prediction
                lstm_output, (cooling_h[i][:, cooling_cases, :], cooling_c[i][:, cooling_cases, :]) = lstm(
                    cooling_input, (cooling_h[i][:, cooling_cases, :], cooling_c[i][:, cooling_cases, :])
                )

                if self.layer_norm:
                    lstm_output = norm(lstm_output)

                # Compute the actual effects only if the valves were open (through the 'mask' defined before)
                # Note that the hidden and cell states still need to be updated, and were thus taken care of
                # before
                if self.feed_output_through_nn:
                    if sum(mask) > 0:
                        temp = lstm_output[mask, -1, :]
                        for layer in module:
                            temp = layer(temp)
                        self.cooling[cooling_cases[mask], i] += temp.squeeze() / self.cooling_division_factor[i]
                        temp = lstm_output[mask, -1, :]
                        for layer in power:
                            temp = layer(temp)
                        self.cooling[cooling_cases[mask], i + len(self.temperature_column)] = temp.squeeze()

                else:
                    if sum(mask) > 0:
                        self.cooling[cooling_cases[mask], i] += lstm_output[mask, -1, i] / self.cooling_division_factor[
                            i]
                        self.cooling[cooling_cases[mask], i + len(self.temperature_column)] = lstm_output[
                            mask, -1, i + len(self.temperature_column)]

        # Final computation of the result
        output = base + self.heating - self.cooling

        # Trick needed since some sequences are padded
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.

        self.last_base[:, :len(self.temperature_column)] = base[:, :len(self.temperature_column)]
        self.last_base[:, len(self.temperature_column):] = output[:, len(self.temperature_column):]

        # Return all the network's outputs
        return output, (base_h, base_c, heating_h, heating_c, cooling_h, cooling_c)