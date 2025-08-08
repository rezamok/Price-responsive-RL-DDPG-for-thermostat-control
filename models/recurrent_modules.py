"""
File containing custom PyTorch modules to build recurrent models
"""

import torch
from torch import nn


class RoomLSTM(nn.Module):
    """
    Class representing the behavior of one room. It builds LSTMs, the output of which are fed in NNs to represent
    the 2 parts of interest about the rooms: its temperature and energy consumption.

    The temperature is computed in to steps: first a base temperature prediction (without heating/cooling) and
    then the added effect of heating/cooling is added/substracted from this base value. This is to ensure that
    the predictions of our model are physically consistent, i.e. it predicts higher temperatures when the
    heating is on that when nothing happens or when cooling is on.
    The main point here is that both the heating and cooling effects pass through a Sigmoid at the output,
    which forces them to be positive. So adding the heating and substracting the cooling from the base
    is bound to yield physically interesting results.

    For the energy computation, there is no base computation, the energy needed to heat or cool is computed
    directly from the added effect.
    """

    def __init__(self, device, component_inputs_indices: dict, effects_inputs_indices: dict, factor: float,
                 hidden_size: int = 1, num_layers: int = 1, NN: bool = True, hidden_sizes: list = [8, 4],
                 output_size: int = 1, interval: int = 15):
        """
        Function to build the LSTM and the consecutive NN.

        The 'components_inputs_indices' dictionary serves to describe which input indices are used
        for each part. Typically, for a room, we want to take weather information, electricity
        consumption, previous state of the room and actuators into account, among others.

        The 'effects_inputs_indices' on the other hand describes which inputs are used to compute the added
        effect of heating/cooling.

        Args:
            device:                    Device on which to build the model (i.e. if uses GPU or not)
            component_inputs_indices:  Dictionary linking the input to the correct parts in the LSTM
            effects_inputs_indices:    Dictionary containing indices needed to compute the added effect of heating
                                        or cooling
            factor:                    Factor used to scale the adding effects to meaningful values
            hidden_size:               Size of the hidden state of the LSTM cells
            num_layers:                Number of layers ot the LSTM
            NN:                        Flag to set true if the output of the LSTM is to be fed in a feedforward
                                        neural network
            hidden_sizes:              Hidden sizes of the NNs
            output_size:               Output size of the NNs (and thus global output size)
            interval:                  Interval between 2 predictions
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.component_inputs_indices= component_inputs_indices
        self.effects_inputs_indices = effects_inputs_indices
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.NN = NN
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = device
        self.factor = factor
        self.interval = interval

        # Define the 2 parts used
        self.parts = ["Temperature", "Energy"]

        # Build the LSTMs, one for each component
        self._build_lstms()

    def _build_lstms(self) -> None:
        """
        Function to build the network, i.e. a base temperature prediction and then the added effect of
        heating/cooling, both for the temperature and energy prediction.
        Each is composed of a LSTM followed by a NN.
        """

        # Use the dictionary to know the input size of each component to build the base LSTM
        self.base_lstm = nn.LSTM(input_size=len(self.component_inputs_indices),
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 batch_first=True)

        # Build the feedforward networks of the base if needed
        if self.NN:
            # Build the input layers of the NNS
            self.base_nn_input_layer = nn.Sequential(nn.Linear(self.hidden_size,
                                                               self.hidden_sizes[0]),
                                                     nn.LeakyReLU())

            # Build the hidden layers
            self.base_nn_hidden_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                self.hidden_sizes[i + 1]),
                                                                      nn.LeakyReLU())
                                                        for i in range(0, len(self.hidden_sizes) - 1)])

            # Build the output layers for each branch
            self.base_nn_output_layer = nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                self.output_size),
                                                      nn.LeakyReLU())

        # Use the dictionary to know the input size of the heating effect on both temperature and energy
        self.heating_lstms = nn.ModuleDict({part: nn.LSTM(input_size=len(self.effects_inputs_indices),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for part in self.parts})

        # Build the feedforward networks if needed
        if self.NN:
            # Build the input layers of the NNS
            self.heating_nn_input_layers = nn.ModuleDict({part: nn.Sequential(nn.Linear(self.hidden_size,
                                                                                     self.hidden_sizes[0]),
                                                                           nn.LeakyReLU())
                                                  for part in self.parts})

            # Build the hidden layers
            self.heating_nn_hidden_layers = nn.ModuleDict({part:
                                                       nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                              self.hidden_sizes[i + 1]),
                                                                                    nn.LeakyReLU())
                                                                      for i in range(0, len(self.hidden_sizes) - 1)])
                                                   for part in self.parts})

            # Build the output layers for each branch
            self.heating_nn_output_layers = nn.ModuleDict({part: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Sigmoid())
                                                   for part in self.parts})

        # Use the dictionary to know the input size of the cooling effect on both temperature and energy
        self.cooling_lstms = nn.ModuleDict({part: nn.LSTM(input_size=len(self.effects_inputs_indices),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for part in self.parts})

        # Build the feedforward networks if needed
        if self.NN:
            # Build the input layers of the NNS
            self.cooling_nn_input_layers = nn.ModuleDict({part: nn.Sequential(nn.Linear(self.hidden_size,
                                                                                     self.hidden_sizes[0]),
                                                                           nn.LeakyReLU())
                                                  for part in self.parts})

            # Build the hidden layers
            self.cooling_nn_hidden_layers = nn.ModuleDict({part:
                                                       nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                              self.hidden_sizes[i + 1]),
                                                                                    nn.LeakyReLU())
                                                                      for i in range(0, len(self.hidden_sizes) - 1)])
                                                   for part in self.parts})

            # Build the output layers for each branch
            self.cooling_nn_output_layers = nn.ModuleDict({part: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Sigmoid())
                                                   for part in self.parts})

    def forward(self, x: torch.FloatTensor, h_0: dict = None, c_0: dict = None) -> dict:
        """
        Custom redefinition of the forward path.

        Args:
            x:      Input
            h_0:    Original hidden state if known (for all LSTMs, i.e. the base, the heating and the cooling)
            c_0:    Original cell state if known (for all LSTMs, i.e. the base, the heating and the cooling)

        Returns:
            A tuple of tuples, containing the base temperature output, the heating and the cooling effects,
             as well as the current hidden and cell states
        """

        # Define the hidden state and cell state as zeros per default, else take them from the input
        if h_0 is None:
            assert c_0 is None, "What??"
            h = torch.zeros((self.num_layers, x.shape[0], self.hidden_size))
            c = torch.zeros_like(h)
        else:
            h = h_0["base"]
            c = c_0["base"]

        # Put everything on the device (for GPU acceleration)
        h = h.to(self.device)
        c = c.to(self.device)

        # LSTM prediction for the base temperature
        out, (h, c) = self.base_lstm(x[:, :, self.component_inputs_indices], (h, c))
        base_h = h
        base_c = c

        # Some manipulations are needed to feed the output through the neural network if wanted
        if self.NN:

            # Put the data is the form needed for the neural net
            temp = out[:, -1, :].view(out.shape[0], -1)

            # Go through the input layer of the NN
            temp = self.base_nn_input_layer(temp)

            # Iterate through the hidden layers
            for hidden_layer in self.base_nn_hidden_layers:
                temp = hidden_layer(temp)

            # Compute the output and save it at the right place
            base = self.base_nn_output_layer(temp)

        else:
            # Store the outputs
            base = out

        # Get the streaks where the heating mode was on in the batch of inputs
        heating_cases = torch.where(x[:, 0, -1] == 0.9)[0]

        if len(heating_cases) > 0:

            # Define the inputs for the heating cases, and tensors that will retain the heating effect,
            # the hidden states and cell states
            heating_input = x[heating_cases, :, :]
            heating = {part: torch.zeros(x.shape[0], self.output_size).to(self.device)
                           for part in self.parts}
            heating_h = {part: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                         for part in self.parts}
            heating_c = {part: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                         for part in self.parts}

            # Get the batches where some heating actually happened (i.e. the valves where open) - otherwise
            # the heating effect has to be zero
            mask = torch.where(heating_input[:,-1,0] > 0.10001)[0]

            # Heating effect on both temperature and energy consumption
            for part in self.parts:

                # Define the hidden state and cell state as zeros per default, else take the given ones
                if h_0 is None:
                    assert c_0 is None, "What??"
                    h = torch.zeros((self.num_layers, len(heating_cases), self.hidden_size))
                    c = torch.zeros_like(h)
                else:
                    h = h_0["heating"][part][:, heating_cases, :]
                    c = c_0["heating"][part][:, heating_cases, :]

                # Put everything to the right device
                h = h.to(self.device)
                c = c.to(self.device)

                # LSTMs prediction
                out, (h, c) = self.heating_lstms[part](heating_input[:, :, self.effects_inputs_indices], (h, c))
                heating_h[part][:, heating_cases, :] = h
                heating_c[part][:, heating_cases, :] = c

                # Compute the actual effects only if the valves were open (through the 'mask' defined before)
                # Note that the hidden and cell states still need to be updated, and were thus taken care of
                # before
                if len(mask) > 0:

                    # Some manipulations are needed to feed the output through the neural network if wanted
                    if self.NN:
                        # Put the data is the form needed for the neural net
                        temp = out[:, -1, :].view(out.shape[0], -1)

                        # Go through the input layer of the NN
                        temp = self.heating_nn_input_layers[part](temp[mask, :])

                        # Iterate through the hidden layers
                        for hidden_layer in self.heating_nn_hidden_layers[part]:
                            temp = hidden_layer(temp)

                        # Compute the output and save it at the right place
                        # Energy case: just save if as is, with a small factor correction
                        if part == "Energy":
                            heating[part][heating_cases[mask], :] = self.heating_nn_output_layers[part](temp) * 0.5

                        # For the temperature, we have a more precise factor correction to ease training. Also, we
                        # add this first term to ensure consistency (it is proportional to the valves opening) and
                        # ensures that the more you open the valves the more you heat the room
                        else:
                            heating[part][heating_cases[mask], :] = \
                                ((heating_input[mask, -1, 0] - 0.1) * 0.625 / int(24 * 60 / self.interval)).view(-1, 1)\
                                + self.heating_nn_output_layers[part](temp) * self.factor

                    else:
                        # Store the outputs
                        heating[part][heating_cases[mask], :] = out[:, -1, :].view(out.shape[0], -1)

        else:
            # No heating cases in the batch of data
            heating_h = None
            heating_c = None
            heating = None

        # Get the streaks where the cooling mode was on in the batch of inputs
        cooling_cases = torch.where(x[:, 0, -1] == 0.1)[0]

        if len(cooling_cases) > 0:

            # Define the inputs for the cooling cases, and tensors that will retain the cooling effect,
            # the hidden states and cell states
            cooling_input = x[cooling_cases, :, :]
            cooling = {part: torch.zeros(x.shape[0], self.output_size).to(self.device)
                       for part in self.parts}
            cooling_h = {part: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                         for part in self.parts}
            cooling_c = {part: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                         for part in self.parts}

            # Get the batches where some cooling actually happened (i.e. the valves where open) - otherwise
            # the cooling effect has to be zero
            mask = torch.where(cooling_input[:,-1,0] > 0.10001)[0]

            # Heating effect on both temperature and energy consumption
            for part in self.parts:

                # Define the hidden state and cell state as zeros per default, else take the given ones
                if h_0 is None:
                    assert c_0 is None, "What??"
                    h = torch.zeros((self.num_layers, len(cooling_cases), self.hidden_size))
                    c = torch.zeros_like(h)
                else:
                    h = h_0["cooling"][part][:, cooling_cases, :]
                    c = c_0["cooling"][part][:, cooling_cases, :]

                # Put everything to the right device (GPU)
                h = h.to(self.device)
                c = c.to(self.device)

                # LSTMs prediction
                out, (h, c) = self.cooling_lstms[part](cooling_input[:,:, self.effects_inputs_indices], (h, c))
                cooling_h[part][:, cooling_cases, :] = h
                cooling_c[part][:, cooling_cases, :] = c

                # Compute the actual effects only if the valves were open (through the 'mask' defined before)
                # Note that the hidden and cell states still need to be updated, and were thus taken care of
                # before
                if len(mask) > 0:

                    # Some manipulations are needed to feed the output through the neural network if wanted
                    if self.NN:
                        # Put the data is the form needed for the neural net
                        temp = out[:, -1, :].view(out.shape[0], -1)

                        # Go through the input layer of the NN
                        temp = self.cooling_nn_input_layers[part](temp[mask, :])

                        # Iterate through the hidden layers
                        for hidden_layer in self.cooling_nn_hidden_layers[part]:
                            temp = hidden_layer(temp)

                        # Compute the output and save it at the right place
                        # Energy case: just save if as is, with a small factor correction
                        if part == "Energy":
                            cooling[part][cooling_cases[mask], :] = self.cooling_nn_output_layers[part](temp) * 0.5

                        # For the temperature, we have a more precise factor correction to ease training. Also, we
                        # add this first term to ensure consistency (it is proportional to the valves opening) and
                        # ensures that the more you open the valves the more you cool the room
                        else:
                            cooling[part][cooling_cases[mask], :] = \
                                ((cooling_input[mask, -1, 0] - 0.1) * 0.625 / int(24 * 60 / self.interval)).view(-1, 1)\
                                + self.cooling_nn_output_layers[part](temp) * self.factor

                    else:
                        # Store the outputs
                        cooling[part][cooling_cases[mask], :, :] = out[:, -1, :].view(out.shape[0], -1)

        else:
            # No cooling streaks were in the given batch
            cooling_h = None
            cooling_c = None
            cooling = None

        # Put the hidden and cell states in a dictionary to make it tidy
        h_n = {"base": base_h, "heating": heating_h, "cooling": cooling_h}
        c_n = {"base": base_c, "heating": heating_c, "cooling": cooling_c}

        # Return all the network's outputs
        return (base, heating, cooling), (h_n, c_n)

##########################################
### Deprecated old version to keep
##########################################

class LSTM(nn.Module):
    """
    LSTM+NN: this branches the inputs to build one LSTM for each component of the model,
    typically each room, the output of which is fed forward in a neural network to make the
    predictions.
    """

    def __init__(self, device, components_inputs_indices: dict, effects_inputs_indices: dict, factors: dict,
                 zero_energies,
                 hidden_size: int = 1, num_layers: int = 1, NN: bool = True, hidden_sizes: list = [8, 4],
                 output_size: int = 1):
        """
        Function to build the LSTMs, one for each component, and the consecutive NNs.

        The 'components_inputs_indices' dictionary serves to describe which input indices are used
        for each branch. Typically, for a room, we want to take weather information, electricity
        consumption, previous state of the room and actuators into account, among others.

        Args:
            components_inputs_indices: Dictionary linking inputs to the right branches
            hidden_size:               Size of the hidden state of the LSTM cells
            num_layers:                Number of layers ot the LSTM
            NN:                        Flag to set true if the output of the LSTM is to be fed in a feedforward
                                        neural network
            hidden_sizes:              Hidden sizes of the NNs
            output_size:               Output size of the NNs (and thus global output size)
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.components_inputs_indices = components_inputs_indices
        self.effects_inputs_indices = effects_inputs_indices
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.NN = NN
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = device
        self.factors = factors
        self.zero_energies = zero_energies
        self.running_agent = False

        # Build the LSTMs, one for each component
        self._build_lstms()

    def _build_lstms(self) -> None:
        """
        Function to build one LSTM per component
        """

        # Use the dictionary to know the input size of each component to build the LSTMs
        self.base_lstms = nn.ModuleDict({component: nn.LSTM(input_size=len(self.components_inputs_indices[component]),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for component in self.components_inputs_indices.keys() if "Energy" not in component})

        # Build the feedforward networks if needed
        if self.NN:
            # Build the input layers of the NNS
            self.base_nn_input_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_size,
                                                                                     self.hidden_sizes[0]),
                                                                           nn.LeakyReLU())
                                                  for component in self.components_inputs_indices.keys() if "Energy" not in component})

            # Build the hidden layers
            self.base_nn_hidden_layers = nn.ModuleDict({component:
                                                       nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                              self.hidden_sizes[i + 1]),
                                                                                    nn.LeakyReLU())
                                                                      for i in range(0, len(self.hidden_sizes) - 1)])
                                                   for component in self.components_inputs_indices.keys() if "Energy" not in component})

            # Build the output layers for each branch
            self.base_nn_output_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Sigmoid())
                                                   for component in self.components_inputs_indices.keys() if "Energy" not in component})

        # Use the dictionary to know the input size of each component to build the LSTMs
        self.heating_lstms = nn.ModuleDict({component: nn.LSTM(input_size=len(self.effects_inputs_indices[component]),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for component in self.effects_inputs_indices.keys()})

        # Build the feedforward networks if needed
        if self.NN:
            # Build the input layers of the NNS
            self.heating_nn_input_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_size,
                                                                                     self.hidden_sizes[0]),
                                                                           nn.LeakyReLU())
                                                  for component in self.effects_inputs_indices.keys()})

            # Build the hidden layers
            self.heating_nn_hidden_layers = nn.ModuleDict({component:
                                                       nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                              self.hidden_sizes[i + 1]),
                                                                                    nn.LeakyReLU())
                                                                      for i in range(0, len(self.hidden_sizes) - 1)])
                                                   for component in self.effects_inputs_indices.keys()})

            # Build the output layers for each branch
            self.heating_nn_output_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Sigmoid())
                                                   for component in self.effects_inputs_indices.keys()})

        # Use the dictionary to know the input size of each component to build the LSTMs
        self.cooling_lstms = nn.ModuleDict({component: nn.LSTM(input_size=len(self.effects_inputs_indices[component]),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for component in self.effects_inputs_indices.keys()})

        # Build the feedforward networks if needed
        if self.NN:
            # Build the input layers of the NNS
            self.cooling_nn_input_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_size,
                                                                                     self.hidden_sizes[0]),
                                                                           nn.LeakyReLU())
                                                  for component in self.effects_inputs_indices.keys()})

            # Build the hidden layers
            self.cooling_nn_hidden_layers = nn.ModuleDict({component:
                                                       nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                              self.hidden_sizes[i + 1]),
                                                                                    nn.LeakyReLU())
                                                                      for i in range(0, len(self.hidden_sizes) - 1)])
                                                   for component in self.effects_inputs_indices.keys()})

            # Build the output layers for each branch
            self.cooling_nn_output_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Sigmoid())
                                                   for component in self.effects_inputs_indices.keys()})

    def forward(self, x: torch.FloatTensor, h_0: dict = None, c_0: dict = None, warm_start: bool = False,
                past_base=None, past_heating=None, past_cooling=None, past_heating_input=None,
                past_cooling_input=None) -> dict:
        """
        Custom redefinition of the forward path.
        Args:
            x:      Input
            h_0:    Original hidden state, typically when one predict temperatures, this should
                        be the last observed temperature
            c_0:    Original cell state if known
        Returns:
            A dictionary containing the output of the module
        """

        # Build a dictionary and loop over the components to forward the prediction through the layers
        # Also record hidden states and cell states for completeness
        if self.running_agent & (not warm_start):
            assert (past_base is not None) & (past_heating is not None) & (past_cooling is not None),\
                "This shoudn't happen!"

        output = {component: torch.zeros(x.shape[0], x.shape[1], self.output_size).to(self.device)
                  for component in self.effects_inputs_indices.keys() if "Energy" not in component}

        for component in self.effects_inputs_indices.keys():
            if "Energy" in component:
                output[component] = torch.ones(x.shape[0], x.shape[1], self.output_size).to(self.device)
                output[component] *= self.zero_energies[component]

        base = {component: torch.zeros(x.shape[0], x.shape[1], self.output_size).to(self.device)
                  for component in self.components_inputs_indices.keys() if "Energy" not in component}
        base_h_n = {}
        base_c_n = {}

        for component in self.components_inputs_indices:

            if "Energy" not in component:

                # Define the hidden state and cell state as zeros per default
                if h_0 is None:
                    h = torch.zeros((self.num_layers, x.shape[0], self.hidden_size))
                else:
                    h = h_0["base"][component]
                if c_0 is None:
                    c = torch.zeros_like(h)
                else:
                    c = c_0["base"][component]

                h = h.to(self.device)
                c = c.to(self.device)


                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.NN:
                    if (self.running_agent) & (not warm_start):
                        if past_base[component] is not None:
                            if len(torch.nonzero(past_base[component] > 0)):
                                x[:, :, self.components_inputs_indices[component][-6]] = past_base[component].view(1,1,-1)

                    for i in range(x.shape[1]):
                        # LSTMs predictions
                        out, (h, c) = self.base_lstms[component](x[:, i, self.components_inputs_indices[component]]
                                                                 .view(x.shape[0], 1, -1), (h, c))

                        # Put the data is the form needed for the neural net
                        temp = out.view(out.shape[0], -1)

                        # Go through the input layer of the NN
                        temp = self.base_nn_input_layers[component](temp)

                        # Iterate through the hidden layers
                        for hidden_layer in self.base_nn_hidden_layers[component]:
                            temp = hidden_layer(temp)

                        # Compute the output and save it at the right place
                        base[component][:, i, :] = self.base_nn_output_layers[component](temp)
                        if (self.running_agent) & (not warm_start):
                            past_base[component] = self.base_nn_output_layers[component](temp)

                        if (i < x.shape[1]-1) & (not warm_start):
                            x[:, i+1, self.components_inputs_indices[component][-6]] = base[component][:, i, 0]

                else:
                    # LSTMs predictions
                    out, (h, c) = self.base_lstms[component](x[:, :, self.components_inputs_indices[component]], (h, c))

                    # Store the outputs
                    base[component] = out

                base_h_n[component] = h
                base_c_n[component] = c

        heating_cases = torch.where(x[:,0,-1] == 0.9)[0]
        heating = past_heating
        if len(heating_cases) > 0:
            heating_input = x[heating_cases, :, :]

            heating = {component: torch.zeros(heating_input.shape[0], heating_input.shape[1],
                                              self.output_size).to(self.device)
                      for component in self.effects_inputs_indices.keys()}
            heating_h_n = {component: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                      for component in self.effects_inputs_indices.keys()}
            heating_c_n = {component: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                      for component in self.effects_inputs_indices.keys()}

            for component in self.effects_inputs_indices:

                if (not self.running_agent) | (warm_start):
                    mask = torch.where(heating_input[:, :, self.effects_inputs_indices[component][0]]
                                       .max(axis=1).values > 0.1)[0]
                elif (len(torch.nonzero(past_heating[component])) > 0) | (heating_input[0, 0, self.effects_inputs_indices[component][0]].cpu().detach().numpy() > 0.101):
                    mask = [0]
                else:
                    mask = []
                #print(mask)

                if len(mask) > 0:

                    # Define the hidden state and cell state as zeros per default
                    if h_0 is None:
                        h = torch.zeros((self.num_layers, len(mask), self.hidden_size))
                    elif h_0["heating"] is None:
                        h = torch.zeros((self.num_layers, len(mask), self.hidden_size))
                    else:
                        h = h_0["heating"][component][:, mask, :]
                    if c_0 is None:
                        c = torch.zeros_like(h)
                    elif c_0["heating"] is None:
                        c = torch.zeros_like(h)
                    else:
                        c = c_0["heating"][component][:, mask, :]

                    h = h.to(self.device)
                    c = c.to(self.device)

                    # LSTMs predictions
                    #print("input", heating_input[:, :, self.effects_inputs_indices[component]])
                    #print("h", h)
                    #print("c", c)
                    temp = heating_input[:, :, self.effects_inputs_indices[component]]
                    out, (h, c) = self.heating_lstms[component](temp[mask, :, :], (h, c))

                    heating_h_n[component][:, heating_cases[mask], :] = h
                    heating_c_n[component][:, heating_cases[mask], :] = c

                    # Some manipulations are needed to feed the output through the neural network if wanted
                    if self.NN:
                        if (not self.running_agent) | (warm_start):
                            for i in range(out.shape[1]):
                                # Put the data is the form needed for the neural net
                                temp = out[:, i, :].view(out.shape[0], -1)

                                # Go through the input layer of the NN
                                temp = self.heating_nn_input_layers[component](temp)

                                # Iterate through the hidden layers
                                for hidden_layer in self.heating_nn_hidden_layers[component]:
                                    temp = hidden_layer(temp)

                                # Compute the output and save it at the right place
                                if (i == 0) | ("Energy" in component):
                                    heating[component][mask, i, :] = self.heating_nn_output_layers[component](temp)
                                else:
                                    heating[component][mask, i, :] = heating[component][mask, i-1, :] + ((heating_input[mask, i-1, self.effects_inputs_indices[component][0]] - 0.1) * 1.25 / 96).view(-1, 1) + self.heating_nn_output_layers[component](temp)

                        else:
                            # Put the data is the form needed for the neural net
                            temp = out.view(out.shape[0], -1)

                            # Go through the input layer of the NN
                            temp = self.heating_nn_input_layers[component](temp)

                            # Iterate through the hidden layers
                            for hidden_layer in self.heating_nn_hidden_layers[component]:
                                temp = hidden_layer(temp)

                            # Compute the output and save it at the right place
                            if ("Energy" in component) | (len(torch.nonzero(past_heating[component])) == 0):
                                heating[component] = self.heating_nn_output_layers[component](temp)
                                past_heating_input[component] = heating_input[:, :,
                                                                self.effects_inputs_indices[component][0]]
                            else:
                                heating[component] = past_heating[component] + ((past_heating_input[component] - 0.1) * 1.25 / 96).view(-1, 1) + self.heating_nn_output_layers[component](temp)
                                past_heating_input[component] = heating_input[:, :, self.effects_inputs_indices[component][0]]
                    else:
                        # Store the outputs
                        heating[component][mask, :, :] = out
        else:
            heating_h_n = None
            heating_c_n = None

        cooling_cases = torch.where(x[:,0,-1] == 0.1)[0]
        cooling = past_cooling
        if len(cooling_cases) > 0:
            cooling_input = x[cooling_cases, :, :]

            cooling = {component: torch.zeros(cooling_input.shape[0], cooling_input.shape[1], self.output_size).to(self.device)
                      for component in self.effects_inputs_indices.keys()}
            cooling_h_n = {component: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                      for component in self.effects_inputs_indices.keys()}
            cooling_c_n = {component: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                      for component in self.effects_inputs_indices.keys()}

            for component in self.effects_inputs_indices:

                if (not self.running_agent) | (warm_start):
                    mask = torch.where(cooling_input[:,:,self.effects_inputs_indices[component][0]].max(axis=1).values > 0.1)[0]
                #elif (len(torch.nonzero(past_cooling[component])) > 0) | (cooling_input[0, 0, self.effects_inputs_indices[component][0]].cpu().detach().numpy() > 0.101):
                 #   mask = [0]
                else:
                    mask = [0]
                #print(mask)
                if len(mask) > 0:

                    # Define the hidden state and cell state as zeros per default
                    if h_0 is None:
                        h = torch.zeros((self.num_layers, len(mask), self.hidden_size))
                    elif h_0["cooling"] is None:
                        h = torch.zeros((self.num_layers, len(mask), self.hidden_size))
                    else:
                        h = h_0["cooling"][component][:, mask, :]
                    if c_0 is None:
                        c = torch.zeros_like(h)
                    elif c_0["cooling"] is None:
                        c = torch.zeros_like(h)
                    else:
                        c = c_0["cooling"][component][:, mask, :]

                    h = h.to(self.device)
                    c = c.to(self.device)

                    # LSTMs predictions
                    temp = cooling_input[:,:, self.effects_inputs_indices[component]]
                    out, (h, c) = self.cooling_lstms[component](temp[mask, :, :], (h, c))

                    cooling_h_n[component][:, cooling_cases[mask], :] = h
                    cooling_c_n[component][:, cooling_cases[mask], :] = c

                    # Some manipulations are needed to feed the output through the neural network if wanted
                    if self.NN:
                        if (not self.running_agent) | (warm_start):
                            for i in range(out.shape[1]):
                                # Put the data is the form needed for the neural net
                                temp = out[:, i, :].view(out.shape[0], -1)

                                # Go through the input layer of the NN
                                temp = self.cooling_nn_input_layers[component](temp)

                                # Iterate through the hidden layers
                                for hidden_layer in self.cooling_nn_hidden_layers[component]:
                                    temp = hidden_layer(temp)

                                # Compute the output and save it at the right place
                                if (i == 0) | ("Energy" in component):
                                    cooling[component][mask, i, :] = self.cooling_nn_output_layers[component](temp)
                                else:
                                    cooling[component][mask, i, :] = cooling[component][mask, i-1, :] + ((cooling_input[mask, i-1, self.effects_inputs_indices[component][0]] - 0.1) * 0.625 / 96).view(-1, 1) + self.cooling_nn_output_layers[component](temp)

                        else:
                            # Put the data is the form needed for the neural net
                            temp = out.view(out.shape[0], -1)

                            # Go through the input layer of the NN
                            temp = self.cooling_nn_input_layers[component](temp)

                            # Iterate through the hidden layers
                            for hidden_layer in self.cooling_nn_hidden_layers[component]:
                                temp = hidden_layer(temp)

                            # Compute the output and save it at the right place
                            if "Energy" in component:
                                cooling[component] = self.cooling_nn_output_layers[component](temp)
                            else:
                                cooling[component] = past_cooling[component] + ((past_cooling_input[component] - 0.1) * 0.625 / 96).view(-1, 1) + self.cooling_nn_output_layers[component](temp)
                                past_cooling_input[component] = cooling_input[:, :, self.effects_inputs_indices[component][0]]
                    else:
                        # Store the outputs
                        cooling[component][mask, :, :] = out

        else:
            cooling_h_n = None
            cooling_c_n = None

        h_n = {"base": base_h_n, "heating": heating_h_n, "cooling": cooling_h_n}
        c_n = {"base": base_c_n, "heating": heating_c_n, "cooling": cooling_c_n}

        for component in self.effects_inputs_indices:

            if "Energy" not in component:
                output[component] = base[component]
                if len(heating_cases) > 0:
                    output[component][heating_cases, :, :] += (heating[component] * self.factors[component])
                if len(cooling_cases) > 0:
                    output[component][cooling_cases, :, :] -= (cooling[component] * self.factors[component])

            else:
                if len(heating_cases) > 0:
                    output[component][heating_cases, :, :] += (heating[component] * 0.5)
                if len(cooling_cases) > 0:
                    output[component][cooling_cases, :, :] -= (cooling[component] * 0.5)

        # Return the network's outputs
        if (not self.running_agent) | (warm_start):
            return output, (h_n, c_n)
        else:
            return output, (h_n, c_n), past_base, heating, cooling, past_heating_input, past_cooling_input


class LSTM2(nn.Module):
    """
    LSTM+NN: this branches the inputs to build one LSTM for each component of the model,
    typically each room, the output of which is fed forward in a neural network to make the
    predictions.
    """

    def __init__(self, device, components_inputs_indices: dict, effects_inputs_indices: dict, factors: dict,
                 zero_energies,
                 hidden_size: int = 1, num_layers: int = 1, NN: bool = True, hidden_sizes: list = [8, 4],
                 output_size: int = 1):
        """
        Function to build the LSTMs, one for each component, and the consecutive NNs.

        The 'components_inputs_indices' dictionary serves to describe which input indices are used
        for each branch. Typically, for a room, we want to take weather information, electricity
        consumption, previous state of the room and actuators into account, among others.

        Args:
            components_inputs_indices: Dictionary linking inputs to the right branches
            hidden_size:               Size of the hidden state of the LSTM cells
            num_layers:                Number of layers ot the LSTM
            NN:                        Flag to set true if the output of the LSTM is to be fed in a feedforward
                                        neural network
            hidden_sizes:              Hidden sizes of the NNs
            output_size:               Output size of the NNs (and thus global output size)
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.components_inputs_indices = components_inputs_indices
        self.effects_inputs_indices = effects_inputs_indices
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.NN = NN
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = device
        self.factors = factors
        self.zero_energies = zero_energies

        # Build the LSTMs, one for each component
        self._build_lstms()

    def _build_lstms(self) -> None:
        """
        Function to build one LSTM per component
        """

        # Use the dictionary to know the input size of each component to build the LSTMs
        self.base_lstms = nn.ModuleDict({component: nn.LSTM(input_size=len(self.components_inputs_indices[component]),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for component in self.components_inputs_indices.keys() if "Energy" not in component})

        # Build the feedforward networks if needed
        if self.NN:
            # Build the input layers of the NNS
            self.base_nn_input_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_size,
                                                                                     self.hidden_sizes[0]),
                                                                           nn.LeakyReLU())
                                                  for component in self.components_inputs_indices.keys() if "Energy" not in component})

            # Build the hidden layers
            self.base_nn_hidden_layers = nn.ModuleDict({component:
                                                       nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                              self.hidden_sizes[i + 1]),
                                                                                    nn.LeakyReLU())
                                                                      for i in range(0, len(self.hidden_sizes) - 1)])
                                                   for component in self.components_inputs_indices.keys() if "Energy" not in component})

            # Build the output layers for each branch
            self.base_nn_output_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Sigmoid())
                                                   for component in self.components_inputs_indices.keys() if "Energy" not in component})

        # Use the dictionary to know the input size of each component to build the LSTMs
        self.heating_lstms = nn.ModuleDict({component: nn.LSTM(input_size=len(self.effects_inputs_indices[component]),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for component in self.effects_inputs_indices.keys()})

        # Build the feedforward networks if needed
        if self.NN:
            # Build the input layers of the NNS
            self.heating_nn_input_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_size,
                                                                                     self.hidden_sizes[0]),
                                                                           nn.LeakyReLU())
                                                  for component in self.effects_inputs_indices.keys()})

            # Build the hidden layers
            self.heating_nn_hidden_layers = nn.ModuleDict({component:
                                                       nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                              self.hidden_sizes[i + 1]),
                                                                                    nn.LeakyReLU())
                                                                      for i in range(0, len(self.hidden_sizes) - 1)])
                                                   for component in self.effects_inputs_indices.keys()})

            # Build the output layers for each branch
            self.heating_nn_output_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Sigmoid())
                                                   for component in self.effects_inputs_indices.keys()})

        # Use the dictionary to know the input size of each component to build the LSTMs
        self.cooling_lstms = nn.ModuleDict({component: nn.LSTM(input_size=len(self.effects_inputs_indices[component]),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for component in self.effects_inputs_indices.keys()})

        # Build the feedforward networks if needed
        if self.NN:
            # Build the input layers of the NNS
            self.cooling_nn_input_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_size,
                                                                                     self.hidden_sizes[0]),
                                                                           nn.LeakyReLU())
                                                  for component in self.effects_inputs_indices.keys()})

            # Build the hidden layers
            self.cooling_nn_hidden_layers = nn.ModuleDict({component:
                                                       nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                              self.hidden_sizes[i + 1]),
                                                                                    nn.LeakyReLU())
                                                                      for i in range(0, len(self.hidden_sizes) - 1)])
                                                   for component in self.effects_inputs_indices.keys()})

            # Build the output layers for each branch
            self.cooling_nn_output_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Sigmoid())
                                                   for component in self.effects_inputs_indices.keys()})

    def forward(self, x: torch.FloatTensor, h_0: dict = None, c_0: dict = None, warm_start: bool = False) -> dict:
        """
        Custom redefinition of the forward path.
        Args:
            x:      Input
            h_0:    Original hidden state, typically when one predict temperatures, this should
                        be the last observed temperature
            c_0:    Original cell state if known
        Returns:
            A dictionary containing the output of the module
        """
        # Build a dictionary and loop over the components to forward the prediction through the layers
        # Also record hidden states and cell states for completeness

        output = {component: torch.zeros(x.shape[0], x.shape[1], self.output_size).to(self.device)
                  for component in self.effects_inputs_indices.keys() if "Energy" not in component}

        for component in self.effects_inputs_indices.keys():
            if "Energy" in component:
                output[component] = torch.ones(x.shape[0], x.shape[1], self.output_size).to(self.device)
                output[component] *= self.zero_energies[component]

        base = {component: torch.zeros(x.shape[0], x.shape[1], self.output_size).to(self.device)
                  for component in self.components_inputs_indices.keys() if "Energy" not in component}
        base_h_n = {}
        base_c_n = {}

        for component in self.components_inputs_indices:

            if "Energy" not in component:

                # Define the hidden state and cell state as zeros per default
                if h_0 is None:
                    h = torch.zeros((self.num_layers, x.shape[0], self.hidden_size))
                else:
                    h = h_0["base"][component]
                if c_0 is None:
                    c = torch.zeros_like(h)
                else:
                    c = c_0["base"][component]

                h = h.to(self.device)
                c = c.to(self.device)


                # Some manipulations are needed to feed the output through the neural network if wanted
                if self.NN:
                    for i in range(x.shape[1]):
                        # LSTMs predictions
                        out, (h, c) = self.base_lstms[component](x[:, i, self.components_inputs_indices[component]].view(x.shape[0], 1, -1), (h, c))

                        # Put the data is the form needed for the neural net
                        temp = out.view(out.shape[0], -1)

                        # Go through the input layer of the NN
                        temp = self.base_nn_input_layers[component](temp)

                        # Iterate through the hidden layers
                        for hidden_layer in self.base_nn_hidden_layers[component]:
                            temp = hidden_layer(temp)

                        # Compute the output and save it at the right place
                        base[component][:, i, :] = self.base_nn_output_layers[component](temp)

                        if (i < x.shape[1]-1) & (not warm_start):
                            x[:, i+1, self.components_inputs_indices[component][-6]] = base[component][:, i, 0]

                    base_h_n[component] = h
                    base_c_n[component] = c

                else:
                    # LSTMs predictions
                    out, (h, c) = self.base_lstms[component](x[:, :, self.components_inputs_indices[component]], (h, c))

                    # Store the outputs
                    base[component] = out

        heating_cases = torch.where(x[:,0,-1] == 0.9)[0]
        if len(heating_cases) > 0:
            heating_input = x[heating_cases, :, :]

            heating = {component: torch.zeros(heating_input.shape[0], heating_input.shape[1], self.output_size).to(self.device)
                      for component in self.effects_inputs_indices.keys()}
            heating_h_n = {component: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                      for component in self.effects_inputs_indices.keys()}
            heating_c_n = {component: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                      for component in self.effects_inputs_indices.keys()}

            for component in self.effects_inputs_indices:

                mask = torch.where(heating_input[:,:,self.effects_inputs_indices[component][0]].max(axis=1).values > 0.1)[0]

                if len(mask) > 0:

                    # Define the hidden state and cell state as zeros per default
                    if h_0 is None:
                        h = torch.zeros((self.num_layers, len(mask), self.hidden_size))
                    elif h_0["heating"] is None:
                        h = torch.zeros((self.num_layers, len(mask), self.hidden_size))
                    else:
                        h = h_0["heating"][component][:, mask, :]
                    if c_0 is None:
                        c = torch.zeros_like(h)
                    elif c_0["heating"] is None:
                        c = torch.zeros_like(h)
                    else:
                        c = c_0["heating"][component][:, mask, :]

                    h = h.to(self.device)
                    c = c.to(self.device)

                    # LSTMs predictions
                    temp = heating_input[:, :, self.effects_inputs_indices[component]]
                    out, (h, c) = self.heating_lstms[component](temp[mask, :, :], (h, c))

                    heating_h_n[component][:, heating_cases[mask], :] = h
                    heating_c_n[component][:, heating_cases[mask], :] = c

                    # Some manipulations are needed to feed the output through the neural network if wanted
                    if self.NN:
                        for i in range(out.shape[1]):
                            # Put the data is the form needed for the neural net
                            temp = out[:, i, :].view(out.shape[0], -1)

                            # Go through the input layer of the NN
                            temp = self.heating_nn_input_layers[component](temp)

                            # Iterate through the hidden layers
                            for hidden_layer in self.heating_nn_hidden_layers[component]:
                                temp = hidden_layer(temp)

                            # Compute the output and save it at the right place
                            if (i == 0) | ("Energy" in component):
                                heating[component][mask, i, :] = self.heating_nn_output_layers[component](temp)
                            else:
                                heating[component][mask, i, :] = heating[component][mask, i-1, :] + ((heating_input[mask,i-1,self.effects_inputs_indices[component][0]] - 0.1) * 1.25 / 96).view(-1, 1) + self.heating_nn_output_layers[component](temp)

                    else:
                        # Store the outputs
                        heating[component][mask, :, :] = out

        else:
            heating_h_n = None
            heating_c_n = None

        cooling_cases = torch.where(x[:,0,-1] == 0.1)[0]
        if len(cooling_cases) > 0:
            cooling_input = x[cooling_cases, :, :]


            cooling = {component: torch.zeros(cooling_input.shape[0], cooling_input.shape[1], self.output_size).to(self.device)
                      for component in self.effects_inputs_indices.keys()}
            cooling_h_n = {component: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                      for component in self.effects_inputs_indices.keys()}
            cooling_c_n = {component: torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
                      for component in self.effects_inputs_indices.keys()}

            for component in self.effects_inputs_indices:

                mask = torch.where(cooling_input[:,:,self.effects_inputs_indices[component][0]].max(axis=1).values > 0.1)[0]

                if len(mask) > 0:

                    # Define the hidden state and cell state as zeros per default
                    if h_0 is None:
                        h = torch.zeros((self.num_layers, len(mask), self.hidden_size))
                    elif h_0["cooling"] is None:
                        h = torch.zeros((self.num_layers, len(mask), self.hidden_size))
                    else:
                        h = h_0["cooling"][component][:, mask, :]
                    if c_0 is None:
                        c = torch.zeros_like(h)
                    elif c_0["cooling"] is None:
                        c = torch.zeros_like(h)
                    else:
                        c = c_0["cooling"][component][:, mask, :]

                    h = h.to(self.device)
                    c = c.to(self.device)

                    # LSTMs predictions
                    temp = cooling_input[:,:, self.effects_inputs_indices[component]]
                    out, (h, c) = self.cooling_lstms[component](temp[mask, :, :], (h, c))

                    cooling_h_n[component][:, cooling_cases[mask], :] = h
                    cooling_c_n[component][:, cooling_cases[mask], :] = c

                    # Some manipulations are needed to feed the output through the neural network if wanted
                    if self.NN:
                        for i in range(out.shape[1]):
                            # Put the data is the form needed for the neural net
                            temp = out[:, i, :].view(out.shape[0], -1)

                            # Go through the input layer of the NN
                            temp = self.cooling_nn_input_layers[component](temp)

                            # Iterate through the hidden layers
                            for hidden_layer in self.cooling_nn_hidden_layers[component]:
                                temp = hidden_layer(temp)

                            # Compute the output and save it at the right place
                            if (i == 0) | ("Energy" in component):
                                cooling[component][mask, i, :] = self.cooling_nn_output_layers[component](temp)
                            else:
                                cooling[component][mask, i, :] = cooling[component][mask, i-1, :] + ((cooling_input[mask,i-1,self.effects_inputs_indices[component][0]] - 0.1) * 0.625 / 96).view(-1, 1) + self.cooling_nn_output_layers[component](temp)

                    else:
                        # Store the outputs
                        cooling[component][mask, :, :] = out

        else:
            cooling_h_n = None
            cooling_c_n = None

        h_n = {"base": base_h_n, "heating": heating_h_n, "cooling": cooling_h_n}
        c_n = {"base": base_c_n, "heating": heating_c_n, "cooling": cooling_c_n}

        for component in self.effects_inputs_indices:

            if "Energy" not in component:
                output[component] = base[component]
                if len(heating_cases) > 0:
                    output[component][heating_cases, :, :] += (heating[component] * self.factors[component])
                if len(cooling_cases) > 0:
                    output[component][cooling_cases, :, :] -= (cooling[component] * self.factors[component])

            else:
                if len(heating_cases) > 0:
                    output[component][heating_cases, :, :] += (heating[component] * 0.5)
                if len(cooling_cases) > 0:
                    output[component][cooling_cases, :, :] -= (cooling[component] * 0.5)

        # Return the network's outputs
        return output, (h_n, c_n)


class LSTM_old(nn.Module):
    """
    LSTM+NN: this branches the inputs to build one LSTM for each component of the model,
    typically each room, the output of which is fed forward in a neural network to make the
    predictions.
    """

    def __init__(self, device, components_inputs_indices: dict, hidden_size: int = 1, num_layers: int = 1,
                 NN: bool = True, hidden_sizes: list = [8, 4], output_size: int = 1):
        """
        Function to build the LSTMs, one for each component, and the consecutive NNs.

        The 'components_inputs_indices' dictionary serves to describe which input indices are used
        for each branch. Typically, for a room, we want to take weather information, electricity
        consumption, previous state of the room and actuators into account, among others.

        Args:
            components_inputs_indices: Dictionary linking inputs to the right branches
            hidden_size:               Size of the hidden state of the LSTM cells
            num_layers:                Number of layers ot the LSTM
            NN:                        Flag to set true if the output of the LSTM is to be fed in a feedforward
                                        neural network
            hidden_sizes:              Hidden sizes of the NNs
            output_size:               Output size of the NNs (and thus global output size)
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.components_inputs_indices = components_inputs_indices
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.NN = NN
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = device

        # Build the LSTMs, one for each component
        self._build_lstms()

    def _build_lstms(self) -> None:
        """
        Function to build one LSTM per component
        """

        # Use the dictionary to know the input size of each component to build the LSTMs
        self.lstms = nn.ModuleDict({component: nn.LSTM(input_size=len(self.components_inputs_indices[component]),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for component in self.components_inputs_indices.keys()})

        # Build the feedforward networks if needed
        if self.NN:
            # Build the input layers of the NNS
            self.nn_input_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_size,
                                                                                     self.hidden_sizes[0]),
                                                                           nn.LeakyReLU())
                                                  for component in self.components_inputs_indices.keys()})

            # Build the hidden layers
            self.nn_hidden_layers = nn.ModuleDict({component:
                                                       nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                              self.hidden_sizes[i + 1]),
                                                                                    nn.LeakyReLU())
                                                                      for i in range(0, len(self.hidden_sizes) - 1)])
                                                   for component in self.components_inputs_indices.keys()})

            # Build the output layers for each branch
            self.nn_output_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Sigmoid())
                                                   for component in self.components_inputs_indices.keys()})

    def forward(self, x: torch.FloatTensor, h_0: dict = None, c_0: dict = None) -> dict:
        """
        Custom redefinition of the forward path.
        Args:
            x:      Input
            h_0:    Original hidden state, typically when one predict temperatures, this should
                        be the last observed temperature
            c_0:    Original cell state if known
        Returns:
            A dictionary containing the output of the module
        """

        # Build a dictionary and loop over the components to forward the prediction through the layers
        # Also record hidden states and cell states for completeness
        output = {component: torch.zeros(x.shape[0], x.shape[1], self.output_size).to(self.device)
                  for component in self.components_inputs_indices.keys()}
        h_n = {}
        c_n = {}

        for component in self.components_inputs_indices:

            # Define the hidden state and cell state as zeros per default
            if h_0 is None:
                h = torch.zeros((self.num_layers, x.shape[0], self.hidden_size))
            else:
                h = h_0[component]
            if c_0 is None:
                c = torch.zeros_like(h)
            else:
                c = c_0[component]

            h = h.to(self.device)
            c = c.to(self.device)

            # LSTMs predictions
            out, (h, c) = self.lstms[component](x[:, :, self.components_inputs_indices[component]], (h, c))

            h_n[component] = h#.to(self.device)
            c_n[component] = c#.to(self.device)

            # Some manipulations are needed to feed the output through the neural network if wanted
            if self.NN:
                for i in range(out.shape[1]):
                    # Put the data is the form needed for the neural net
                    temp = out[:, i, :].view(out.shape[0], -1)#.to(self.device)

                    # Go through the input layer of the NN
                    temp = self.nn_input_layers[component](temp)#.to(self.device)

                    # Iterate through the hidden layers
                    for hidden_layer in self.nn_hidden_layers[component]:
                        temp = hidden_layer(temp)#.to(self.device)

                    # Compute the output and save it at the right place
                    output[component][:, i, :] = self.nn_output_layers[component](temp)#.to(self.device)

            else:
                # Store the outputs
                output[component] = out#.to(self.device)

        # Return the network's outputs
        return output, (h_n, c_n)
