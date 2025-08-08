"""
File containing custom PyTorch modules to build NN models
"""

import torch
from torch import nn


class NN(nn.Module):
    """
    Basic Neural Network class with branches for different components, typically different rooms.
    The number of branches, inputs and outputs are flexible.
    """

    def __init__(self, components_inputs_indices: dict, hidden_sizes=None, output_size: int = 1) -> None:
        """
        Initializing the module by building a branched NN, typically one for each room.

        The dictionary of 'components_inputs_indices' serves to describe which input indices are used
        for each branch. Typically, for a room, we want to take weather information, electricity
        consumption, previous state of the room and actuators into account.

        Args:
            components_inputs_indices: Dictionary relating each component (branch) of the network to the
                                        input indices to take into account for this branch
            hidden_sizes:              Sizes of the hidden layers (currently all branches the same)
            output_size:               Output size of each branch
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Define global parameters
        if hidden_sizes is None:
            hidden_sizes = [10, 5]
        self.components_inputs_indices = components_inputs_indices
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Build the branched NN
        self._build_nn()

    def _build_nn(self) -> None:
        """
        Custom construction of the branched NN
        """

        # Build one input layer for each component according to the wanted input indices
        self.input_layers = nn.ModuleDict({component:
                                               nn.Sequential(nn.Linear(len(self.components_inputs_indices[component]),
                                                                       self.hidden_sizes[0]), nn.LeakyReLU())
                                           for component in self.components_inputs_indices.keys()})

        # Build all the hidden layers for each branch
        self.hidden_layers = nn.ModuleDict({component: nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                              self.hidden_sizes[i+1]),
                                                                                    nn.LeakyReLU())
                                                                      for i in range(0, len(self.hidden_sizes)-1)])
                                            for component in self.components_inputs_indices.keys()})

        # Build the output layers for each branch
        self.output_layers = nn.ModuleDict({component: nn.Linear(self.hidden_sizes[-1],
                                                                 self.output_size)
                                            for component in self.components_inputs_indices.keys()})

    def forward(self, x: torch.FloatTensor) -> dict:
        """
        Custom redefinition of a forward path in the designed branched NN.

        Args:
            x:      The input

        Returns:
            output: Dictionary of tensors contraining the NN outputs
        """

        # Define the output dictionary and iterate over the components to get their output
        output = {}
        for component in self.components_inputs_indices.keys():

            # Go through the input layer, using the defined indices for each component
            temp = self.input_layers[component](x[:, self.components_inputs_indices[component]])

            # Iterate through the hidden layers
            for hidden_layer in self.hidden_layers[component]:
                temp = hidden_layer(temp)

            # Compute the output
            output[component] = self.output_layers[component](temp)

        # Return the network's outputs
        return output


class NNExtraction(NN):
    """
    Extends the base NN process with feature extraction layer(s) in the beginning
    """

    def __init__(self, components_inputs_indices: dict, autoregressive_input_indices: dict,
                 feature_extraction_sizes=None, hidden_sizes=None, output_size: int = 1) -> None:
        """
        Initializing the module by building a branched NN, typically one for each room, using the
        NESTNN module. Additionally, layers are added in the beginning to extract features from
        autoregressive terms.

        The dictionary of 'components_inputs_indices' serves to describe which input indices are used
        for each branch. Typically, for a room, we want to take weather information, electricity
        consumption, previous state of the room and actuators into account.

        The dictionary of 'autoregressive_input_indices' describes which input indices
        correspond to each autoregressive terms for the sensors to build the feature extraction

        Args:
            components_inputs_indices:    Dictionary relating each component (branch) of the network to the
                                                   input indices to take into account for this branch
            autoregressive_input_indices: Dictionary relating input indices to each sensor
            hidden_sizes:                 Sizes of the hidden layers (currently all branches the same)
            output_size:                  Output size of each branch
        """

        # Start by initializing a PyTorch module
        super().__init__(components_inputs_indices, hidden_sizes, output_size)

        # Define global parameters
        if feature_extraction_sizes is None:
            feature_extraction_sizes = [2, 1]
        self.autoregressive_input_indices = autoregressive_input_indices
        self.feature_extraction_sizes = feature_extraction_sizes

        # Build the feature extraction layers
        self._build_feature_extraction()

    def _build_feature_extraction(self) -> None:
        """
        Custom construction of the feature extraction layers
        """

        # Build one extraction layer for each autoregressive sensor according to the wanted input indices
        self.feature_extraction_input_layer = \
            nn.ModuleDict({sensor: nn.Sequential(nn.Linear(len(self.autoregressive_input_indices[sensor]),
                                                           self.feature_extraction_sizes[0]),
                                                 nn.LeakyReLU())
                           for sensor in self.autoregressive_input_indices.keys()
                           if len(self.autoregressive_input_indices[sensor]) > 0})

        # Build all the hidden layers for each feature to extract
        self.feature_extraction_layers = \
            nn.ModuleDict({sensor: nn.ModuleList([nn.Sequential(nn.Linear(self.feature_extraction_sizes[i],
                                                                          self.feature_extraction_sizes[i + 1]),
                                                                nn.LeakyReLU())
                                                  for i in range(0, len(self.feature_extraction_sizes) - 1)])
                           for sensor in self.autoregressive_input_indices.keys()
                           if len(self.autoregressive_input_indices[sensor]) > 0})

    def forward(self, x: torch.FloatTensor) -> dict:
        """
        Custom redefinition of a forward path in the designed NN.

        Args:
            x:      The input

        Returns:
            output: Dictionary of tensors contraining the NN outputs
        """

        # Build the features from past data for all autoregressive sensors
        features = []
        for num, sensor in enumerate(self.autoregressive_input_indices.keys()):

            # If it is an autoregressive sensor, pass it through the custom extraction layer
            if len(self.autoregressive_input_indices[sensor]) > 0:

                # Go through the input layer, using the defined indices for each component
                temp = self.feature_extraction_input_layer[sensor](
                    x[:, self.autoregressive_input_indices[sensor]])

                # Go through the hidden layers
                for hidden_layer in self.feature_extraction_layers[sensor]:
                    temp = hidden_layer(temp)

                features.append(temp)

            # Else, it is not autoregressive, just pass the input directly
            else:
                features.append(x[:, num].view(-1, 1))

        # Concatenate the features to input to the branched network
        features = torch.cat(features, dim=1)

        # Define the output dictionary and iterate over the components to get their output
        output = {}
        for component in self.components_inputs_indices.keys():

            # Go through the input layer, using the defined indices for each component
            temp = self.input_layers[component](features[:, self.components_inputs_indices[component]])

            # Iterate through the hidden layers
            for hidden_layer in self.hidden_layers[component]:
                temp = hidden_layer(temp)

            # Compute the output
            output[component] = self.output_layers[component](temp)

        # Return the network's outputs
        return output


class LSTMExtraction(nn.Module):
    """
    Basic LSTM module: this branches the inputs to build one LSTM for each component of the model,
    typically each room
    """

    def __init__(self, components_inputs_indices: dict, hidden_size: int = 1, num_layers: int = 1):
        """
        Function to build the LSTMs, one for each component.

        The 'components_inputs_indices' dictionary serves to describe which input indices are used
        for each branch. Typically, for a room, we want to take weather information, electricity
        consumption, previous state of the room and actuators into account.

        Args:
            components_inputs_indices: Dictionary linking inputs to the right branches
            hidden_size:               Size of the hidden state of the LSTM cells
            num_layers:                Number of layers ot the LSTM
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.components_inputs_indices = components_inputs_indices
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Build the LSTMs, one for each component
        self._build_lstms()

    def _build_lstms(self) -> None:
        """
        Function to build one LSTM per component
        """

        # Use the dictionary to know the input size of each component
        self.lstms = nn.ModuleDict({component: nn.LSTM(input_size=len(self.components_inputs_indices[component]),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for component in self.components_inputs_indices.keys()})

    def forward(self, x: torch.FloatTensor) -> dict:
        """
        Custom redefinition of the forward path, simply passing the right input to each branch. Note
        that we only keep the last output, because we want to predict the next value only.
        Args:
            x: input
        Returns:
            A dictionary containing the last output of the LSTM for each component
        """

        # Build a dictionary and loop over the components to forward the prediction
        output = {}
        for component in self.components_inputs_indices:
            output[component], _ = self.lstms[component](x[:, :, self.components_inputs_indices[component]])

            # We only need to recall the last prediction
            output[component] = output[component][:, -1, :]

        # Return the network's outputs
        return output


class LSTMNNExtraction(nn.Module):
    """
    LSTM+NN: this branches the inputs to build one LSTM for each component of the model,
    typically each room, the output of which is fed forward in a neural network to make the
    predictions.
    """

    def __init__(self, components_inputs_indices: dict, hidden_size: int = 1, num_layers: int = 1,
                 hidden_sizes: list = [8, 4], output_size: int = 1, predict_differences=True):
        """
        Function to build the LSTMs, one for each component, and the consecutive NNs.

        The 'components_inputs_indices' dictionary serves to describe which input indices are used
        for each branch. Typically, for a room, we want to take weather information, electricity
        consumption, previous state of the room and actuators into account, among others.

        Args:
            components_inputs_indices: Dictionary linking inputs to the right branches
            hidden_size:               Size of the hidden state of the LSTM cells
            num_layers:                Number of layers ot the LSTM
            hidden_sizes:              Hidden sizes of the NNs
            output_size:               Output size of the NNs (and thus global output size)
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.components_inputs_indices = components_inputs_indices
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.predict_differences = predict_differences

        # Build the LSTMs, one for each component
        self._build_lstms_NN()

    def _build_lstms_NN(self) -> None:
        """
        Function to build one LSTM and then one NN per component
        """

        # Use the dictionary to know the input size of each component to build the LSTMs
        self.lstms = nn.ModuleDict({component: nn.LSTM(input_size=len(self.components_inputs_indices[component]),
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       batch_first=True)
                                    for component in self.components_inputs_indices.keys()})

        # Build the input layers of the NNS
        self.nn_input_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_size,
                                                                                 self.hidden_sizes[0]),
                                                                       nn.LeakyReLU())
                                              for component in self.components_inputs_indices.keys()})

        # Build the hidden layers
        self.nn_hidden_layers = nn.ModuleDict({component:
                                                   nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_sizes[i],
                                                                                          self.hidden_sizes[i+1]),
                                                                                nn.LeakyReLU())
                                                                  for i in range(0, len(self.hidden_sizes) - 1)])
                                               for component in self.components_inputs_indices.keys()})

        # Build the output layers for each branch
        if self.predict_differences:
            self.nn_output_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Tanh())
                                                   for component in self.components_inputs_indices.keys()})
        else:
            self.nn_output_layers = nn.ModuleDict({component: nn.Sequential(nn.Linear(self.hidden_sizes[-1],
                                                                                      self.output_size),
                                                                            nn.Sigmoid())
                                              for component in self.components_inputs_indices.keys()})

    def forward(self, x: torch.FloatTensor) -> dict:
        """
        Custom redefinition of the forward path.
        Args:
            x: input
        Returns:
            A dictionary containing the output of the module
        """

        # Build a dictionary and loop over the components to forward the prediction through the layers
        output = {}
        for component in self.components_inputs_indices:

            # LSTMs predictions
            out, _ = self.lstms[component](x[:, :, self.components_inputs_indices[component]])

            # We only need to recall the last prediction
            out = out[:, -1, :]

            # Prepare the output for the feedforward NN, we need to reduce the dimension because LSTMs
            # make predictions over sequences of data whereas NNs only take static inputs
            temp = out.view(out.shape[0], -1)

            # Go through the input layer of the NN
            temp = self.nn_input_layers[component](temp)

            # Iterate through the hidden layers
            for hidden_layer in self.nn_hidden_layers[component]:
                temp = hidden_layer(temp)

            # Compute the output
            output[component] = self.nn_output_layers[component](temp)

        # Return the network's outputs
        return output
