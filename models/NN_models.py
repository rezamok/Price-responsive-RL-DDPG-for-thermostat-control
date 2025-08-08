# -*- coding: utf-8 -*-
"""
Specific Neural Networks NEST models
"""
import os
import sys
import torch
from torch import optim

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import kosta.hyperparams as hp

from models.base_models import BaseModel
from models.data_handling import NESTData
from models.NN_modules import NN, NNExtraction, LSTMExtraction, LSTMNNExtraction

from util.util import flat_list, compute_mae
from util.plots import _save_or_show, _plot_helpers

from models.parameters import model_kwargs


class ARX:
    """
    ARX model definition, based on Felix Bünnning's provided coefficients. This just loads the
    coefficient of the linear model, creates training and testing sets and provides
    a "model" function that is queried by gym environments to create trajectories
    """
    def __init__(self, data, min_, max_, model_kwargs):
        """
        the data has to be normalized, with the mins and maxs provided. This is to comply with the general
        form of data with recurrent networks, with agents also accessing electricity price data
        and comfort bounds information
        Args:
            data:           Normalized data of the corresponding model (ambient conditions)
            min_:           Corresponding minimums
            max_:           Corresponding maximums
            model_kwargs:   All the model kwargs as usual
        """

        # This model is only defined on 3 rooms
        self.rooms = model_kwargs['rooms']#["272", "273", "274"]
        print(f'Environment created for room {self.rooms}')
        self.n_autoregression = model_kwargs["n_autoregression"]
        self.threshold_length = model_kwargs["threshold_length"]
        self.heating = model_kwargs["heating"]
        self.cooling = model_kwargs["cooling"]
        self.interval = model_kwargs["interval"]
        self.save_path = model_kwargs["save_path"]
        self.name = model_kwargs["model_name"]
        self.heating = model_kwargs["heating"]
        self.cooling = model_kwargs["cooling"]
        self.min_ = min_
        self.max_ = max_
        self.data = data

        # Build the models (one for each room), prepare a DataFrame to receive current observations
        # from which the model will predict the next temperatures
        self.room_models = self.build_models(self.save_path)
        self.observation = pd.DataFrame(index=["Coeffs"], columns=self.data.columns, dtype=float)
        self.train_indices, self.test_indices = self.train_test_separation()

    def build_models(self, save_path: str = None):
        """
        Function creating one model for each room by downloading the according coefficients
        and preprocessing them
        Args:
            save_path:  Where to find the data
        Returns:
            Dictionary of models, which are just coefficients in a DataFrame
        """
        if save_path is None:
            save_path = self.save_path

        # Download the models from the path, i.e. the "model" is ust the coefficients
        models_path = {room: os.path.join(save_path, self.name, room + "_" + \
                                          str(self.interval) + ".csv") for room in self.rooms}
        models = {room: pd.read_csv(models_path[room], index_col=[0], names=["Coeffs"]) for room in models_path.keys()}
        [model.drop(index=[np.nan], inplace=True) for model in models.values()]

        # Preprocessing: this is a linear model, each coefficient multiplies the current value of the
        # corresponding variable. For this to work, we have to ensure the coefficient and variables
        # have the same name
        for room in models.keys():
            models[room] = models[room].transpose()
            for column in models[room].columns:
                # Drop the "neigbor" term in the names of neighboring rooms so the model can find the wanted
                # temperature in the data
                if "neighbor" in column:
                    models[room].rename(columns={column: column.split("_neighbor")[0] + column.split("_neighbor")[1]},
                                        inplace=True)
            models[room].loc[:,'uk'] = models[room]['uk']*hp.uk_coeff_scaling
            # Rename the power input and room temperature using the actual room name 
            models[room].rename(columns={"uk": f"u_{room}", "T_room": f"T_{room}"}, inplace=True)

        return models

    def train_test_separation(self):
        """
        Function to separate the data in training and testing indices
        """
        # List the indices
        indices = np.arange(len(self.data.index))

        # Only keep heating or cooling indices if wanted
        if not self.cooling:
            indices = indices[(self.data.iloc[:, -4] > 0.5).values]
        if not self.heating:
            indices = indices[(self.data.iloc[:, -4] < 0.5).values]
            
        indices = list(indices)

        # Separate into training and testing, taking 1 month in summer and 1 month in winter as testing set
        test_indices = indices[self.n_autoregression: (len(indices) // 12) - self.threshold_length] + \
                       indices[6 * (len(indices) // 12) + self.n_autoregression: 7 * (
                                   len(indices) // 12) - self.threshold_length]
        train_indices = indices[(len(indices) // 12) + self.n_autoregression: 6 * (
                    len(indices) // 12) - self.threshold_length] + \
                        indices[7 * (len(indices) // 12) + self.n_autoregression: - self.threshold_length]

        return train_indices, test_indices

    def model(self, observation):
        """
        Function modeling the next temperature given the current observation
        """

        # Put the observation in the dataFrame (to give them the right name)
        self.observation.iloc[0, :] = observation
        output = []

        # Multiply the coefficients of each model with the observation and sum them
        for room in self.rooms:
            output.append(np.sum(np.sum(self.room_models[room] * self.observation)))

        return np.array(output)


class NNBase(BaseModel):
    """
    Base class for NEST modeling, mainly containing analysis funcions
    """

    def __init__(self, nest_data: NESTData, model_name: str = "NNBase", model_kwargs: dict = model_kwargs):

        # Initialize a general model
        super().__init__(model_name=model_name,
                         dataset=nest_data.dataset,
                         data=nest_data.data,
                         differences=nest_data.differences,
                         model_kwargs=model_kwargs)

        print("\nConstructing a NEST model...")
        print("WARNING: Current implementation assume normalization, not robust to standardization")
        print("Indeed, the output is currently passed through a sigmoid which crushes it between 0 and 1")

        # Define parameters
        self.n_autoregression = model_kwargs["n_autoregression"]
        self.autoregressive = model_kwargs["autoregressive"]

        # Compute and save the usable indices of the data
        self.predictable_indices = self.get_predictable_indices(n_autoregression=model_kwargs['n_autoregression'],
                                                                verbose=model_kwargs['verbose'])

        # Training, validation and testing indices separation
        self.train_test_validation_separation(validation_percentage=model_kwargs["validation_percentage"],
                                              test_percentage=model_kwargs["test_percentage"])

    def train_test_validation_separation(self, validation_percentage: float = 0.2,
                                         test_percentage: float = 0.1, shuffle: bool = True) -> None:
        """
        Function to separate the data into training and testing parts. The trick here is that
        the data is not actually split - this function actually defines the indices of the
        data points that are in the training/testing part

        Args:
            validation_percentage:  Percentage of the data to keep out of the training process
                                    for validation
            test_percentage:        Percentage of data to keep out of the training process for
                                    the testing
            shuffle:                To shuffle the indices

        Returns:
            Nothing, in place definition of all the indices
        """

        # Sanity checks: the given inputs are given as percentage between 0 and 1
        if 1 <= validation_percentage <= 100:
            validation_percentage /= 100
            print("The train-test-validation separation rescaled the validation_percentage between 0 and 1")
        if 1 <= test_percentage <= 100:
            test_percentage /= 100
            print("The train-test-validation separation rescaled the test_percentage between 0 and 1")

        # Copy the usable indices in the data and shuffle them
        indices = self.predictable_indices.copy()

        # Shuffle if wanted
        if shuffle:
            np.random.shuffle(indices)

        # Get the number of indices
        n_indices = len(indices)

        # Define the different indices with the right proportions
        self.train_indices = indices[:int((1 - test_percentage - validation_percentage) * n_indices)]
        self.validation_indices = indices[int((1 - test_percentage - validation_percentage) * n_indices):
                                          int((1 - test_percentage) * n_indices)]
        self.test_indices = indices[int((1 - test_percentage) * n_indices):]

    def batch_iterator(self, iterator_type: str = "train", batch_size=None, shuffle: bool = True) -> None:
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

        # Firstly control that the training indices exist - create them otherwise
        if self.train_indices is None:
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
            indices = self.train_indices.copy()
        elif "alidation" in iterator_type:
            indices = self.validation_indices.copy()
        elif "est" in iterator_type:
            indices = self.test_indices.copy()
        else:
            raise ValueError(f"Unknown type of batch creation {iterator_type}")

        # Shuffle them if wanted
        if shuffle:
            random.shuffle(indices)

        # Define the right number of batches according to the wanted batch_size - taking care of the
        # special case where the indicies ae exactly divisible by the batch size, which can induce
        # an additional empty batch breaking the simulation down the line
        n_batches = int(np.ceil(len(indices) / batch_size))

        # Iterate to yield the right batches with the wanted size
        for batch in range(n_batches):
            yield indices[batch * batch_size:(batch + 1) * batch_size]

    def build_input_from_index(self, index: int, data=None):
        """
        Function to build an input from a given index in the data, which can then be forwarded
        to the model.
        This has to be overridden for each subclass as the input may have a different form.
        """
        raise NotImplementedError(f"Needs to be implemented in each subclass - or think of a general"
                                  f" approach to put in the parent general class.")

    def build_input_output_from_indices(self, indices: list):
        """
        Input and output generator from given indices corresponding to a batch

        Args:
            indices: indices of the batch to prepare

        Returns:
            The input and output of the batch in the wanted form
        """

        # Iterate over the indices to build the input in the right form
        input_tensor_list = [self.build_input_from_index(index) for index in indices]

        # Create the final input tensor
        input_tensor = torch.stack(input_tensor_list).view(len(input_tensor_list), input_tensor_list[0].shape[1])

        # Create the output DataFrame
        if self.predict_differences:
            output_data = self.differences.iloc[indices]
        else:
            output_data = self.data[self.components].iloc[indices]

        # Return both
        return input_tensor, output_data

    def evaluate_loss(self, predictions, targets):
        """
        Function to evaluate the losses of the various component, returning a new one, the sum
        so that PyTorch managed the right backward path.

        Args:
            predictions:    Predicted values
            targets:        Actual values

        Returns:
            A new loss function for PyTorch to manage the backward path
        """

        # Build a list of losses and iterate over the components
        losses = []
        for component in self.components:
            target = torch.FloatTensor([targets[component].values]).view(-1, 1)
            target.to(self.device)
            # Evaluate each loss
            losses.append(
                self.losses[component](predictions[component],
                                       target))

        # Return the sum of losses for PyTorch
        return sum(losses)

    def compute_loss(self, batch_indices):
        """
        Custom function to compute the loss of a batch for that special NN model

        Args:
            batch_indices: The indices of the batch

        Returns:
            The loss
        """

        # Build the input tensor and the output data in the wanted form
        input_tensor, output_data = self.build_input_output_from_indices(batch_indices)

        input_tensor = input_tensor.to(self.device)

        # Forward the batch
        predictions = self.model(input_tensor)

        # Return the loss
        return self.evaluate_loss(predictions, output_data)

    def predict_from_index(self, index: int, horizon: int = 96):
        """
        Function making a prediction over the 'horizon' using the model. By Default, this gives a
        day-ahead prediction, looping over the time intervals and iteratively take the last output
        as input for the next prediction.

        Args:
            index:  Index from which to start the prediction
            horizon: Length of the prediction (number of intervals)

        Returns:
            Both the predictions and the true observations over the horizon to facilitate an
            analysis down the line
        """

        # Get the true data: here we need to take enough terms in the past for the model's
        # autoregression to work and enough in the future for the prediction horizon
        true_data = self.data.iloc[index - self.n_autoregression:index + horizon]

        # Copy the true data and erase the values we want to predict over the horizon
        predictions = true_data.copy()
        predictions.loc[predictions.index[self.n_autoregression:], self.components] = np.nan

        # Use the model to make predictions, iterating over the horizon, putting each new
        # prediction in the data to use as input for the next one
        self.model.eval()
        for i in range(horizon):

            # Build the input tensor from the data and forward it through the model
            input_tensor = self.build_input_from_index(index=i + self.n_autoregression,
                                                       data=predictions)
            output = self.model(input_tensor)

            # If the model predicts differences, some manipulations are needed
            if self.predict_differences:

                columns = [x for x in self.components if x not in self.not_differences_components]

                # First step: put the predictions in a DataFrame
                out = pd.DataFrame(index=[0],
                                   columns=columns)
                for col in columns:
                    out[col] = output[col].item()

                # We can then use either the normalization or standardization parameters to normalize (i.e.
                # standardize) the difference that was predicted by the model
                if self.dataset.is_normalized:
                    out = 0.8 * out / (self.dataset.max_[self.components] - self.dataset.min_[self.components])

                elif self.dataset.is_standardized:
                    out = out.divide(self.dataset.std[self.components])

                else:
                    pass

                # Loop through the components of the model and augment the data with their prediction, adding
                # it to the previous one since the model predicts differences
                for component in self.components:
                    predictions.loc[predictions.index[self.n_autoregression + i], component] \
                        = predictions.loc[predictions.index[self.n_autoregression + i - 1], component] + \
                          out[component].values

                for component in self.not_differences_components:
                    predictions.loc[predictions.index[self.n_autoregression + i], component]\
                            = output[component].item()

            # Otherwise, if we predict direct values, we can simply recall them
            else:
                for component in self.components:
                    predictions.loc[predictions.index[self.n_autoregression + i], component]\
                            = output[component].item()

        # Scale both predictions and true data back to the original values - and only keep the data of
        # importance, i.e. the values of each component over the horizon
        if self.dataset.is_normalized:
            predictions = self.dataset.inverse_normalize(predictions.loc[predictions.index[self.n_autoregression:],
                                                                         self.components])
            true_data = self.dataset.inverse_normalize(true_data.loc[true_data.index[self.n_autoregression:],
                                                                     self.components])

        elif self.dataset.is_standardized:
            predictions = self.dataset.inverse_standardize(predictions.loc[predictions.index[self.n_autoregression:],
                                                                           self.components])
            true_data = self.dataset.inverse_standardize(true_data.loc[true_data.index[self.n_autoregression:],
                                                                       self.components])
        else:
            pass

        # Return both predictions and true values
        return predictions, true_data

    def plot_predictions(self, index: int, horizon: int = 96, how_many: int = None, **kwargs) -> None:
        """
        Small function to plot predictions from a certain index into the future over the 'horizon'
        Args:
            index:      Where to start the prediction
            horizon:    Horizon of the prediction
            how_many:   To tune  how many of the components to plot
            **kwargs:   Kwargs of the plots, see util.plots
        """

        # By default, plot each component
        if how_many is None:
            how_many = len(self.components)

        # Compute the predictions and get the true data
        predictions, true_data = self.predict_from_index(index=index,
                                                         horizon=horizon)

        # Loop over the number of components wanted and plot the prediction vs the true observations
        for component in self.components[:how_many]:

            # Design the plot with custom helpers
            _plot_helpers(title=component, **kwargs)

            # Plot both informations
            plt.plot(predictions[component], label="Prediction")
            plt.plot(true_data[component], label="Observations")

            # Save or show the plot
            _save_or_show(legend=True, **kwargs)

    def predictions_analysis(self, horizon: int = 96, **kwargs) -> None:
        """
        Function analyzing the prediction performances of the model in terms of MAE propagation
        over a given horizon. It takes 100 random indices from the data, and predicts over the
        'horizon' starting from there, and analyzing how the MAE evolves along the horizon

        Args:
            horizon:    Horizon to analyze
            **kwargs:   Kwargs of the plots, see util.plots
        """

        # Print the start
        print("\nAnalyzing 100 predictions...")

        # Get a copy of the validation indices, shuffle it and keep the 100 first
        indices = self.validation_indices.copy()
        random.shuffle(indices)
        indices = indices[:100]

        # Build a dictionary of errors for each component and iterate over the indices to predict
        errors = {component: [] for component in self.components}
        for num, index in enumerate(indices):

            # Use the model to predict over the horizon
            predictions, true_data = self.predict_from_index(index, horizon)

            # Store the MAE of each component
            for component in self.components:
                errors[component].append(compute_mae(predictions[component], true_data[component]).values)

            # Informative print
            if num % 10 == 9:
                print(f"{num + 1} predictions done")

        # Create dictionary to store the mean and max errors at each time step over the horizon and
        # fill it by iterating over the components
        mean_errors = {}
        max_errors = {}
        for component in self.components:

            # Nested use of list comprehension to compute the mean and the max errors
            mean_errors[component] =\
                [np.nanmean([x[i] for x in errors[component]]) for i in range(len(errors[component][0]))]
            max_errors[component] =\
                [np.nanmax([x[i] for x in errors[component]]) for i in range(len(errors[component][0]))]

        # Plot the mean errors using the usual helpers, looping over the components to have
        # all errors in one plot
        _plot_helpers(title="Mean MAE", **kwargs)
        for component in self.components:
            plt.plot(mean_errors[component], label=component)
        _save_or_show(legend=True, **kwargs)

        # Plot the max errors using the usual helpers, looping over the components to have
        # all errors in one plot
        _plot_helpers(title="Max MAE", **kwargs)
        for component in self.components:
            plt.plot(max_errors[component], label=component)
        _save_or_show(legend=True, **kwargs)


class NNModel(NNBase):
    """
    Class using a branching NN as model for the NEST, using the NESTNN module created with PyTorch
    """

    def __init__(self, nest_data: NESTData, model_name: str = "NNModel", model_kwargs=model_kwargs,
                 initialize: bool = True):

        # Initialize a NEST model
        super().__init__(nest_data=nest_data,
                         model_name=model_name,
                         model_kwargs=model_kwargs)

        # Stupid way of making inheritance work: when using a subclass, just set initialize to false
        # This avoids a NESTNNModel being fully built but ensures the base is constructed above
        if initialize:
            print("\nConstructing the NN model...")

            # Build the input indices to link to each component
            self._build_components_inputs_indices()

            # Build the model itself, using the custom branching NN
            self.model = NN(components_inputs_indices=self.components_inputs_indices,
                            hidden_sizes=model_kwargs["hidden_sizes"],
                            output_size=model_kwargs["output_size"])

            # Build the optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=model_kwargs["learning_rate"])

            # Try to load an existing model
            if model_kwargs["load_model"]:
                self.load_model()

    def _build_components_inputs_dict(self):
        """
        Rewrite the components inputs dictionary: if we use a simple NN model, we also want to include
        'self-regression', i.e. use previous temperature measurements to predict the next one
        """

        super._build_components_inputs_dict()

        # Add self autoregression sensors
        for component in self.components:
            self.components_inputs_dict[component].append(component)


    def _build_components_inputs_indices(self):
        """
        TODO: how robust is it?
        Function linking the input indices corresponding to the sensors defined in the
        'components_inputs_dict' for each component.
        In other words, this builds a dictionary containing for each sensor which indices
        of the input impact it (e.g. the indices where weather measurements are, as well as those
        of the previous temperature (among others) are defined for a room temperature sensor)
        """

        # Build the dictionary of sensors impacting each component
        self._build_components_inputs_dict()

        # Create a dictionary containing the input indices corresponding to each sensor as well as
        # an index to recall which index of the input we are currently analyzing
        sensors_input_indices = dict()
        current_index = 0

        # Iterate over the sensors and check i they are 'autoregressive' i.e. if several values
        # in the past are to be taken into account
        for sensor in self.data.columns:
            if sensor in self.autoregressive:
                # If it is an autoregressive sensor, the next several indices (defined by the
                # 'n_autoregression' parameter) correspond to the sensor
                sensors_input_indices[sensor] = list(
                    np.arange(current_index, current_index + self.n_autoregression))
                # Increase the counter
                current_index += self.n_autoregression
            else:
                # Only the current index needs to be taken
                sensors_input_indices[sensor] = [current_index]
                current_index += 1

        # Create the wanted dictionary linking inputs to each component and iterate over the latter
        self.components_inputs_indices = {}
        for component in self.components:

            # Build a list of indices corresponding to the sensors needed
            indices = []
            for key in self.components_inputs_dict[component]:
                indices.append(sensors_input_indices[key])

            # Save it in a singly flat list
            self.components_inputs_indices[component] = flat_list(indices)

    def build_input_from_index(self, index: int, data=None):
        """
        Function to put the input into the wanted form, from an index, i.e. the wanted prediction.
        The index gives the target value of the DataFrame, and several past values are put into
        form to build the input according to their autoregressive value.

        If no data is provided, this considers the entire dataset and assumes the index is based on it.
        Otherwise, it part of the data is passed in argument, this assumes the index corresponds to
        it (for predictions), i.e. the index 7 is the 8th position in the passed data, not the original
        data.

        Args:
            index:  The index of the data to predict
            data:   Data to build the input from - it none take the full data

        Returns:
            The input in the right form
        """

        # Take the entire data if None given
        if data is None:
            data = self.data

        # Build the input and iterate over all sensors to put it in form
        input_tensor = []
        for sensor in data.columns:

            # If it is an 'autoregressive' sensor, put the past data in
            if sensor in self.autoregressive:
                input_tensor.append(list(data[sensor].iloc[index - self.n_autoregression:index].values))

            # Otherwise, only put the previous data
            else:
                input_tensor.append([data[sensor].iloc[index - 1]])

        # Return it in the needed form
        return torch.FloatTensor(flat_list(input_tensor)).view(1, -1)


class NNExtractionModel(NNModel):
    """
    Class using a branching NN as model for the NEST with some feature extraction from past data,
    using the NESTNNModelFeatureExtraction module created with PyTorch.

    We need to redefine the 'components_input_indices' because they don't have the same form
    anymore, and we also need to define the 'autoregressive_sensors_input_indices' that are
    needed for the feature extraction model
    """

    def __init__(self, nest_data: NESTData, model_kwargs=model_kwargs):

        # Initialize a NEST model base (through the parent NESTNNModel)
        super().__init__(nest_data=nest_data,
                         model_name=model_kwargs["model_name"],
                         model_kwargs=model_kwargs,
                         initialize=False)

        print("\nConstructing the Feature extraction model...")

        # Build the input indices to link to each component
        self._build_inputs_indices()

        # Build the model itself, using the custom branching NN with feature extraction
        self.model = NNExtraction(components_inputs_indices=self.components_inputs_indices,
                                  autoregressive_input_indices=self.autoregressive_input_indices,
                                  feature_extraction_sizes=model_kwargs["feature_extraction_sizes"],
                                  hidden_sizes=model_kwargs["hidden_sizes"],
                                  output_size=model_kwargs["output_size"])

        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_kwargs["learning_rate"])

        # Try to load an existing model
        if model_kwargs["load_model"]:
            self.load_model()

    def _build_inputs_indices(self):
        """
        TODO: how robust is it?
        Function linking the input indices corresponding to the sensors defined in the
        'components_inputs_dict' for each component.
        In other words, this builds a dictionary containing for each sensor which indices
        of the input impact it (e.g. the indices where weather measurements are, as well as those
        of the previous temperature (among others) are defined for a room temperature sensor)

        Additionally, the 'autoregressive_sensors_input_indices' are also constructed, a dictionary
        linking each sensor to its corresponding input indices.
        this is useful to know what part of the input to use to extract features for each
        autoregessive sensor.
        """

        # Build the dictionary of sensors impacting each component
        self._build_components_inputs_dict()

        # Create a dictionary containing the input indices corresponding to each sensor as well as
        # an index to recall which index of the input we are currently analyzing
        self.autoregressive_input_indices = dict()
        current_index = 0

        # Create dictionary of wich sensor corresponds to which input
        sensors_input_indices = dict()
        sensor_number = 0

        # Iterate over the sensors and check i they are 'autoregressive' i.e. if several values
        # in the past are to be taken into account
        for sensor in self.data.columns:
            if sensor in self.autoregressive:

                # Autoregressive sensors: recall corresponding inputs
                self.autoregressive_input_indices[sensor] =\
                    list(np.arange(current_index, current_index + self.n_autoregression))
                # Increase the counter
                current_index += self.n_autoregression
            else:
                # Only the current index needs to be taken
                self.autoregressive_input_indices[sensor] = []
                current_index += 1

            # Recall which sensor corresponds to which index
            sensors_input_indices[sensor] = sensor_number
            sensor_number += 1

        # Create the wanted dictionary linking inputs to each component and iterate over the latter
        self.components_inputs_indices = {}
        for component in self.components:

            # Build a list of indices corresponding to the sensors needed
            indices = []
            for key in self.components_inputs_dict[component]:
                indices.append(sensors_input_indices[key])

            # Save it in a singly flat list
            self.components_inputs_indices[component] = indices


class LSTMExtractionModel(NNBase):
    """
    Class using LSTMs to model each component, by branching the input.
    In its simplest form one each LSTM simply predicts the next temperature in the component.
    If 'NN' is true, then the output of the LSTM (i.e. the hidden_size) can be larger than one,
    and it is then fed to a Neural Network which then outputs the prediction
    """

    def __init__(self, nest_data: NESTData, model_kwargs: dict = model_kwargs, NN: bool = True):
        """
        Create the model using the wanted arguments. If 'NN' is True, one LSTM is built for each
        component, and their outputs are fed in a NN which then make the prediction.
        If 'NN' is False, the LSTMs directly predict each component.

        Args:
            nest_data:     The data to train/test the model
            model_kwargs:  All kinds of arguments (see parameters.py)
            NN:            Flag to set False if you want direct predictions from the LSTM, without
                           the feedforward NN at the end
        """

        # Initialize a NEST model (through the parent NESTNN model)
        super().__init__(nest_data=nest_data,
                         model_name=model_kwargs["model_name"],
                         model_kwargs=model_kwargs)

        print("\nConstructing the LSTM model...")

        # Build the input indices to link to each component, i.e. define for each component which
        # indices to take in the input
        self._build_components_inputs_indices()

        # Build the model itself, using the right custom LSTM module, depending on the 'NN' flag
        if NN:
            self.model = LSTMNNExtraction(components_inputs_indices=self.components_inputs_indices,
                                          hidden_size=model_kwargs["hidden_size"],
                                          num_layers=model_kwargs["num_layers"],
                                          hidden_sizes=model_kwargs["hidden_sizes"],
                                          output_size=model_kwargs["output_size"],
                                          predict_differences=model_kwargs["predict_differences"])
        else:
            self.model = LSTMExtraction(components_inputs_indices=self.components_inputs_indices,
                                        hidden_size=model_kwargs["hidden_size"],
                                        num_layers=model_kwargs["num_layers"])

        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_kwargs["learning_rate"])

        # Try to load an existing model
        if model_kwargs["load_model"]:
            self.load_model()

    def _build_components_inputs_indices(self):
        """
        TODO: how robust is it?
        Function linking the input indices corresponding to the sensors defined in the
        'components_inputs_dict' for each component.
        In other words, this builds a dictionary containing for each sensor which indices
        of the input impact it (e.g. the indices where weather measurements are, as well as those
        of the previous temperature (among others) are defined for a room temperature sensor)
        """

        # Build the dictionary of sensors impacting each component
        self._build_components_inputs_dict()

        # Create an empty list of indices for each component
        self.components_inputs_indices = {component: [] for component in self.components}

        # Loop over the snesors and components to fill the lists of indices for each component
        for index, sensor in enumerate(self.data.columns):
            for component in self.components:
                if sensor in self.components_inputs_dict[component]:
                    self.components_inputs_indices[component].append(index)

    def build_input_from_index(self, index: int, data=None):
        """
        Input and output generator from given indices corresponding to a batch. Here we overwrite the
        general function as LSTMs require sequential inputs, which are in 3 dimensions, not 2 like
        normal feedforward networks.

        Args:
            index: index to build the input from
            data: data to use to build the input from
        Returns:
            The input in the wanted form
        """

        if data is None:
            data = self.data

        # Transform the data from the DataFrame in a tensor to be handled later
        input_tensor = torch.FloatTensor(data.iloc[index - self.n_autoregression: index, :].values).\
            view(1, self.n_autoregression, -1)

        # Return the input
        return input_tensor

    def build_input_output_from_indices(self, indices: list):
        """
        Input and output generator from given indices corresponding to a batch. Here we overwrite the
        general function as LSTMs require sequential inputs, which are in 3 dimensions, not 2 like
        normal feedforward networks.

        Args:
            indices: indices of the batch to prepare

        Returns:
            The input and output of the batch in the wanted form
        """

        # Iterate over the indices to build the input in the right form
        input_tensor_list = [torch.FloatTensor(self.data.iloc[index - self.n_autoregression:index, :].values)
                             for index in indices]

        # Create the final input tensor, in a way robust to batch of size 1
        # General case: stack the list of tensors together
        if len(input_tensor_list) > 1:
            input_tensor = torch.stack(input_tensor_list)
        # Batch_size=1
        else:
            input_tensor = input_tensor_list[0].view(1, self.n_autoregression, -1)

        # Create the output DataFrame
        if self.predict_differences:
            output_data = self.differences.iloc[indices]
        else:
            output_data = self.data[self.components].iloc[indices]

        # Return both
        return input_tensor, output_data
