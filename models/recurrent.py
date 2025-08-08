"""
File containing different recurrent models
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from models.data_handling import NESTData
from models.base_models import BaseModel
from models.recurrent_modules import RoomLSTM

from util.util import compute_mae, get_room
from util.plots import _save_or_show, _plot_helpers, plot_time_series

from parameters import  ROOMS
from models.parameters import model_kwargs


class ModelList(object):
    """
    Class to model a UNIT/building/apartment: defined as a list of rooms.
    """

    def __init__(self, nest_data: NESTData, model_kwargs: dict = model_kwargs):
        """
        Args:
            nest_data:      An object of the NESTData class containing various information about the data
            model_kwargs:   Variety of parameters of the models (see 'parameters.py')
        """

        print("\nConstructing the LSTM models...")
        print("WARNING: Current implementation assume normalization, not robust to standardization")
        print("Indeed, the output is currently passed through a sigmoid which crushes it between 0 and 1")

        self.columns = nest_data.data.columns
        self.unit = model_kwargs["unit"]
        self.n_autoregression = model_kwargs["n_autoregression"]
        self.threshold_length = model_kwargs["threshold_length"]
        self.heating = model_kwargs["heating"]
        self.cooling = model_kwargs["cooling"]
        self.rooms = model_kwargs["rooms"]
        self.max_ = nest_data.dataset.max_
        self.min_ = nest_data.dataset.min_
        self.interval = nest_data.dataset.interval
        self.data = nest_data.data
        self.autoregressive = [x for x in self.data.columns if ("Time" not in x) & ("Case" not in x)]

        self.components = [x for x in model_kwargs["components"][:-1] if x[-3:] in self.rooms] + \
                          [x for x in self.columns if "Energy" in x]

        # Build the input indices to link to each component, i.e. define for each component which
        # indices to take in the input
        self._build_components_inputs_indices()

        # Define the zero energy points for each room and the scaling factor
        self.factors, self.zero_energies = self._build_zeroes_factors()

        # Define a dictionary, with a model for each room needed
        self.room_models = {room: LSTMModel(nest_data=nest_data,
                                            model_kwargs=model_kwargs,
                                            room=room,
                                            room_inputs=self.rooms_inputs[room],
                                            component_inputs_indices=self.components_inputs_indices[room],
                                            effects_inputs_indices=self.effects_inputs_indices[room],
                                            factor=self.factors[room],
                                            zero_energy=self.zero_energies[room])
                            for room in self.rooms}

        # Separate the data
        self.train_test_validation_separations()

        # Compute sequences of usable data (i.e. without missing values) that are valid for ALL the rooms
        print("Computing common sequences")
        self.train_heating_sequences, self.train_cooling_sequences,\
            self.test_heating_sequences, self.test_cooling_sequences = self._build_common_sequences()
        self.train_sequences = self.train_heating_sequences + self.train_cooling_sequences
        self.test_sequences = self.test_heating_sequences + self.test_cooling_sequences

    def _build_components_inputs_dict(self):
        """
        Function to define which inputs impact which component. For example, the weather impacts
        the room temperature, the occupancy of a specific room impacts it, and so on.
        A dictionary is built where a list of sensor influencing each component is created.
        """

        # Define the dictionary and iterate over the components
        self.components_inputs_dict = {}

        for component in self.components:

            # Get the inputs impacting the energy consumption of each room: the inlet water temperature,
            # the room temperature, valve openings and the case (heating or cooling)
            if "Energy" in component:

                # In DFAB we have outlet temperature information, the rest are in both datasets
                sensors = [x for x in self.columns if ("inlet" in x) | ("outlet" in x)]
                sensors.append("Case")
                room = get_room(component)
                sensors.append(f"Thermal valve {room}")
                sensors.append(f"Thermal temperature measurement {room}")

            # Inputs impacting the room temperature
            else:
                # All rooms are influenced by the weather, the time, the electricity consumption
                # and the total thermal energy consumed
                sensors = [x for x in self.columns if ("Time" in x) | ("Weather" in x) | ("outlet" in x)]
                sensors += ["Electricity total consumption", "Case"]

                # Add all sensor related to the room under consideration (temperature, brightness, ...)
                room = get_room(component)
                sensors += [x for x in self.columns if room in x]

                if self.unit == "UMAR":
                    sensors.append("Thermal inlet temperature")

            # Save the sensors in the dictionary
            self.components_inputs_dict[component] = sensors

    def _build_components_inputs_indices(self):
        """
        TODO: how robust is it?
        Function linking the input indices corresponding to the sensors defined in the
        'components_inputs_dict' for each component.

        In other words, this builds a dictionary containing for each room which indices
        of the input impact it. For example, the outside temperature will impact the room temperature
        of room 272. This function will put the index of the outside temperature among the data
        (say it's in the 3rd place) in the list of indices influencing that room.

        Then, given the total input (with all information at our disposal), if we want to predict the
        temperature of a particular room, this dictionary allows us to select the right inputs.

        Additionally, the 'effects_inputs_indices' is also defined. This does the same thing but for
        the added effect of heating/cooling (see definition of the 'RoomLSTM' module).

        Finally, the 'rooms_inputs' are also built, containing the indices of the data that influence
        each room
        """

        # Build the dictionary of sensors impacting each component
        self._build_components_inputs_dict()

        # Create an empty list of indices for each component
        self.rooms_inputs = {room: [] for room in self.rooms}
        self.components_inputs_indices = {room: [] for room in self.rooms}
        self.effects_inputs_indices = {room: [] for room in self.rooms}

        # Loop over the sensors and components to fill the lists of indices for each component
        for index, sensor in enumerate(self.columns):
            for component in self.components:
                if sensor in self.components_inputs_dict[component]:
                    if "Energy" not in component:
                        self.rooms_inputs[get_room(component)].append(index)
                        if ("valve" not in sensor) & ("inlet" not in sensor) & ("Case" not in sensor) & (
                                "Energy" not in sensor):
                            self.components_inputs_indices[get_room(component)].append(
                                len(self.rooms_inputs[get_room(component)]) - 1)
                        if ("Case" in sensor) | ("valve" in sensor) | ("inlet" in sensor) | ("measurement" in sensor):
                            self.effects_inputs_indices[get_room(component)].append(
                                len(self.rooms_inputs[get_room(component)]) - 1)

    def _build_zeroes_factors(self):
        """
        Small helper function to compute the scaled zero energy and the scaling factor of each room. This
        is used later in training and predictions

        Returns:
            factors:    Scaling factors for each room temperature
            zeroes:     Zero energy in scaled terms
        """

        # The factors are simply corresponding to a max-min normalization. This is used to put a value
        # from the original scale to the normalized one (between 0.1 and 0.9)
        factors = {get_room(component): 0.8 / (self.max_[component] - self.min_[component])
                   for component in self.components if "temp" in component}

        # Scaled zero energy: For each room, since the data is normalized, this corresponds to zero energy
        # in the scaled case between 0.1 and 0.9
        zeroes = {
            get_room(component): (-self.min_[component]) / (self.max_[component] - self.min_[component]) * 0.8 + .1
            for component in self.components if "Energy" in component}

        return factors, zeroes

    def _build_common_sequences(self):
        """
        Function to build common sequences of data, i.e. sequences where there is no missing value for
        all the rooms --> these sequences can then be used to compare all the rooms at the same time

        Returns:
            heating_sequences:      Training heating events
            cooling_sequences:      Training cooling events
            test_heating_sequences: Testing heating events
            test_cooling_sequences: Testing cooling events
        """

        # Prepare the list to store sequences
        heating_sequences = []
        cooling_sequences = []
        test_heating_sequences = []
        test_cooling_sequences = []

        # We need to separate between heating and cooling to ensure that one sequence of data only contains
        # heating or cooling events
        if self.heating:

            # Get the heating indices of the first room, then intersect it with all the other rooms
            heating_indices = self.room_models[self.rooms[0]].heating_indices
            if len(self.rooms) > 1:
                for room in self.rooms[1:]:
                    heating_indices = [index for index in heating_indices if
                                       index in self.room_models[room].heating_indices]

            # Get the jumps, i.e. places where missing values intervene - the difference between jumps
            # corresponds to the length of data without missing values
            heating_jumps = list(np.where(np.diff(heating_indices) > 1)[0] + 1)

            # Store long enough sequences (here defined as 6h + whatever autoregression is needed)
            for beginning, end in zip([0] + heating_jumps, heating_jumps + [len(heating_indices) - 1]):
                if end-beginning > int(24 * 60 / self.interval / 4) + self.n_autoregression:
                    heating_sequences.append(list(np.arange(heating_indices[beginning] - self.n_autoregression,
                                                            heating_indices[beginning])) + heating_indices[beginning: end])

            # Put some of the sequences in testing, until we have at least 14 days in testing
            enough_test = False
            while not enough_test:
                test_heating_sequences.append(heating_sequences.pop(np.argmax([len(seq) for seq in heating_sequences])))
                if np.sum([len(seq) for seq in test_heating_sequences]) > int(14 * 24 * 60 / self.interval):
                    enough_test = True

        if self.cooling:

            # Get the cooling indices of the first room, then intersect it with all the other rooms
            cooling_indices = self.room_models[self.rooms[0]].cooling_indices
            if len(self.rooms) > 1:
                for room in self.rooms[1:]:
                    cooling_indices = [index for index in cooling_indices if
                                       index in self.room_models[room].cooling_indices]

            # Get the jumps, i.e. places where missing values intervene - the difference between jumps
            # corresponds to the length of data without missing values
            cooling_jumps = list(np.where(np.diff(cooling_indices) > 1)[0] + 1)

            # Store long enough sequences (here defined as 6h + whatever autoregression is needed)
            for beginning, end in zip([0] + cooling_jumps, cooling_jumps + [len(cooling_indices) - 1]):
                if end - beginning > int(24 * 60 / self.interval / 4) + self.n_autoregression:
                    cooling_sequences.append(list(np.arange(cooling_indices[beginning] - self.n_autoregression,
                                                            cooling_indices[beginning])) + cooling_indices[beginning: end])

            # Put some of the sequences in testing, until we have at least 14 days in testing
            enough_test = False
            while not enough_test:
                test_cooling_sequences.append(cooling_sequences.pop(np.argmax([len(seq) for seq in cooling_sequences])))
                if np.sum([len(seq) for seq in test_cooling_sequences]) > int(14 * 24 * 60 / self.interval):
                    enough_test = True

        # Return everything
        return heating_sequences, cooling_sequences, test_heating_sequences, test_cooling_sequences

    def train_test_validation_separations(self):
        """
        Separate the data in train, validation and test for each model
        """

        for model in self.room_models.values():
            model.train_test_validation_separation(threshold_length=self.threshold_length)

    def fit(self, epochs: int = 20):
        """
        Fit the model: fits all the rooms one after the other
        """
        for model in self.room_models.values():
            print(f"\n Fitting room {model.room}\n")
            model.fit(epochs)

    def plot_predictions(self, sequence: list) -> None:
        """
        Small function to plot predictions from a certain index into the future

        Args:
            sequence:   Sequence to predict
        """

        # Define the plot, one line for each room and one column for the temperature prediction, one for
        # the energy
        fig, axes = plt.subplots(nrows=len(self.rooms), ncols=2, sharex=True, figsize=(20, 6 * len(self.rooms)))
        fig.autofmt_xdate()
        j = 0

        # Loop over the room models to compute the predictions and plot everything
        for model in self.room_models.values():

            # Compute the predictions and get the true data
            predictions, true_data = model.predict_sequence(sequence=sequence)

            # Plot the true and predicted temperature
            axes[j, 0].plot(true_data[f"Thermal temperature measurement {model.room}"], label="Observations")
            axes[j, 0].plot(predictions[f"Thermal temperature measurement {model.room}"], label="Prediction")

            # Plot the true and predicted energy
            axes[j, 1].plot(true_data[f"Energy room {model.room}"], label="Observations")
            axes[j, 1].plot(predictions[f"Energy room {model.room}"], label="Prediction")

            # Make the plot look nice
            axes[j, 0].set_title(f"Temperature - Room {model.room}", size=25)
            axes[j, 1].set_title(f"Energy - Room {model.room}", size=25)
            if j == len(self.rooms) - 1:
                axes[j, 0].set_xlabel("Time", size=20)
                axes[j, 1].set_xlabel("Time", size=20)
            axes[j, 0].set_ylabel("Degrees", size=20)
            axes[j, 0].set_ylabel("Energy", size=20)

            axes[j, 0].tick_params(axis='x', which='major', labelsize=15)
            axes[j, 1].tick_params(axis='x', which='major', labelsize=15)
            axes[j, 0].tick_params(axis='y', which='major', labelsize=15)
            axes[j, 1].tick_params(axis='y', which='major', labelsize=15)
            axes[j, 0].legend(prop={'size': 15})
            axes[j, 1].legend(prop={'size': 15})

            j += 1

        # Define the right layout, save the figure and plot it
        plt.tight_layout()
        plt.savefig("Predictions.png", bbox_inches="tight")
        plt.show()

    def predictions_analysis(self, showfliers: bool = True, whiskers: list = [5, 95], savename: str = "Model_errors"):
        """
        Analyze the predictions of the entire model, i.e. for each room. This will compare predictions
        to the true data for 500 sequences (of one day long predictons) and then plot the absolute error
        over the horizon, binned by hours for clarity.

        Args:
            showfliers: Flage to decide whether to show the outliers in the plot
            whiskers:   Parameter of the whiskers (percentiles)
            savename:   Under which name to save it
        """

        # Define the plot, one lin for each room, one column for the temperature and one for the energy
        fig, axes = plt.subplots(nrows=len(self.rooms), ncols=2, sharex=True, sharey=True,
                                 figsize=(20, 6 * len(self.rooms)))
        j = 0
        for room in self.room_models.keys():

            # Get the errors of the room model
            errors = self.room_models[room].predictions_analysis(return_=True)

            for i, part in enumerate(self.room_models[room].parts):

                # Plot the information
                axes[j, i % 2].boxplot(errors[part], notch=True, whis=whiskers, showfliers=showfliers)

                # Make the plot look nice
                axes[j, i % 2].set_title(f"Absolute error - Room {room}" if i % 2 == 0
                                         else "Absolute error - Energy", size=25)
                if j == 4:
                    axes[j, i % 2].set_xlabel("Prediction hour ahead", size=20)
                axes[j, i % 2].set_ylabel("Degrees" if i == 0 else "Energy", size=20)
                axes[j, i % 2].tick_params(axis='x', which='major', labelsize=15)
                axes[j, i % 2].tick_params(axis='y', which='major', labelsize=15)

            j += 1

        # Define the right layout, save the figure and plot it
        plt.tight_layout()
        plt.savefig(os.path.join("saves", "Figures", savename + ".png"), bbox_inches="tight")
        plt.show()


class LSTMModel(BaseModel):
    """
    Model of one room, using the 'RoomLSTM' module as its core.
    """

    def __init__(self, nest_data: NESTData, model_kwargs: dict, room: str, room_inputs: dict,
                 component_inputs_indices: dict, effects_inputs_indices: dict, factor: float, zero_energy: float):
        """
        Create the model using the wanted arguments
        Args:
            nest_data:                  The data to train/test the model
            model_kwargs:               All kinds of arguments (see parameters.py)
            room:                       Name of the room
            room_inputs:                Input indices corresponding to that room (which part of the whole
                                         data actually influences that room)
            component_inputs_indices:   Among those room inputs, inputs used in the RoomLSTM module
            effects_inputs_indices:     Indices of the heating/cooling effect inputs
            factor:                     Scaling temperature factor (defined in the 'ModelList')
            zero_energy:                Scaled zero energy (defined in the 'ModelList')
        """

        # Little trick to initialize each room in the same ModelList with a different name
        model_kwargs["model_name"] += room

        # Initialize a general BaseModel
        super().__init__(dataset=nest_data.dataset,
                         data=nest_data.data.iloc[:, room_inputs].copy(),
                         differences=nest_data.differences,
                         model_kwargs=model_kwargs)

        # Recover the true model name
        model_kwargs["model_name"] = model_kwargs["model_name"][:-3]

        # Save all the needed parameters
        self.room = room
        self.room_inputs = room_inputs
        self.component_inputs_indices = component_inputs_indices
        self.effects_inputs_indices = effects_inputs_indices

        self.threshold_length = model_kwargs["threshold_length"]
        self.n_autoregression = model_kwargs["n_autoregression"]
        self.autoregressive = self.data.columns  # Again needed
        self.hidden_size = model_kwargs["hidden_size"]
        self.NN = model_kwargs["NN"]
        self.heating = model_kwargs["heating"]
        self.cooling = model_kwargs["cooling"]

        self.validation_percentage = model_kwargs["validation_percentage"]
        self.test_percentage = model_kwargs["test_percentage"]
        self.threshold_length = model_kwargs["threshold_length"]
        self.predictable_indices = []
        self.heating_indices = []
        self.cooling_indices = []
        self.zero_energy = zero_energy

        self.heating_sequences = []
        self.cooling_sequences = []
        self.sequences = []
        self.train_sequences = []
        self.validation_sequences = []
        self.test_sequences = []


        self.parts = ["Temperature", "Energy"]
        self.pred_column = np.where(self.data.columns == f"Thermal temperature measurement {self.room}")[0].item()
        self.ctrl_column = np.where(self.data.columns == f"Thermal valve {self.room}")[0].item()

        # Build the model itself, using the custom LSTM module - if 'NN' is True, then a neural network
        # is created after the LSTM to build the output
        self.model = RoomLSTM(device=self.device,
                              component_inputs_indices=self.component_inputs_indices,
                              effects_inputs_indices=self.effects_inputs_indices,
                              factor=factor,
                              interval=model_kwargs["interval"],
                              hidden_size=model_kwargs["hidden_size"],
                              num_layers=model_kwargs["num_layers"],
                              NN=model_kwargs["NN"],
                              hidden_sizes=model_kwargs["hidden_sizes"],
                              output_size=model_kwargs["output_size"])

        # Build the losses (there are 2, one for the temperature prediction, one for the energy)
        self.losses = {part: F.mse_loss for part in self.parts}

        # Put the model on the right device: this has to be done here because otherwise the definition
        # of the optimizer and loading of the model can fail
        self.model = self.model.to(self.device)

        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_kwargs["learning_rate"])

        # Try to load an existing model
        if model_kwargs["load_model"]:
            self.load_model()

        # Once the model is load, put it to the right device (again) --> ensures the loaded
        # version is actually at the right place
        self.model = self.model.to(self.device)

    def get_sequences(self, case: str = None, threshold_length: int = 96) -> None:
        """
        Function to prepare sequence of data, i.e. consecutive timestamps where there are no missing
        values. The threshold length avoids the sequences being too long (Truncated Back Propagation
        Through Time) to improve training.

        Args:
            case:               Usually "Heating" or "Cooling", used to differentiate both
            threshold_length:   Max length of a sequence

        Returns:
            Sequences that can be used
        """

        # Use the global function to get all indices with no missing values, we will do that for both the
        # heating and cooling cases
        predictable_indices = self.get_predictable_indices(n_autoregression=self.n_autoregression, case=case)
        self.predictable_indices += predictable_indices
        if case == "Heating":
            self.heating_indices = predictable_indices
        elif case == "Cooling":
            self.cooling_indices = predictable_indices

        # Get the jumps, i.e. places where missing values intervene - the difference between jumps
        # corresponds to the length of data without missing values
        jumps = list(np.where(np.diff(predictable_indices) > 1)[0] + 1)
        # Prepare the list to store sequences
        sequences = []

        # Loop over the sequences
        for beginning, end in zip([0] + jumps, jumps + [len(predictable_indices) - 1]):

            # Check the sequence length: if it is higher than the threshold, break it down
            if end - beginning > threshold_length - 1:
                temp_beg = beginning
                temp_end = beginning + threshold_length

                # Iterate with a sliding window and recall all sequences
                while temp_end <= end:
                    sequences.append(list(np.arange(predictable_indices[temp_beg] - self.n_autoregression,
                                                    predictable_indices[temp_beg])) +
                                     predictable_indices[temp_beg: temp_end])
                    temp_beg += 1
                    temp_end += 1

            # Else the sequence is stored as is
            else:
                sequences.append(list(
                    np.arange(predictable_indices[beginning] - self.n_autoregression, predictable_indices[beginning])) +
                                 predictable_indices[beginning: end])

        return sequences

    def prepare_sequences(self, threshold_length: int = 96):
        """
        Function to prepare the sequences. This takes care of the differentiation between the heating
        and cooling cases, and uses the 'get_sequences' function to compute the sequences.

        The trick to remember is that the sequences are indices in the full DataFrame --> the later
        has to remain in the same shape for the indices to be the same.

        Args:
            threshold_length:   Max length of a sequence

        Returns:
            Nothing, the sequences that can be used are computed in place
        """

        # Heating sequences
        # Trick to ensure the indices stay the same: put all the cooling cases to NaN. That way, the
        # DataFrame keeps its shape and only heating sequences will be recognized
        index = self.data.index[np.where(self.data.iloc[:, -1] == 0.1)[0]]
        temp = self.data.loc[index, :].copy()
        self.data.loc[index, :] = np.nan

        # Compute the sequences
        if self.heating:
            self.heating_sequences = self.get_sequences(case="Heating", threshold_length=threshold_length)
        else:
            self.heating_sequences = []

        # recover the original data
        self.data.loc[index, :] = temp.copy()

        # Cooling case: as before
        index = self.data.index[np.where(self.data.iloc[:, -1] == 0.9)[0]]
        temp = self.data.loc[index, :].copy()
        self.data.loc[index, :] = np.nan
        if self.cooling:
            self.cooling_sequences = self.get_sequences(case="Cooling", threshold_length=threshold_length)
        else:
            self.cooling_sequences = []
        self.data.loc[index, :] = temp.copy()

        # Memory considerations
        del temp

        # Save the sequences
        self.sequences = self.heating_sequences + self.cooling_sequences

    def train_test_validation_separation(self, validation_percentage: float = 0.2,
                                         test_percentage: float = 0.1, shuffle: bool = True,
                                         threshold_length: int = 96) -> None:
        """
        Function to separate the data into training and testing parts. The trick here is that
        the data is not actually split - this function actually defines the sequences of
        data points that are in the training/testing part.

        Args:
            validation_percentage:  Percentage of the data to keep out of the training process
                                    for validation
            test_percentage:        Percentage of data to keep out of the training process for
                                    the testing
            shuffle:                To shuffle the indices
            threshold_length:       Maximal length of a sequence to train the LSTM

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

        # Create the sequences and copy them
        self.prepare_sequences(threshold_length=threshold_length)
        sequences = self.sequences.copy()

        # Shuffle if wanted
        if shuffle:
            np.random.shuffle(sequences)

        # Compute the cumulative length of the sequences
        len_sequences = np.cumsum([len(x) for x in sequences])

        # Given the total length of sequences, define aproximate separations between training
        # validation and testing sets
        train_validation_sep = int((1 - test_percentage - validation_percentage) * len_sequences[-1])
        validation_test_sep = int((1 - test_percentage) * len_sequences[-1])

        # Prepare the lists
        self.train_sequences = []
        self.validation_sequences = []
        self.test_sequences = []

        # Loop over sequences and save them in the right list
        for i, total in enumerate(len_sequences):

            # The second condition ensures to have at list on sequence in validation and testing
            if (total < train_validation_sep) & (i < len(len_sequences) - 2):
                self.train_sequences.append(sequences[i])

            # Same: conditions ensures at list one sequence in each set
            elif ((train_validation_sep <= total < validation_test_sep) & (i < len(len_sequences) - 1)) | (
                    i == len(len_sequences) - 2):
                self.validation_sequences.append(sequences[i])

            else:
                self.test_sequences.append(sequences[i])

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
            sequences = self.train_sequences.copy()
        elif "alidation" in iterator_type:
            sequences = self.validation_sequences.copy()
        elif "est" in iterator_type:
            sequences = self.test_sequences.copy()
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
            yield sequences[batch * batch_size:(batch + 1) * batch_size]

    def build_tensor_from_sequence(self, sequence: list, data=None):
        """
       Tensor generator from a given sequence of indices in the data.

        Args:
            sequence:   Sequence to build the input from
            data:       Data to use to build the input from

        Returns:
            The tensor in the wanted form
        """

        # Get the model data if None is given
        if data is None:
            data = self.data

        # Transform the data from the DataFrame in a tensor to be handled later
        tensor = torch.FloatTensor(data.iloc[sequence, :].values.copy())

        # Return the tensor
        return tensor.to(self.device)

    def build_input_output_from_sequences(self, sequences: list):
        """
        Input and output generator from given sequences of indices corresponding to a batch.

        Args:
            sequences: sequences of the batch to prepare

        Returns:
            input_tensor:   Batch input of the model
            output_dict:    Targets of the model, a dict with a batch of temperature targets and one of energy
        """

        # Ensure the given sequences are a list of list, not only one list
        if type(sequences[0]) == int:
            sequences = [sequences]

        # Iterate over the indices to build the input in the right form (the last index is not
        # take here as it will be predicted only)
        input_tensor_list = [self.build_tensor_from_sequence(sequence[:-1]) for sequence in sequences]

        # Create the final input tensor, in a way robust to batch of size 1
        # General case: stack the list of tensors together, using pad_sequence to handle variable
        # length sequences
        if len(sequences) > 1:
            input_tensor = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
        # For batch_size 1
        else:
            size = input_tensor_list[0].shape[0]
            input_tensor = input_tensor_list[0].view(1, size, -1)

        # Prepare the output for the temperature and energy consumption

        if self.predict_differences:
            #output_dict = [torch.FloatTensor(self.differences[self.component][sequence[self.n_autoregression:]].values).to(self.device) for sequence in sequences]
            raise NotImplementedError("TO DO")

        else:
            output_dict = {part: [] for part in self.parts}
            for part in self.parts:
                # Take the right data for each part
                if part == "Temperature":
                    output_dict[part] = [torch.FloatTensor(self.data[f"Thermal temperature measurement {self.room}"][
                                                               sequence[self.n_autoregression:]].values).to(self.device)
                                         for sequence in sequences]
                else:
                    output_dict[part] = [torch.FloatTensor(
                        self.data[f"Energy room {self.room}"][sequence[self.n_autoregression:]].values).to(self.device)
                                         for sequence in sequences]

        # Again, build the final results by taking care of the batch_size=1 case
        # Pad the sequences to handle variable length
        for part in self.parts:
            if len(sequences) > 1:
                output_dict[part] = pad_sequence(output_dict[part], batch_first=True, padding_value=0).unsqueeze(-1)

            # Batch size of 1
            else:
                size = output_dict[part][0].shape[0]
                output_dict[part] = output_dict[part][0].view(1, size, -1)

        # Return everything
        return input_tensor, output_dict

    def evaluate_loss(self, predictions, targets):
        """
        Function to evaluate the losses of the two parts (temperature and Energy), returning the sum
        so that PyTorch managed the right backward path.

        Args:
            predictions:    Predicted values
            targets:        Actual values

        Returns:
            A new loss function for PyTorch to manage the backward path
        """

        # Evaluate the loss in temperature and energy predictions
        losses = []
        for part in self.parts:
            losses.append(self.losses[part](predictions[part], targets[part]))

        # Return the sum of losses for PyTorch
        return sum(losses)

    def compute_loss(self, batch_sequences: list):
        """
        Custom function to compute the loss of a batch for that special NN model. The predictions
        indeed need to be made in a sequential manner, as the temperature prediction of one time
        step is needed as input for the next one.

        Args:
            batch_sequences: The sequences in the batch

        Returns:
            The loss
        """

        # Build the input tensor and the output data in the wanted form
        input_tensor, output_dict = self.build_input_output_from_sequences(batch_sequences)

        # Dict to store the results
        predictions = {part: torch.zeros_like(output_dict[part]) for part in self.parts}

        # Warm up step: Take the past data (up to n_autoregression) and let the model run. This allows
        # the model to build up the initial hidden and cell states of the LSTMs
        (base, heating, cooling), (h, c) = self.model(input_tensor[:, :self.n_autoregression, :])

        # To store the output
        output = {}

        # The energy output is set to zero
        output["Energy"] = torch.ones(input_tensor.shape[0], 1).to(self.device)
        output["Energy"] *= self.zero_energy

        # The first temperature output (i.e. base prediction of the module)
        output["Temperature"] = base

        # Add the heating and/or cooling effects
        if heating is not None:
            past_heating = heating["Temperature"]
            output["Temperature"] += past_heating
            output["Energy"] += heating["Energy"]
        if cooling is not None:
            past_cooling = cooling["Temperature"]
            output["Temperature"] -= past_cooling
            output["Energy"] -= cooling["Energy"]

        # Recall everything
        for part in self.parts:
            predictions[part][:, 0, :] = output[part]

        # Now iterate along the given sequence of data, each time feeding it to the network
        # and storing the outputs
        for i in range(input_tensor.shape[1] - self.n_autoregression):

            # Build the tnesor and feed it in the network
            input_tensor[:, self.n_autoregression + i, self.pred_column] = base.squeeze()
            (base, heating, cooling), (h, c) = self.model(
                input_tensor[:, self.n_autoregression + i, :].view(input_tensor.shape[0], 1, -1), h, c)

            # Store everything as before
            output = {}
            output["Energy"] = torch.ones(input_tensor.shape[0], 1).to(self.device)
            output["Energy"] *= self.zero_energy
            output["Temperature"] = base

            if heating is not None:
                past_heating += heating["Temperature"]
                output["Temperature"] += past_heating
                output["Energy"] += heating["Energy"]
            if cooling is not None:
                past_cooling += cooling["Temperature"]
                output["Temperature"] -= past_cooling
                output["Energy"] -= cooling["Energy"]

            for part in self.parts:
                predictions[part][:, i + 1, :] = output[part]

        # Memory considerations
        del input_tensor

        # Return the loss of the preictions
        return self.evaluate_loss(predictions, output_dict)

    def predict_sequence(self, sequence: list):
        """
        Function making a prediction over a sequence.

        Args:
            sequence:  Sequence of data to predict

        Returns:
            Both the predictions and the true observations over the horizon to facilitate an
            analysis down the line
        """

        # Get the true data that the model will want to predict
        true_data = self.data.iloc[sequence, :][
            [f"Thermal temperature measurement {self.room}", f"Energy room {self.room}"]]

        # Build the input tensor and the predictions dictionary
        input_tensor, _ = self.build_input_output_from_sequences(sequences=[sequence])
        predictions = {part: torch.zeros((len(sequence) - self.n_autoregression, 1)) for part in self.parts}

        # Warm up step: feed the first input to build the hidden and cell states
        (base, heating, cooling), (h, c) = self.model(input_tensor[:, :self.n_autoregression, :])

        # Set the energy to zero and the temperature to the base prediction of the network
        output = {}
        output["Energy"] = torch.ones(input_tensor.shape[0], 1).to(self.device)
        output["Energy"] *= self.zero_energy
        output["Temperature"] = base

        # Add the effect of heating/cooling
        if heating is not None:
            past_heating = heating["Temperature"]
            output["Temperature"] += past_heating
            output["Energy"] += heating["Energy"]
        if cooling is not None:
            past_cooling = cooling["Temperature"]
            output["Temperature"] -= past_cooling
            output["Energy"] -= cooling["Energy"]

        # Recall everything
        for part in self.parts:
            predictions[part][0, :] = output[part]

        # Iterate tis over the sequence of data to predict
        for i in range(input_tensor.shape[1] - self.n_autoregression):
            input_tensor[:, self.n_autoregression + i, self.pred_column] = base.squeeze()
            (base, heating, cooling), (h, c) = self.model(
                input_tensor[:, self.n_autoregression + i, :].view(input_tensor.shape[0], 1, -1), h, c)

            output = {}
            output["Energy"] = torch.ones(input_tensor.shape[0], 1).to(self.device)
            output["Energy"] *= self.zero_energy
            output["Temperature"] = base

            if heating is not None:
                past_heating += heating["Temperature"]
                output["Temperature"] += past_heating
                output["Energy"] += heating["Energy"]
            if cooling is not None:
                output["Temperature"] -= past_cooling
                output["Energy"] -= cooling["Energy"]
                past_cooling += cooling["Temperature"]

            for part in self.parts:
                predictions[part][i + 1, :] = output[part]

        # If the model predicts differences, some manipulations are needed
        if self.predict_differences:
            raise NotImplementedError

        # Otherwise, if we predict direct values, we can simply recall them
        else:
            for part in self.parts:
                predictions[part] = predictions[part].cpu().squeeze().detach().numpy()

        # Turn the predictions (dictionary) into a DataFrame
        prediction = pd.DataFrame(predictions)
        prediction.rename(columns={"Temperature": f"Thermal temperature measurement {self.room}",
                                   "Energy": f"Energy room {self.room}"}, inplace=True)

        # Scale both predictions and true data back to the original values - and only keep the data of
        # importance, i.e. the values of each component over the horizon
        if self.dataset.is_normalized:
            prediction = self.dataset.inverse_normalize(prediction)
            true_data = self.dataset.inverse_normalize(true_data)

        elif self.dataset.is_standardized:
            prediction = self.dataset.inverse_standardize(predictions)
            true_data = self.dataset.inverse_standardize(true_data)
        else:
            pass

        # Put the predictions and true_data at the same index
        prediction.index = true_data.index[self.n_autoregression:]

        # Memory considerations
        del input_tensor

        # Return both predictions and true values
        return prediction, true_data

    def aggregate_energy(self, predictions, true_data=None):
        """
        Function to aggregate individual predictions and possibly true data to get the total energy.

        TODO: not very useful as is

        Args:
            predictions:    Predictions from the Pytorch module
            true_data:      True data corresponding to the predictions

        Returns:
            The aggregated predicted and aggregated true energy consumption for comparison down the line
        """
        # Prepare the tensors
        tot_pred = np.zeros_like(predictions[self.components[0]])
        if true_data is not None:
            tot_true = np.zeros_like(predictions[self.components[0]])

        # Loop over the energy component and aggregate them
        for component in self.components:
            if "Energy" in component:
                tot_pred += predictions[component]
                if true_data is not None:
                    tot_true += true_data[component]

        if true_data is not None:
            return tot_pred, tot_true
        else:
            return tot_pred

    def plot_predictions(self, sequence: list, **kwargs) -> None:
        """
        Small function to plot predictions and true data from a certain index into the future

        Args:
            sequence:   Sequence to predice
            **kwargs:   Kwargs of the plots, see util.plots
        """

        # Compute the predictions and get the true data
        predictions, true_data = self.predict_sequence(sequence=sequence)

        # Loop over the number of components wanted and plot the prediction vs the true observations
        for component in predictions.columns:
            # Design the plot with custom helpers
            _plot_helpers(title=component, **kwargs)

            # Plot both informations
            plt.plot(true_data[component], label="Observations")
            plt.plot(predictions[component], label="Prediction")

            # Save or show the plot
            _save_or_show(legend=True, **kwargs)

    def predictions_analysis(self, return_: bool = False, showfliers: bool = False) -> None:
        """
        Function analyzing the prediction performances of the model in terms of MAE propagation.
        It takes 500 random sequences from the data, and predicts over the analyzes how the MAE
        evolves along the horizon.
        The errors are binned by hour to make the plot clearer.

        Args:
            return_:    Flag to set to True if you want to return the analysis instead of plotting it directly
                         (used by the ModelList to gather predictions for all the rooms and make a unique
                         plot for all of them)

        Returns:
            Erors by hour of predictions or plots (boxplot) them
        """

        # Print the start
        print("\nAnalyzing up to 500 predictions...")
        print("Warning: Consistent for 15 minutes interval, what about the rest?")

        # Get a copy of the validation indices and shuffle it
        sequences = self.validation_sequences.copy()
        np.random.shuffle(sequences)

        # Take long enough sequences to have interesting predictions
        sequences = [x for x in sequences if len(x) == self.threshold_length + self.n_autoregression]
        sequences = sequences[:500]

        # Build a dictionary of errors for each component and iterate over the indices to predict
        errors = {part: [] for part in self.parts}
        for num, sequence in enumerate(sequences):

            # Use the model to predict over the sequences, and rename the DataFrame column for the plots
            prediction, true_data = self.predict_sequence(sequence)
            prediction.rename(columns={f"Thermal temperature measurement {self.room}": "Temperature",
                                       f"Energy room {self.room}": "Energy"}, inplace=True)
            true_data.rename(columns={f"Thermal temperature measurement {self.room}": "Temperature",
                                      f"Energy room {self.room}": "Energy"}, inplace=True)
            true_data = true_data.iloc[self.n_autoregression:, :]

            # Compute the errors
            for part in self.parts:
                errors[part].append(compute_mae(prediction[part], true_data[part]).values)

            # Informative print
            if num % 25 == 24:
                print(f"{num + 1} predictions done")

        # Create dictionary to store the the errors of each prediction, binned by hours (assuming
        # 15 minutes interval
        errors_timesteps = {}
        for part in self.parts:
            errors_timesteps[part] = [[x[i] for x in errors[part]] + [x[i + 1] for x in errors[part]] +
                                      [x[i + 2] for x in errors[part]] + [x[i + 3] for x in errors[part]]
                                      for i in range(0, self.threshold_length, 4)]

        if return_:
            return errors_timesteps

        else:
            # Create the subplot and loop over the parts (Temperature and Energy)
            fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20, 7))

            for i, part in enumerate(self.parts):
                # Don't show outliers for the energy, as they can be big (when the energy peaks the model can
                # be quite wrong) and will deform the plot
                axes[i].boxplot(errors_timesteps[part], notch=True, whis=[5, 95],
                                showfliers=showfliers)

                # Make the plot look nice
                axes[i].set_title(f"Absolute error - Room {self.room}" if i == 0
                                  else "Absolute error - Energy", size=25)
                axes[i].set_xlabel("Prediction hour ahead", size=20)
                axes[i].tick_params(axis='x', which='major', labelsize=15)
                axes[i].tick_params(axis='y', which='major', labelsize=15)
            axes[0].set_ylabel("Degrees", size=20)
            axes[1].set_ylabel("Energy", size=20)

            # Define the right layout, save the figure and plot it
            plt.tight_layout()
            plt.savefig(f"Errors{self.room}.png", bbox_inches="tight")
            plt.show()

    def plot_consistency(self, sequence):
        """
        Function to analyze the physical consistency of the model: plots:
         - the predicted temperatures and energy given the observed patterns of valves opening and closing
         - the predicted values in the case where the valves are kept open along the prediction horizon
         - the predicted values in the case where the valves are kept closed along the prediction horizon

         Args:
             sequence:  the sequence of inputs to analyze
        """

        print("Warning: Working with normalization, not standardization")

        room = self.room

        # Make predictions
        y, _ = self.predict_sequence(sequence)
        y1 = y[f"Thermal temperature measurement {room}"]
        y1.name = "Prediction"
        y1_energy = y[f"Energy room {room}"]
        y1_energy.name = "Prediction"

        # Recall te true opening and closing pattern
        temp = self.data.loc[self.data.index[sequence], f"Thermal valve {room}"].values.copy()

        # Put the valves to the closed state
        self.data.loc[self.data.index[sequence], f"Thermal valve {room}"] = 0.1
        # Compute the predictions in that case
        y, _ = self.predict_sequence(sequence)
        y2 = y[f"Thermal temperature measurement {room}"]
        y2.name = "All closed"
        y2_energy = y[f"Energy room {room}"]
        y2_energy.name = "All closed"

        # Put the valves to the opened state
        self.data.loc[self.data.index[sequence], f"Thermal valve {room}"] = 0.9
        # Compute the predictions in that case
        y, _ = self.predict_sequence(sequence)
        y3 = y[f"Thermal temperature measurement {room}"]
        y3.name = "All open"
        y3_energy = y[f"Energy room {room}"]
        y3_energy.name = "All open"

        # Bring back the original data so that further operations ar not impacted
        self.data.loc[self.data.index[sequence], f"Thermal valve {room}"] = temp

        # Plot the results
        plot_time_series([y2, y3, y1], ylabel="Degrees", title=f"Temperature room {room}")
        plot_time_series([y2_energy, y3_energy, y1_energy], ylabel="Energy", title=f"Energy consumption room {room}")

    def consistency_analysis(self, sequences=None, boxplots: bool = False, **kwargs):
        """
        Function to analyze the physical consistency of the model, i.e. make sure that the temperature of the room
        when the valves are open is higher (heating case) or lower (cooling case) than when the valves are closed.
        Additionally, we would like our predictions, which are always based on a sequence of opening and closing
        of the valves, to always lie in between those two extrema.
        This analysis is thus performed for each room (and for the energy), i.e. all the components, each of
        them separated into its cooling and heating behavior, with 3 comparisons in each case.

        Deprecated: the new models shouldn't contain those inconsistencies anymore.

        The plots show aggregated values for each hour (i.e. binning 4 time steps together) to be more
        readable

        Args:
            sequences: the sequences to analyze (if None, then the usual validation sequences are chosen)
            kwargs:    plotting kwargs
        """

        # Define the sequences if none is given
        if sequences is None:
            print("Analyzing up to 500 predictions")
            sequences = self.validation_sequences.copy()
            np.random.shuffle(sequences)

            # Take long enough sequences to have interesting predictions
            sequences = [x for x in sequences if len(x) == self.threshold_length + self.n_autoregression]
            sequences = sequences[:500]

        # Define a dictionary for each type of error, i.e. heating and cooling errors, comparing
        # predictions, valves open and valves closed 2 by 2
        # We will record the 6 arising errors for each component
        cooling_errors_open = []
        heating_errors_open = []
        cooling_errors_closed = []
        heating_errors_closed = []
        cooling_errors_extrema = []
        heating_errors_extrema = []

        # Loop over the sequences to gather the errors
        for num, sequence in enumerate(sequences):

            # First, use the model to predict the influence of the actual pattern of valves
            prediction, _ = self.predict_sequence(sequence)

            # Recall te true opening and closing pattern for later
            valves = f"Thermal valve {self.room}"
            temp = self.data.loc[self.data.index[sequence], valves].values

            # Put the valves to the closed state
            self.data.loc[self.data.index[sequence], valves] = 0.1
            # Compute the predictions in that case
            all_closed, _ = self.predict_sequence(sequence)

            # Put the valves to the opened state
            self.data.loc[self.data.index[sequence], valves] = 0.9
            # Compute the predictions in that case
            all_open, _ = self.predict_sequence(sequence)

            # Bring back the original data so that further operations ar not impacted
            self.data.loc[self.data.index[sequence], valves] = temp

            # Case separation
            if self.data.iloc[sequence[0], -1] == 0.1:
                cooling_errors_open.append((all_open[f"Thermal temperature measurement {self.room}"] - prediction[
                    f"Thermal temperature measurement {self.room}"]).values)
                cooling_errors_closed.append((prediction[f"Thermal temperature measurement {self.room}"] - all_closed[
                    f"Thermal temperature measurement {self.room}"]).values)
                cooling_errors_extrema.append((all_open[f"Thermal temperature measurement {self.room}"] - all_closed[
                    f"Thermal temperature measurement {self.room}"]).values)

            else:
                heating_errors_open.append((prediction[f"Thermal temperature measurement {self.room}"] - all_open[
                    f"Thermal temperature measurement {self.room}"]).values)
                heating_errors_closed.append((all_closed[f"Thermal temperature measurement {self.room}"] - prediction[
                    f"Thermal temperature measurement {self.room}"]).values)
                heating_errors_extrema.append((all_closed[f"Thermal temperature measurement {self.room}"] - all_open[
                    f"Thermal temperature measurement {self.room}"]).values)

            # Informative print
            if num % 5 == 4:
                print(f"{num + 1} predictions done")

        # Nested use of list comprehension to compute errors for each time step
        # here we additionally cluster 4 time steps together at each time to make the plot more visible
        # I.e. we gather all the errors of each prediction hour together in on bin
        cooling_errors_open_timesteps = [[x[i] for x in cooling_errors_open] +
                                         [x[i + 1] for x in cooling_errors_open] +
                                         [x[i + 2] for x in cooling_errors_open] +
                                         [x[i + 3] for x in cooling_errors_open]
                                         for i in
                                         range(0, len(cooling_errors_open[0]), 4)]
        heating_errors_open_timesteps = [[x[i] for x in heating_errors_open] +
                                         [x[i + 1] for x in heating_errors_open] +
                                         [x[i + 2] for x in heating_errors_open] +
                                         [x[i + 3] for x in heating_errors_open]
                                         for i in
                                         range(0, len(heating_errors_open[0]), 4)]
        cooling_errors_closed_timesteps = [[x[i] for x in cooling_errors_closed] +
                                           [x[i + 1] for x in cooling_errors_closed] +
                                           [x[i + 2] for x in cooling_errors_closed] +
                                           [x[i + 3] for x in cooling_errors_closed]
                                           for i in
                                           range(0, len(cooling_errors_closed[0]), 4)]
        heating_errors_closed_timesteps = [[x[i] for x in heating_errors_closed] +
                                           [x[i + 1] for x in heating_errors_closed] +
                                           [x[i + 2] for x in heating_errors_closed] +
                                           [x[i + 3] for x in heating_errors_closed]
                                           for i in
                                           range(0, len(heating_errors_closed[0]), 4)]
        cooling_errors_extrema_timesteps = [[x[i] for x in cooling_errors_extrema] +
                                            [x[i + 1] for x in cooling_errors_extrema] +
                                            [x[i + 2] for x in cooling_errors_extrema] +
                                            [x[i + 3] for x in cooling_errors_extrema]
                                            for i in
                                            range(0, len(cooling_errors_extrema[0]), 4)]
        heating_errors_extrema_timesteps = [[x[i] for x in heating_errors_extrema] +
                                            [x[i + 1] for x in heating_errors_extrema] +
                                            [x[i + 2] for x in heating_errors_extrema] +
                                            [x[i + 3] for x in heating_errors_extrema]
                                            for i in
                                            range(0, len(heating_errors_extrema[0]), 4)]

        # Medians
        median_cooling_errors_open_timesteps = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_open], 50)
            for i in range(len(cooling_errors_open[0]))]
        median_heating_errors_open_timesteps = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_open], 50)
            for i in range(len(heating_errors_open[0]))]
        median_cooling_errors_closed_timesteps = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_closed], 50)
            for i in range(len(cooling_errors_closed[0]))]
        median_heating_errors_closed_timesteps = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_closed], 50)
            for i in range(len(heating_errors_closed[0]))]
        median_cooling_errors_extrema_timesteps = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_extrema], 50)
            for i in range(len(cooling_errors_extrema[0]))]
        median_heating_errors_extrema_timesteps = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_extrema], 50)
            for i in range(len(heating_errors_extrema[0]))]

        # 75% percentiles
        cooling_errors_open_timesteps_75 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_open], 75)
            for i in range(len(cooling_errors_open[0]))]
        heating_errors_open_timesteps_75 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_open], 75)
            for i in range(len(heating_errors_open[0]))]
        cooling_errors_closed_timesteps_75 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_closed], 75)
            for i in range(len(cooling_errors_closed[0]))]
        heating_errors_closed_timesteps_75 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_closed], 75)
            for i in range(len(heating_errors_closed[0]))]
        cooling_errors_extrema_timesteps_75 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_extrema], 75)
            for i in range(len(cooling_errors_extrema[0]))]
        heating_errors_extrema_timesteps_75 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_extrema], 75)
            for i in range(len(heating_errors_extrema[0]))]

        # 95% percentiles
        cooling_errors_open_timesteps_95 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_open], 95)
            for i in range(len(cooling_errors_open[0]))]
        heating_errors_open_timesteps_95 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_open], 95)
            for i in range(len(heating_errors_open[0]))]
        cooling_errors_closed_timesteps_95 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_closed], 95)
            for i in range(len(cooling_errors_closed[0]))]
        heating_errors_closed_timesteps_95 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_closed], 95)
            for i in range(len(heating_errors_closed[0]))]
        cooling_errors_extrema_timesteps_95 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_extrema], 95)
            for i in range(len(cooling_errors_extrema[0]))]
        heating_errors_extrema_timesteps_95 = [
            np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_extrema], 95)
            for i in range(len(heating_errors_extrema[0]))]

        # List all the errors
        errors_ = [median_cooling_errors_open_timesteps, cooling_errors_open_timesteps_75,
                   cooling_errors_open_timesteps_95,
                   median_cooling_errors_closed_timesteps, cooling_errors_closed_timesteps_75,
                   cooling_errors_closed_timesteps_95,
                   median_cooling_errors_extrema_timesteps, cooling_errors_extrema_timesteps_75,
                   cooling_errors_extrema_timesteps_95,
                   median_heating_errors_open_timesteps, heating_errors_open_timesteps_75,
                   heating_errors_open_timesteps_95,
                   median_heating_errors_closed_timesteps, heating_errors_closed_timesteps_75,
                   heating_errors_closed_timesteps_95,
                   median_heating_errors_extrema_timesteps, heating_errors_extrema_timesteps_75,
                   heating_errors_extrema_timesteps_95]

        # list the labels to give to the curves
        labels = ["Open - Pred", "Pred - Closed", "Open - Closed",
                  "Pred - Open", "Closed - Pred", "Closed - Open"]

        # Define colors and line styles
        colors = ["blue", "green", "red"]  # , "black", "orange", "violet"]
        styles = ["solid", "dashed", "dotted"]

        # Define the list of errors for each component
        errors_list = [cooling_errors_open_timesteps, cooling_errors_closed_timesteps,
                       cooling_errors_extrema_timesteps,
                       heating_errors_open_timesteps, heating_errors_closed_timesteps,
                       heating_errors_extrema_timesteps, ]

        # If all the boxplots are wanted
        if boxplots:

            print("Warning: Deprecated, unsure that works")

            # Corresponding prints
            prints = ["Open vs Prediction Error", "Prediction vs Closed Error", "Open vs Closed Error",
                      "Prediction vs Open Error", "Closed vs Prediction Error", "Closed vs Open Error"]

            # Define the label of the y-axis
            ylabel = "Degrees" if "Energy" not in component else "Energy"

            # Print the start of the analysis
            print(f"\n==========================================================\n{component}")
            print("==========================================================")

            # Loop over the 6 errors
            for num, errors in enumerate(errors_list):

                # Print the case we are looking at
                if num == 0:
                    print("\n----------------\nHeating Case")
                    print("----------------\n")
                elif num == 3:
                    print("\n----------------\nCooling Case")
                    print("----------------\n")

                print(prints[num])
                # _plot_helpers(xlabel="Prediction time step", ylabel=ylabel, **kwargs)
                # plt.boxplot(errors, notch=True, whis=[5, 95], showfliers=True)
                # _save_or_show(**kwargs)

                # Loop over the errors only keep the positive ones (negative errors mean that the model
                # was physically sound) and count along the way the good predictions
                count_good_predictions = 0
                total_predictions = 0

                # Define the list of bins of errors for the boxplot and loop along the errors to prepare them
                error_bins = []
                for i in range(len(errors)):

                    # Count the number of time predictions we have
                    number = len(errors[i])
                    # Compute the true errors (i.e. the positive ones)
                    true_errors = [x for x in errors[i] if x > 0]

                    # Update the counters
                    count_good_predictions += (number - len(true_errors))
                    total_predictions += number

                    # Append the true errors for the boxplot
                    if len(true_errors) > 0:
                        error_bins.append(true_errors)
                    else:
                        error_bins.append([])

                # Print the results
                print(f"{count_good_predictions} out of {total_predictions} comparisons were physically sound,"
                      f"{count_good_predictions / total_predictions * 100:.2f}%")

                print(f"The error magnitude of the physically inconsistent steps is plotted below:")

                # Boxplot of the true errors
                _plot_helpers(xlabel="Prediction hour", ylabel=ylabel, **kwargs)
                plt.boxplot(error_bins, notch=True, whis=[5, 95], showfliers=True)
                _save_or_show(**kwargs)

        # Plot all informations in one plot, separating the heating and cooling cases

        # Prepare the plot
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(16, 7))

        # Loop over the components to analyze the errors
        for num, error in enumerate(errors_):
            if num % 3 == 0:
                errors = errors_list[int(num / 3)]
                count_wrong_predictions = 0
                total_predictions = 0
                # Compute the error percentage
                for j in range(len(errors)):
                    # Count the number of time predictions we have
                    number = len(errors[j])
                    # Update the counters
                    count_wrong_predictions += len([x for x in errors[j] if x > 0])
                    total_predictions += number

            # Define the column, label, color and style of the plot
            column = 0 if num < 9 else 1
            label = f"{labels[int(num / 3)]} ({count_wrong_predictions / total_predictions * 100:.0f}%)" if num % 3 == 0 else None
            color = colors[int(num / 3)] if num < 9 else colors[int(num / 3) - 3]
            style = styles[num % 3]

            # Plot the mdeian, 75%, 95% percentile
            toplot = [0 if np.isnan(x) else x for x in error]
            axes[column].plot(toplot, label=label, color=color, linestyle=style)

        # Define labels and legends
        axes[0].set_ylabel(f"Room {self.room} ($^\circ$C)", size=22)
        axes[0].tick_params(axis='y', which='major', labelsize=15)
        axes[0].legend(prop={'size': 15})
        axes[1].legend(prop={'size': 15})
        # Set title and x label
        axes[0].set_xlabel("Prediction time step", size=20)
        axes[1].set_xlabel("Prediction time step", size=20)
        axes[0].set_title("Cooling error", size=25)
        axes[1].set_title("Heating error", size=25)
        axes[0].tick_params(axis='x', which='major', labelsize=15)
        axes[1].tick_params(axis='x', which='major', labelsize=15)

        # Put it in the riht layout, save the figure and plot it
        plt.tight_layout()
        plt.savefig("Consistency.png", bbox_inches="tight")
        plt.show()

        # Plot separating all the 6 different cases
        # New labels
        labels = ["Cooling\nOpen - Pred", "Cooling\nPred - Closed", "Cooling\nOpen - Closed",
                  "Heating\nPred - Open", "Heating\nClosed - Pred", "Heating\nClosed - Open"]

        # Prepare the figure and loop over the components to analyze them
        fig, axes = plt.subplots(nrows=1, ncols=6, sharex=True, sharey=True, figsize=(36, 7))
        # Recall the total wrong predictions, predictions and maximum error for each room
        wrong_predictions = 0
        predictions = 0
        maxima = []
        for num, error in enumerate(errors_):
            if num % 3 == 0:
                errors = errors_list[int(num / 3)]
                count_wrong_predictions = 0
                total_predictions = 0
                # Compute the error percentages
                for j in range(len(errors)):
                    # Count the number of time predictions we have
                    number = len(errors[j])
                    # Update the counters
                    count_wrong_predictions += len([x for x in errors[j] if x > 0])
                    total_predictions += number

                wrong_predictions += count_wrong_predictions
                predictions += total_predictions

            if num % 3 == 2:
                maxima.append(np.nanmax(error))

            # Define the column label, color and line style
            column = int(num / 3)
            label = f"{count_wrong_predictions / total_predictions * 100:.0f}%" if num % 3 == 0 else None
            color = colors[int(num / 3)] if num < 9 else colors[int(num / 3) - 3]
            style = styles[num % 3]

            # Plot the info and legend
            toplot = [0 if np.isnan(x) else x for x in error]
            axes[column].plot(toplot, label=label, color=color, linestyle=style)
            axes[column].legend(prop={'size': 20})

        # Set the labels
        axes[0].set_ylabel(f"Room {self.room} ($^\circ$C)", size=22)
        axes[0].tick_params(axis='y', which='major', labelsize=15)

        # Print the statistics
        print(f"\n________________________________\n{self.room}:")
        print(f"  The model is physically inconsistent {wrong_predictions / predictions * 100:.0f}% of the time.")
        print(
            f"  When wrong, the model makes an error smaller than {np.nanmax(maxima):.2f} C with 95% probability.")

        # Set the labels and titles
        for j in range(6):
            axes[j].set_title(labels[j], size=25)
            axes[j].set_xlabel("Prediction time step", size=20)
            axes[j].tick_params(axis='x', which='major', labelsize=15)

        # Define the right layout, save the figure and plot it
        plt.tight_layout()
        plt.savefig("ConsistencyBis.png", bbox_inches="tight")
        plt.show()

##########################################
### Deprecated old version to keep
##########################################


from models.base_models import BaseModel_old
from models.recurrent_modules import LSTM_old

class LSTMModel_old_bis(BaseModel_old):
    """
    Class using LSTMs to model each component, by branching the input. If the argument NN is true, then
    the output of the LSTM is fed to a neural network, which yields the final output.
    """

    def __init__(self, nest_data: NESTData, model_kwargs: dict = model_kwargs):
        """
        Create the model using the wanted arguments
        Args:
            nest_data:     The data to train/test the model
            model_kwargs:  All kinds of arguments (see parameters.py)
        """

        # Initialize a general model
        super().__init__(dataset=nest_data.dataset,
                         data=nest_data.data,
                         differences=nest_data.differences,
                         model_kwargs=model_kwargs)

        print("\nConstructing the LSTM model...")
        print("WARNING: Current implementation assume normalization, not robust to standardization")
        print("Indeed, the output is currently passed through a sigmoid which crushes it between 0 and 1")

        self.threshold_length = model_kwargs["threshold_length"]
        self.n_autoregression = 1  # needed to save and load the right indices/models
        self.autoregressive = self.data.columns  # Again needed
        self.hidden_size = model_kwargs["hidden_size"]
        self.NN = model_kwargs["NN"]

        self.components = self.components[:-1] + [x for x in self.data.columns if "Energy" in x]
        # Define the losses
        self._build_losses()

        # Build the input indices to link to each component, i.e. define for each component which
        # indices to take in the input
        self._build_components_inputs_indices()

        self.predictable_indices = []
        # Training, validation and testing indices separation
        self.train_test_validation_separation(validation_percentage=model_kwargs["validation_percentage"],
                                              test_percentage=model_kwargs["test_percentage"],
                                              threshold_length=model_kwargs["threshold_length"])

        factors = {}
        for component in self.components:
            factors[component] = 0.8 / (self.dataset.max_[component] - self.dataset.min_[component])

        # Computation of the scaled value for zero energy
        self.zero_energies = self.get_zero_energies()

        # Build the model itself, using the custom LSTM module - if 'NN' is True, then a neraul network
        # is created after the LSTM to build the output
        self.model = LSTM_old(device=self.device,
                          components_inputs_indices=self.components_inputs_indices,
                          effects_inputs_indices=self.effects_inputs_indices,
                          factors=factors,
                          zero_energies=self.zero_energies,
                          hidden_size=model_kwargs["hidden_size"],
                          num_layers=model_kwargs["num_layers"],
                          NN=model_kwargs["NN"],
                          hidden_sizes=model_kwargs["hidden_sizes"],
                          output_size=model_kwargs["output_size"])
        print(self.model)

        self.model = self.model.to(self.device)
        # Define the losses (one for each component) and the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_kwargs["learning_rate"])

        # Try to load an existing model
        if model_kwargs["load_model"]:
            self.load_model()
        self.model = self.model.to(self.device)

    def _build_components_inputs_dict(self):
        """
        Function to define which inputs impact which component. For example, the weather impacts
        the room temperature, the occupancy of a specific room impacts it, and so on.
        A dictionary is built where a list of sensor influencing each component is created.
        """

        # Define the dictionary and iterate over the components
        self.components_inputs_dict = {}

        for component in self.components:

            # this takes both DFAB and UMAR cases, either the total energy, heating energy or cooling energy
            if "Energy" in component:
                # In DFAB we have outlet temperature information, the rest are in both datasets
                sensors = [x for x in self.data.columns if ("inlet" in x) | ("outlet" in x)]
                sensors.append("Case")

                room = get_room(component)
                sensors.append(f"Thermal valve {room}")
                sensors.append(f"Thermal temperature measurement {room}")

            else:
                # All components are influenced by the weather, the time, the electricity consumption
                # and the total thermal energy consumed
                sensors = [x for x in self.data.columns if ("Time" in x) | ("Weather" in x) | ("outlet" in x)]
                sensors += ["Electricity total consumption", "Case"]

                # Special case of the domestic hot water heat pump
                if "HP" in component:

                    if self.unit == "DFAB":
                        # Add the occupany of the unit
                        sensors.append("Occupancy 371")

                    # Add all sensors related to the pump
                    sensors += [x for x in self.data.columns if ("HP" in x) & (x != component)]

                # Otherwise: currently all room temperatures
                else:
                    # Add all sensor related to the room
                    room = get_room(component)
                    sensors += [x for x in self.data.columns if room in x]

                if self.unit == "UMAR":
                    sensors.append("Thermal inlet temperature")

            # Save the sensors in the dictionary
            self.components_inputs_dict[component] = sensors

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
        self.components_inputs_indices = {component: [] for component in self.components if "Energy" not in component}
        self.effects_inputs_indices = {component: [] for component in self.components}

        # Loop over the sensors and components to fill the lists of indices for each component
        for index, sensor in enumerate(self.data.columns):
            for component in self.components:
                if sensor in self.components_inputs_dict[component]:
                    if "Energy" not in component:
                        if ("valve" not in sensor) & ("inlet" not in sensor) & ("Case" not in sensor) & (
                                "Energy" not in sensor):
                            self.components_inputs_indices[component].append(index)
                    if ("Case" in sensor) | ("valve" in sensor) | ("inlet" in sensor) | ("measurement" in sensor):
                        self.effects_inputs_indices[component].append(index)

    def get_zero_energies(self):

        zeroes = {}
        for component in self.components:
            if "Energy" in component:
                zeroes[component] = (-self.dataset.min_[component]) / \
                                    (self.dataset.max_[component] - self.dataset.min_[component]) * 0.8 + .1

        return zeroes

    def get_sequences(self, threshold_length: int = 96, data=None) -> None:
        """
        Function to prepare sequence of data, i.e. consecutive timestamps where there are no missing
        values. The threshold length avoids the sequences being too long (Truncated Back Propagation
        Through Time) to improve training.
        """

        # Use the global function to get all indices with no missing values
        if len(data) > 0:
            predictable_indices = self.get_predictable_indices(n_autoregression=1, data=data)
            self.predictable_indices += predictable_indices
        else:
            return []

        # Get the jumps, i.e. places where missing values intervene - the difference between jumps
        # corresponds to the length of data without missing values
        jumps = list(np.where(np.diff(predictable_indices) > 1)[0] + 1)

        # Prepare the list to store sequences
        sequences = []

        # Loop over the sequences
        for beginning, end in zip([0] + jumps, jumps + [len(predictable_indices) - 1]):

            # Check the sequence length: if it is higher than the threshold, break it down
            if end - beginning > threshold_length - 1:
                temp_beg = beginning
                temp_end = beginning + threshold_length

                # Iterate with a sliding window and recall all sequences
                while temp_end <= end:
                    sequences.append([predictable_indices[temp_beg] - 1] +
                                     predictable_indices[temp_beg: temp_end])
                    temp_beg += 1
                    temp_end += 1

            # Else the sequence is stored as is
            else:
                sequences.append([predictable_indices[beginning] - 1] +
                                 predictable_indices[beginning: end])

        # Save the computations as an attribute
        return sequences

    def prepare_sequences(self, threshold_length: int = 96):

        temp = self.data.iloc[np.where(self.data.iloc[:, -1] == 0.1)[0], :].copy()
        self.data.iloc[np.where(self.data.iloc[:, -1] == 0.1)[0], :] = np.nan
        self.heating_sequences = self.get_sequences(data=self.data)
        self.data.loc[temp.index, :] = temp

        temp = self.data.iloc[np.where(self.data.iloc[:, -1] == 0.9)[0], :].copy()
        self.data.iloc[np.where(self.data.iloc[:, -1] == 0.9)[0], :] = np.nan
        self.cooling_sequences = self.get_sequences(data=self.data)
        self.data.loc[temp.index, :] = temp

        self.sequences = self.heating_sequences + self.cooling_sequences

    def train_test_validation_separation(self, validation_percentage: float = 0.2,
                                         test_percentage: float = 0.1, shuffle: bool = True,
                                         threshold_length: int = 96) -> None:
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
            threshold_length:       Maximal length of a sequence to train the LSTM

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

        # Create the sequences and copy them
        self.prepare_sequences(threshold_length=threshold_length)
        sequences = self.sequences.copy()

        # Shuffle if wanted
        if shuffle:
            np.random.shuffle(sequences)

        # Compute the cumulative length of the sequences
        len_sequences = np.cumsum([len(x) for x in sequences])

        # Given the total length of sequences, define aproximate separations between training
        # validation and testing sets
        train_validation_sep = int((1 - test_percentage - validation_percentage) * len_sequences[-1])
        validation_test_sep = int((1 - test_percentage) * len_sequences[-1])

        # Prepare the lists
        self.train_sequences = []
        self.validation_sequences = []
        self.test_sequences = []

        # Loop over sequences and save them in the right list
        for i, total in enumerate(len_sequences):
            # the second condition ensures to have at list on sequence in validation and testing
            if (total < train_validation_sep) & (i < len(len_sequences) - 2):
                self.train_sequences.append(sequences[i])
            # Same: conditions ensures at list one sequence in each set
            elif ((train_validation_sep <= total < validation_test_sep) & (i < len(len_sequences) - 1)) | (
                    i == len(len_sequences) - 2):
                self.validation_sequences.append(sequences[i])
            else:
                self.test_sequences.append(sequences[i])

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
            sequences = self.train_sequences.copy()
        elif "alidation" in iterator_type:
            sequences = self.validation_sequences.copy()
        elif "est" in iterator_type:
            sequences = self.test_sequences.copy()
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
            yield sequences[batch * batch_size:(batch + 1) * batch_size]

    def build_tensor_from_sequence(self, sequence: list, data=None):
        """
        Input and output generator from given indices corresponding to a batch. Here we overwrite the
        general function as LSTMs require sequential inputs, which are in 3 dimensions, not 2 like
        normal feedforward networks.

        Args:
            sequence:   Sequence to build the input from
            data:       Data to use to build the input from
        Returns:
            The input in the wanted form
        """

        if data is None:
            data = self.data

        # Transform the data from the DataFrame in a tensor to be handled later
        tensor = torch.FloatTensor(data.iloc[sequence, :].values)

        # Return the tensor
        return tensor.to(self.device)

    def build_input_output_from_sequences(self, sequences: list):
        """
        Input and output generator from given indices corresponding to a batch. Additionally,
        the initial hidden state is also computed, which correspond to the previous state
        of each component (i.e. typically the previous temperature)

        Args:
            sequences: sequences of the batch to prepare

        Returns:
            input_tensor:   Batch input of the model
            h_0:            Initial initial state of the model, a dict
            output_dict:    Targets of the model, a dict
        """

        # Ensure the given sequences are a list of list, not only one list
        if type(sequences[0]) == int:
            sequences = [sequences]

        # Iterate over the indices to build the input in the right form
        input_tensor_list = [self.build_tensor_from_sequence(sequence[:-1]) for sequence in sequences]

        # Create the final input tensor, in a way robust to batch of size 1
        # General case: stack the list of tensors together, using pad_sequence to handle variable
        # length sequences
        if len(sequences) > 1:
            input_tensor = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
        # For batch_size 1
        else:
            size = input_tensor_list[0].shape[0]
            input_tensor = input_tensor_list[0].view(1, size, -1)

        # Prepare the output and hidden state dict and loop over component
        output_dict = {component: [] for component in self.components}
        h_0 = {}
        for component in self.components:

            # Build lists of tensors corresponding to each sequence
            # If we want to predict differences
            if self.predict_differences:
                output_dict[component] = [torch.FloatTensor(self.differences[component][sequence[1:]].values)
                                              .to(self.device) for sequence in sequences]
                h_0[component] = [torch.FloatTensor([self.differences[component][sequence[0]]]).to(self.device)
                                  for sequence in sequences]
            # Otherwise just use the data
            else:
                output_dict[component] = [torch.FloatTensor(self.data[component][sequence[1:]].values).to(self.device)
                                          for sequence in sequences]
                h_0[component] = [torch.FloatTensor([self.data[component][sequence[0]]]).to(self.device)
                                  for sequence in sequences]

            # Again, build the final results by taking care of the batch_size=1 case
            # Pad the sequences to handle variable length
            if len(sequences) > 1:
                output_dict[component] = pad_sequence(output_dict[component], batch_first=True,
                                                      padding_value=0).unsqueeze(-1)
                h_0[component] = torch.stack(h_0[component]).unsqueeze(0)

            # Batch size of 1
            else:
                size = output_dict[component][0].shape[0]
                output_dict[component] = output_dict[component][0].view(1, size, -1)
                h_0[component] = h_0[component][0].view(1, len(sequences), 1)

            # output_dict[component] = output_dict[component].to(self.device)
            # h_0[component] = h_0[component].to(self.device)

        if self.hidden_size > 1:
            h_0 = None

        # input_tensor = input_tensor.to(self.device)

        # Return everything
        return input_tensor, h_0, output_dict

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
            # Evaluate each loss
            losses.append(self.losses[component](predictions[component], targets[component]))
            # losses.append(self.losses[component](predictions[component].to(self.device), targets[component].to(self.device)))

        # Return the sum of losses for PyTorch
        return sum(losses)

    def compute_loss(self, batch_sequences):
        """
        Custom function to compute the loss of a batch for that special NN model

        Args:
            batch_indices: The indices of the batch

        Returns:
            The loss
        """

        # Build the input tensor and the output data in the wanted form
        input_tensor, h_0, output_dict = self.build_input_output_from_sequences(batch_sequences)

        # Forward the batch
        predictions, _ = self.model(input_tensor, h_0=h_0)

        # Return the loss
        return self.evaluate_loss(predictions, output_dict)

    def warm_start(self, sequence: list, limit_back: int = 96):
        """
        Function looking back in time to compute a warm start for the LSTM prediction, i.e.
        computing the hidden and cell states at the beginning of the seuqence

        Args:
            sequence:  Sequence to predict
            limit_back: How far back do you want to look

        Returns:
            Both the predictions and the true observations over the horizon to facilitate an
            analysis down the line
        """

        # Put the model in evaluation mode
        self.model.eval()

        # Look back from where the sequence start to get past values. Then use these past values to build
        # up the hidden and cell states values that should arise at the beginning of the prediction
        low_limit = sequence[0]

        # Find the lowest index that is still predictable (i.e. no missing values)
        while (low_limit >= sequence[1] - limit_back) & (low_limit in self.predictable_indices) & (
                self.data.iloc[low_limit, -1] == self.data.iloc[sequence[0], -1]):
            low_limit -= 1

        # Build the sequence previous to the one we want to predict
        previous_sequence = np.arange(low_limit + 1, sequence[1] + 1)

        if len(previous_sequence) == 1:
            previous_sequence = [sequence[0], sequence[1]]

        #if (self.verbose > 0) & (len(previous_sequence) < 16):
         #   print("One potentially bad prediction spotted")

        # Run the LSTM to compute the initial hidden and cell states
        input_tensor, h_0, _ = self.build_input_output_from_sequences(sequences=[previous_sequence])
        _, (h_current, c_current) = self.model(input_tensor, h_0=h_0, warm_start=True)

        return h_current, c_current

    def predict_sequence(self, sequence: list, limit_back: int = 96):
        """
        Function making a prediction over a sequence.

        Args:
            sequence:  Sequence to predict
            limit_back: How far back do you want to look

        Returns:
            Both the predictions and the true observations over the horizon to facilitate an
            analysis down the line
        """

        # Warm start
        h_current, c_current = self.warm_start(sequence=sequence,
                                               limit_back=limit_back)

        # Get the true data that the model will want to predict
        true_data = self.data.iloc[sequence, :]

        # Copy the true data and erase the values we want to predict over the horizon
        predictions = true_data.copy()
        predictions.loc[predictions.index[1:], self.components] = np.nan

        # Build the input and output for the wanted sequence
        input_tensor, h_0, output_dict = self.build_input_output_from_sequences(sequences=[sequence])

        # If the hidden_size is 1, then we directly want the LSTM to predict the temperature. This means
        # That we actually know the previous hidden state, which is computed as h_0 when the input is built
        if self.hidden_size == 1:
            h_current = h_0

        # Given the current hidden and cell state, predict the sequence
        output, _ = self.model(input_tensor,
                               h_0=h_current,
                               c_0=c_current)

        # If the model predicts differences, some manipulations are needed
        if self.predict_differences:

            # Get the components for which we want to predict differences
            columns = [x for x in self.components if x not in self.not_differences_components]

            # First step: put the predictions in a DataFrame
            out = pd.DataFrame(index=predictions.index[1:],
                               columns=columns)
            for col in columns:
                out[col] = output[col].cpu().squeeze().detach().numpy()

            # We can then use either the normalization or standardization parameters to normalize (i.e.
            # standardize) the difference that was predicted by the model
            if self.dataset.is_normalized:
                out = 0.8 * out / (self.dataset.max_[columns] - self.dataset.min_[columns])

            elif self.dataset.is_standardized:
                out = out.divide(self.dataset.std[columns])

            else:
                pass

            # Loop through the components of the model and augment the data with their prediction, adding
            # it to the previous one since the model predicts differences
            for i in range(len(sequence) - 1):
                predictions.loc[predictions.index[i + 1], columns] \
                    = predictions.loc[predictions.index[i], columns] + out.loc[out.index[i], columns].values

            # For the other components not in differences
            for component in self.not_differences_components:
                predictions.loc[predictions.index[1:], component] \
                    = output[component].cpu().squeeze().detach().numpy()

        # Otherwise, if we predict direct values, we can simply recall them
        else:
            for component in self.components:
                predictions.loc[predictions.index[1:], component] = output[component].cpu().squeeze().detach().numpy()

        # Scale both predictions and true data back to the original values - and only keep the data of
        # importance, i.e. the values of each component over the horizon
        if self.dataset.is_normalized:
            predictions = self.dataset.inverse_normalize(predictions[self.components])
            true_data = self.dataset.inverse_normalize(true_data[self.components])

        elif self.dataset.is_standardized:
            predictions = self.dataset.inverse_standardize(predictions[self.components])
            true_data = self.dataset.inverse_standardize(true_data[self.components])
        else:
            pass

        # Return both predictions and true values
        return predictions, true_data

    def aggregate_energy(self, predictions, true_data=None):
        """
        Function to aggregate individual predictions and possibly true data to get the total energy

        Args:
            predictions:    Predictions from the Pytorch module
            true_data:      True data corresponding to the predictions
        """
        # Prepare the tensors
        tot_pred = np.zeros_like(predictions[self.components[0]])
        if true_data is not None:
            tot_true = np.zeros_like(predictions[self.components[0]])

        # Loop over the energy component and aggregate them
        for component in self.components:
            if "Energy" in component:
                tot_pred += predictions[component]
                if true_data is not None:
                    tot_true += true_data[component]

        if true_data is not None:
            return tot_pred, tot_true
        else:
            return tot_pred

    def plot_predictions(self, sequence: list, how_many: int = None, **kwargs) -> None:
        """
        Small function to plot predictions from a certain index into the future over the 'horizon'
        Args:
            sequence:   Sequence to predice
            how_many:   To tune  how many of the components to plot
            **kwargs:   Kwargs of the plots, see util.plots
        """

        # By default, plot each component
        if how_many is None:
            how_many = len(self.components)

        # Compute the predictions and get the true data
        predictions, true_data = self.predict_sequence(sequence=sequence)

        # Loop over the number of components wanted and plot the prediction vs the true observations
        for component in self.components[:how_many]:
            # Design the plot with custom helpers
            _plot_helpers(title=component, **kwargs)

            # Plot both informations
            plt.plot(predictions[component], label="Prediction")
            plt.plot(true_data[component], label="Observations")

            # Save or show the plot
            _save_or_show(legend=True, **kwargs)

    def predictions_analysis(self, **kwargs) -> None:
        """
        Function analyzing the prediction performances of the model in terms of MAE propagation
        over a given horizon. It takes 100 random indices from the data, and predicts over the
        'horizon' starting from there, and analyzing how the MAE evolves along the horizon

        Args:
            horizon:    Horizon to analyze
            **kwargs:   Kwargs of the plots, see util.plots
        """

        # Print the start
        print("\nAnalyzing up to 250 predictions...")

        # Get a copy of the validation indices and shuffle it
        sequences = self.validation_sequences.copy()
        np.random.shuffle(sequences)

        # Take long enough sequences to have interesting predictions
        sequences = [x for x in sequences if len(x) == self.threshold_length + 1]
        sequences = sequences[:250]

        # Build a dictionary of errors for each component and iterate over the indices to predict
        errors = {component: [] for component in self.components}
        for num, sequence in enumerate(sequences):

            # Use the model to predict over the horizon
            predictions, true_data = self.predict_sequence(sequence)

            # Loop over the number of components wanted and plot the prediction vs the true observations
            tot_pred = np.zeros_like(predictions[self.components[0]])
            tot_true = np.zeros_like(predictions[self.components[0]])
            for component in self.components:
                if "Energy" in component:
                    tot_pred += predictions[component]
                    tot_true += true_data[component]

            tot_pred, tot_true = self.aggregate_energy(predictions, true_data)

            # Store the MAE of each component
            for component in self.components:
                if "Energy" not in component:
                    errors[component].append(compute_mae(predictions[component], true_data[component]).values)
                else:
                    errors[component].append(compute_mae(tot_pred, tot_true).values)
                    break

            # Informative print
            if num % 10 == 9:
                print(f"{num + 1} predictions done")

        # Create dictionary to store the mean and max errors at each time step over the horizon and
        # fill it by iterating over the components
        errors_timesteps = {}
        for component in self.components:
            # Nested use of list comprehension to compute the mean and the max errors
            errors_timesteps[component] = [[x[i] for x in errors[component]] + [x[i + 1] for x in errors[component]] +
                                           [x[i + 2] for x in errors[component]] + [x[i + 3] for x in errors[component]]
                                           for i in range(1, len(errors[component][0]), 4)]
            if "Energy" in component:
                break

        # Plot the mean errors using the usual helpers, looping over the components to have
        # all errors in one plot
        # for component in self.components:
        # ylabel = "Energy" if "Energy" in component else "Degrees"
        # _plot_helpers(title=f"Absolute error - {component}", xlabel="Prediction hour", ylabel=ylabel, **kwargs)
        # plt.boxplot(errors_timesteps[component], notch=True, whis=[5, 95], showfliers=True)
        # _save_or_show(**kwargs)

        # Create the subplot and loop over component
        fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(20, 20))
        for i, component in enumerate(self.components[:6]):

            # Don't show outliers for the energy, as they can be big (when the energy peaks the model can
            # be quite wrong) and will deform the plot
            axes[i // 2, i % 2].boxplot(errors_timesteps[component], notch=True, whis=[5, 95],
                                        showfliers=True if "Energy" not in component else False)

            # Ma ke the plot look nice
            axes[i // 2, i % 2].set_title(f"Absolute error - Room {component[-3:]}" if i != 5
                                          else "Absolute error - Energy", size=25)
            if i in [4, 5]:
                axes[i // 2, i % 2].set_xlabel("Prediction hour ahead", size=20)
            if i not in [1, 3]:
                axes[i // 2, i % 2].set_ylabel("Degrees" if i != 5 else "Energy", size=20)
            axes[i // 2, i % 2].tick_params(axis='x', which='major', labelsize=15)
            axes[i // 2, i % 2].tick_params(axis='y', which='major', labelsize=15)

        # Define the right layout, save the figure and plot it
        plt.tight_layout()
        plt.savefig("Errors.png", bbox_inches="tight")
        plt.show()

    def plot_consistency(self, sequence, room: int = 273):
        """
        Function to analyze the physical consistency of the model: plots:
         - the predicted temperatures and energy given the observed patterns of valves opening and closing
         - the predicted values in the case where the valves are kept open along the prediction horizon
         - the predicted values in the case where the valves are kept closed along the prediction horizon

         Args:
             sequence:  the sequence of inputs to analyze
             room:      which room to consider the analysis on
        """

        print("Warning: Working with normalization, not standardization")

        # Make predictions
        y, _ = self.predict_sequence(sequence)
        y1 = y[f"Thermal temperature measurement {room}"]
        y1.name = "Prediction"
        y1_energy = y[f"Energy room {room}"]
        y1_energy.name = "Prediction"

        # Recall te true opening and closing pattern
        temp = self.data.loc[self.data.index[sequence], f"Thermal valve {room}"].values

        # Put the valves to the closed state
        self.data.loc[self.data.index[sequence], f"Thermal valve {room}"] = 0.1
        # Compute the predictions in that case
        y, _ = self.predict_sequence(sequence)
        y2 = y[f"Thermal temperature measurement {room}"]
        y2.name = "All closed"
        y2_energy = y[f"Energy room {room}"]
        y2_energy.name = "All closed"

        # Put the valves to the opened state
        self.data.loc[self.data.index[sequence], f"Thermal valve {room}"] = 0.9
        # Compute the predictions in that case
        y, _ = self.predict_sequence(sequence)
        y3 = y[f"Thermal temperature measurement {room}"]
        y3.name = "All open"
        y3_energy = y[f"Energy room {room}"]
        y3_energy.name = "All open"

        # Bring back the original data so that further operations ar not impacted
        self.data.loc[self.data.index[sequence], f"Thermal valve {room}"] = temp

        # Plot the results
        plot_time_series([y2, y3, y1], ylabel="Degrees", title=f"Temperature room {room}")
        plot_time_series([y2_energy, y3_energy, y1_energy], ylabel="Energy", title=f"Energy consumption room {room}")

    def consistency_analysis(self, sequences=None, boxplots: bool = True, **kwargs):
        """
        Function to analyze the physical consistency of the model, i.e. make sure that the temperature of the room
        when the valves are open is higher (heating case) or lower (cooling case) than when the valves are closed.
        Additionally, we would like our predictions, which are always based on a sequence of opening and closing
        of the valves, to always lie in between those two extrema.
        This analysis is thus performed for each room (and for the energy), i.e. all the components, each of
        them separated into its cooling and heating behavior, with 3 comparisons in each case.

        The plots show aggregated values for each hour (i.e. binning 4 time steps together) to be more
        readable

        Args:
            sequences: the sequences to analyze (if None, then the usual validation sequences are chosen)
            kwargs:    plotting kwargs
        """

        # Define the sequences if none is given
        if sequences is None:
            print("Analyzing up to 500 predictions")
            sequences = self.validation_sequences.copy()
            np.random.shuffle(sequences)

            # Take long enough sequences to have interesting predictions
            sequences = [x for x in sequences if len(x) == self.threshold_length + 1]
            sequences = sequences[:500]

        # Define a dictionary for each type of error, i.e. heating and cooling errors, comparing
        # predictions, valves open and valves closed 2 by 2
        # We will record the 6 arising errors for each component
        cooling_errors_open = {component: [] for component in self.components}
        heating_errors_open = {component: [] for component in self.components}
        cooling_errors_closed = {component: [] for component in self.components}
        heating_errors_closed = {component: [] for component in self.components}
        cooling_errors_extrema = {component: [] for component in self.components}
        heating_errors_extrema = {component: [] for component in self.components}

        # Loop over the sequences to gather the errors
        for num, sequence in enumerate(sequences):

            # First, we need to define if we are in a heating or cooling situation. We thus scale the
            # data back and compare the temperature of the room (average) against the inlet temperature
            # If the former is higher, we are in a cooling situation, otherwise heating

            # Define the columns of interest
            columns = [f"Thermal temperature measurement {room}" for room in ROOMS] + ["Thermal inlet temperature"]
            # Scale the data if needed
            if self.dataset.is_normalized:
                temp_data = self.dataset.inverse_normalize(self.data.loc[self.data.index[sequence], columns])
            elif self.dataset.is_standardized:
                temp_data = self.dataset.inverse_standardize(self.data.loc[self.data.index[sequence], columns])
            else:
                temp_data = self.data.loc[self.data.index[sequence], columns]

            # Define in which case we are: here we allow some noise, using a threshold at 95 time steps
            if np.sum(
                    np.mean(temp_data.loc[:, [f"Thermal temperature measurement {room}" for room in ROOMS]].values,
                            axis=1) > temp_data.loc[:, "Thermal inlet temperature"]) > 95:
                case = "Cooling"
            elif np.sum(
                    np.mean(temp_data.loc[:, [f"Thermal temperature measurement {room}" for room in ROOMS]].values,
                            axis=1) < temp_data.loc[:, "Thermal inlet temperature"]) > 95:
                case = "Heating"
            else:
                case = "Other"
                print("Undecisive case")

            # For the meaningful cases, let's analyze our model
            if case != "Other":

                # First, use the model to predict the influence of the actual pattern of valves
                prediction, _ = self.predict_sequence(sequence)

                # Recall te true opening and closing pattern for later
                valves = [x for x in self.data.columns if "valve" in x]
                temp = self.data.loc[self.data.index[sequence], valves].values

                # Put the valves to the closed state
                self.data.loc[self.data.index[sequence], valves] = 0.1
                # Compute the predictions in that case
                all_closed, _ = self.predict_sequence(sequence)

                # Put the valves to the opened state
                self.data.loc[self.data.index[sequence], valves] = 0.9
                # Compute the predictions in that case
                all_open, _ = self.predict_sequence(sequence)

                # Bring back the original data so that further operations ar not impacted
                self.data.loc[self.data.index[sequence], valves] = temp

                # Loop over the components to gather the errors
                for component in self.components:

                    # Case separation
                    if case == "Cooling":
                        cooling_errors_open[component].append((all_open[component] - prediction[component]).values)
                        cooling_errors_closed[component].append(
                            (prediction[component] - all_closed[component]).values)
                        cooling_errors_extrema[component].append(
                            (all_open[component] - all_closed[component]).values)

                    elif case == "Heating":
                        heating_errors_open[component].append((prediction[component] - all_open[component]).values)
                        heating_errors_closed[component].append(
                            (all_closed[component] - prediction[component]).values)
                        heating_errors_extrema[component].append(
                            (all_closed[component] - all_open[component]).values)

            # Informative print
            if num % 5 == 4:
                print(f"{num + 1} predictions done")

        # Define the dictionaries retaining the errors by time steps
        cooling_errors_open_timesteps = {}
        heating_errors_open_timesteps = {}
        cooling_errors_closed_timesteps = {}
        heating_errors_closed_timesteps = {}
        cooling_errors_extrema_timesteps = {}
        heating_errors_extrema_timesteps = {}

        median_cooling_errors_open_timesteps = {}
        median_heating_errors_open_timesteps = {}
        median_cooling_errors_closed_timesteps = {}
        median_heating_errors_closed_timesteps = {}
        median_cooling_errors_extrema_timesteps = {}
        median_heating_errors_extrema_timesteps = {}

        cooling_errors_open_timesteps_75 = {}
        heating_errors_open_timesteps_75 = {}
        cooling_errors_closed_timesteps_75 = {}
        heating_errors_closed_timesteps_75 = {}
        cooling_errors_extrema_timesteps_75 = {}
        heating_errors_extrema_timesteps_75 = {}

        cooling_errors_open_timesteps_95 = {}
        heating_errors_open_timesteps_95 = {}
        cooling_errors_closed_timesteps_95 = {}
        heating_errors_closed_timesteps_95 = {}
        cooling_errors_extrema_timesteps_95 = {}
        heating_errors_extrema_timesteps_95 = {}

        # Loop over the components to transform the current lists of list of prediction errors to
        # lists of errors for each time step
        for component in self.components:
            # Nested use of list comprehension to compute errors for each time step
            # here we additionally cluster 4 time steps together at each time to make the plot more visible
            # I.e. we gather all the errors of each prediction hour together in on bin
            cooling_errors_open_timesteps[component] = [[x[i] for x in cooling_errors_open[component]] +
                                                        [x[i + 1] for x in cooling_errors_open[component]] +
                                                        [x[i + 2] for x in cooling_errors_open[component]] +
                                                        [x[i + 3] for x in cooling_errors_open[component]]
                                                        for i in
                                                        range(1, len(cooling_errors_open[component][0]), 4)]
            heating_errors_open_timesteps[component] = [[x[i] for x in heating_errors_open[component]] +
                                                        [x[i + 1] for x in heating_errors_open[component]] +
                                                        [x[i + 2] for x in heating_errors_open[component]] +
                                                        [x[i + 3] for x in heating_errors_open[component]]
                                                        for i in
                                                        range(1, len(heating_errors_open[component][0]), 4)]
            cooling_errors_closed_timesteps[component] = [[x[i] for x in cooling_errors_closed[component]] +
                                                          [x[i + 1] for x in cooling_errors_closed[component]] +
                                                          [x[i + 2] for x in cooling_errors_closed[component]] +
                                                          [x[i + 3] for x in cooling_errors_closed[component]]
                                                          for i in
                                                          range(1, len(cooling_errors_closed[component][0]), 4)]
            heating_errors_closed_timesteps[component] = [[x[i] for x in heating_errors_closed[component]] +
                                                          [x[i + 1] for x in heating_errors_closed[component]] +
                                                          [x[i + 2] for x in heating_errors_closed[component]] +
                                                          [x[i + 3] for x in heating_errors_closed[component]]
                                                          for i in
                                                          range(1, len(heating_errors_closed[component][0]), 4)]
            cooling_errors_extrema_timesteps[component] = [[x[i] for x in cooling_errors_extrema[component]] +
                                                           [x[i + 1] for x in cooling_errors_extrema[component]] +
                                                           [x[i + 2] for x in cooling_errors_extrema[component]] +
                                                           [x[i + 3] for x in cooling_errors_extrema[component]]
                                                           for i in
                                                           range(1, len(cooling_errors_extrema[component][0]), 4)]
            heating_errors_extrema_timesteps[component] = [[x[i] for x in heating_errors_extrema[component]] +
                                                           [x[i + 1] for x in heating_errors_extrema[component]] +
                                                           [x[i + 2] for x in heating_errors_extrema[component]] +
                                                           [x[i + 3] for x in heating_errors_extrema[component]]
                                                           for i in
                                                           range(1, len(heating_errors_extrema[component][0]), 4)]

            # Medians
            median_cooling_errors_open_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_open[component]], 50)
                for i in range(len(cooling_errors_open[component][0]))]
            median_heating_errors_open_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_open[component]], 50)
                for i in range(len(heating_errors_open[component][0]))]
            median_cooling_errors_closed_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_closed[component]], 50)
                for i in range(len(cooling_errors_closed[component][0]))]
            median_heating_errors_closed_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_closed[component]], 50)
                for i in range(len(heating_errors_closed[component][0]))]
            median_cooling_errors_extrema_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_extrema[component]], 50)
                for i in range(len(cooling_errors_extrema[component][0]))]
            median_heating_errors_extrema_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_extrema[component]], 50)
                for i in range(len(heating_errors_extrema[component][0]))]

            # 75% percentiles
            cooling_errors_open_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_open[component]], 75)
                for i in range(len(cooling_errors_open[component][0]))]
            heating_errors_open_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_open[component]], 75)
                for i in range(len(heating_errors_open[component][0]))]
            cooling_errors_closed_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_closed[component]], 75)
                for i in range(len(cooling_errors_closed[component][0]))]
            heating_errors_closed_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_closed[component]], 75)
                for i in range(len(heating_errors_closed[component][0]))]
            cooling_errors_extrema_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_extrema[component]], 75)
                for i in range(len(cooling_errors_extrema[component][0]))]
            heating_errors_extrema_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_extrema[component]], 75)
                for i in range(len(heating_errors_extrema[component][0]))]

            # 95% percentiles
            cooling_errors_open_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_open[component]], 95)
                for i in range(len(cooling_errors_open[component][0]))]
            heating_errors_open_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_open[component]], 95)
                for i in range(len(heating_errors_open[component][0]))]
            cooling_errors_closed_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_closed[component]], 95)
                for i in range(len(cooling_errors_closed[component][0]))]
            heating_errors_closed_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_closed[component]], 95)
                for i in range(len(heating_errors_closed[component][0]))]
            cooling_errors_extrema_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_extrema[component]], 95)
                for i in range(len(cooling_errors_extrema[component][0]))]
            heating_errors_extrema_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_extrema[component]], 95)
                for i in range(len(heating_errors_extrema[component][0]))]

        # List all the errors
        errors_ = [median_cooling_errors_open_timesteps, cooling_errors_open_timesteps_75,
                   cooling_errors_open_timesteps_95,
                   median_cooling_errors_closed_timesteps, cooling_errors_closed_timesteps_75,
                   cooling_errors_closed_timesteps_95,
                   median_cooling_errors_extrema_timesteps, cooling_errors_extrema_timesteps_75,
                   cooling_errors_extrema_timesteps_95,
                   median_heating_errors_open_timesteps, heating_errors_open_timesteps_75,
                   heating_errors_open_timesteps_95,
                   median_heating_errors_closed_timesteps, heating_errors_closed_timesteps_75,
                   heating_errors_closed_timesteps_95,
                   median_heating_errors_extrema_timesteps, heating_errors_extrema_timesteps_75,
                   heating_errors_extrema_timesteps_95]

        # list the labels to give to the curves
        labels = ["Open - Pred", "Pred - Closed", "Open - Closed",
                  "Pred - Open", "Closed - Pred", "Closed - Open"]

        # Define colors and line styles
        colors = ["blue", "green", "red"]  # , "black", "orange", "violet"]
        styles = ["solid", "dashed", "dotted"]

        # Define the list of errors for each component
        errors_list = [cooling_errors_open_timesteps, cooling_errors_closed_timesteps,
                       cooling_errors_extrema_timesteps,
                       heating_errors_open_timesteps, heating_errors_closed_timesteps,
                       heating_errors_extrema_timesteps, ]

        # If all the boxplots are wanted
        if boxplots:

            # Corresponding prints
            prints = ["Open vs Prediction Error", "Prediction vs Closed Error", "Open vs Closed Error",
                      "Prediction vs Open Error", "Closed vs Prediction Error", "Closed vs Open Error"]

            # Print all the information gathered
            for component in self.components:

                # Define the label of the y-axis
                ylabel = "Degrees" if "Energy" not in component else "Energy"

                # Print the start of the analysis
                print(f"\n==========================================================\n{component}")
                print("==========================================================")

                # Loop over the 6 errors
                for num, errors in enumerate(errors_list):

                    # Print the case we are looking at
                    if num == 0:
                        print("\n----------------\nHeating Case")
                        print("----------------\n")
                    elif num == 3:
                        print("\n----------------\nCooling Case")
                        print("----------------\n")

                    print(prints[num])
                    # _plot_helpers(xlabel="Prediction time step", ylabel=ylabel, **kwargs)
                    # plt.boxplot(errors, notch=True, whis=[5, 95], showfliers=True)
                    # _save_or_show(**kwargs)

                    # Loop over the errors only keep the positive ones (negative errors mean that the model
                    # was physically sound) and count along the way the good predictions
                    count_good_predictions = 0
                    total_predictions = 0

                    # Define the list of bins of errors for the boxplot and loop along the errors to prepare them
                    error_bins = []
                    for i in range(len(errors[component])):

                        # Count the number of time predictions we have
                        number = len(errors[component][i])
                        # Compute the true errors (i.e. the positive ones)
                        true_errors = [x for x in errors[component][i] if x > 0]

                        # Update the counters
                        count_good_predictions += (number - len(true_errors))
                        total_predictions += number

                        # Append the true errors for the boxplot
                        if len(true_errors) > 0:
                            error_bins.append(true_errors)
                        else:
                            error_bins.append([])

                    # Print the results
                    print(f"{count_good_predictions} out of {total_predictions} comparisons were physically sound,"
                          f"{count_good_predictions / total_predictions * 100:.2f}%")

                    print(f"The error magnitude of the physically inconsistent steps is plotted below:")

                    # Boxplot of the true errors
                    _plot_helpers(xlabel="Prediction hour", ylabel=ylabel, **kwargs)
                    plt.boxplot(error_bins, notch=True, whis=[5, 95], showfliers=True)
                    _save_or_show(**kwargs)

        # Plot all informations in one plot, separating the heating and cooling cases

        # Prepare the plot
        fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16, 25))

        # Loop over the components to analyze the errors
        for i, component in enumerate(self.components[:-1]):
            for num, error in enumerate(errors_):
                if num % 3 == 0:
                    errors = errors_list[int(num / 3)]
                    count_wrong_predictions = 0
                    total_predictions = 0
                    # Compute the error percentage
                    for j in range(len(errors[component])):
                        # Count the number of time predictions we have
                        number = len(errors[component][j])
                        # Update the counters
                        count_wrong_predictions += len([x for x in errors[component][j] if x > 0])
                        total_predictions += number

                # Define the column, label, color and style of the plot
                column = 0 if num < 9 else 1
                label = f"{labels[int(num / 3)]} ({count_wrong_predictions / total_predictions * 100:.0f}%)" if num % 3 == 0 else None
                color = colors[int(num / 3)] if num < 9 else colors[int(num / 3) - 3]
                style = styles[num % 3]

                # Plot the mdeian, 75%, 95% percentile
                toplot = [0 if np.isnan(x) else x for x in error[component]]
                axes[i, column].plot(toplot, label=label, color=color, linestyle=style)

            # Define labels and legends
            axes[i, 0].set_ylabel(f"Room {component[-3:]} ($^\circ$C)", size=22)
            axes[i, 0].tick_params(axis='y', which='major', labelsize=15)
            axes[i, 0].legend(prop={'size': 15})
            axes[i, 1].legend(prop={'size': 15})
        # Set title and x label
        axes[4, 0].set_xlabel("Prediction time step", size=20)
        axes[4, 1].set_xlabel("Prediction time step", size=20)
        axes[0, 0].set_title("Cooling error", size=25)
        axes[4, 0].tick_params(axis='x', which='major', labelsize=15)
        axes[4, 1].tick_params(axis='x', which='major', labelsize=15)

        # Put it in the riht layout, save the figure and plot it
        plt.tight_layout()
        plt.savefig("Consistency.png", bbox_inches="tight")
        plt.show()

        # Plot separating all the 6 different cases
        # New labels
        labels = ["Cooling\nOpen - Pred", "Cooling\nPred - Closed", "Cooling\nOpen - Closed",
                  "Heating\nPred - Open", "Heating\nClosed - Pred", "Heating\nClosed - Open"]

        # Prepare the figure and loop over the components to analyze them
        fig, axes = plt.subplots(nrows=5, ncols=6, sharex=True, sharey=True, figsize=(36, 25))
        for i, component in enumerate(self.components[:-1]):
            # Recall the total wrong predictions, predictions and maximum error for each room
            wrong_predictions = 0
            predictions = 0
            maxima = []
            for num, error in enumerate(errors_):
                if num % 3 == 0:
                    errors = errors_list[int(num / 3)]
                    count_wrong_predictions = 0
                    total_predictions = 0
                    # Compute the error percentages
                    for j in range(len(errors[component])):
                        # Count the number of time predictions we have
                        number = len(errors[component][j])
                        # Update the counters
                        count_wrong_predictions += len([x for x in errors[component][j] if x > 0])
                        total_predictions += number

                    wrong_predictions += count_wrong_predictions
                    predictions += total_predictions

                if num % 3 == 2:
                    maxima.append(np.nanmax(error[component]))

                # Define the column label, color and line style
                column = int(num / 3)
                label = f"{count_wrong_predictions / total_predictions * 100:.0f}%" if num % 3 == 0 else None
                color = colors[int(num / 3)] if num < 9 else colors[int(num / 3) - 3]
                style = styles[num % 3]

                # Plot the info and legend
                toplot = [0 if np.isnan(x) else x for x in error[component]]
                axes[i, column].plot(toplot, label=label, color=color, linestyle=style)
                axes[i, column].legend(prop={'size': 20})

            # Set the labels
            axes[i, 0].set_ylabel(f"Room {component[-3:]} ($^\circ$C)", size=22)
            axes[i, 0].tick_params(axis='y', which='major', labelsize=15)

            # Print the statistics
            print(f"\n________________________________\n{component[7:]}:")
            print(f"  The model is physically inconsistent {wrong_predictions / predictions * 100:.0f}% of the time.")
            print(
                f"  When wrong, the model makes an error smaller than {np.nanmax(maxima):.2f} C with 95% probability.")

        # Set the labels and titles
        for j in range(6):
            axes[0, j].set_title(labels[j], size=25)
            axes[4, j].set_xlabel("Prediction time step", size=20)
            axes[4, j].tick_params(axis='x', which='major', labelsize=15)

        # Define the right layout, save the figure and plot it
        plt.tight_layout()
        plt.savefig("ConsistencyBis.png", bbox_inches="tight")
        plt.show()


class LSTMModel_old(BaseModel_old):
    """
    Class using LSTMs to model each component, by branching the input. If the argument NN is true, then
    the output of the LSTM is fed to a neural network, which yields the final output.
    """

    def __init__(self, nest_data: NESTData, model_kwargs: dict = model_kwargs):
        """
        Create the model using the wanted arguments
        Args:
            nest_data:     The data to train/test the model
            model_kwargs:  All kinds of arguments (see parameters.py)
        """

        # Initialize a general model
        super().__init__(dataset=nest_data.dataset,
                         data=nest_data.data,
                         differences=nest_data.differences,
                         model_kwargs=model_kwargs)

        print("\nConstructing the LSTM model...")
        print("WARNING: Current implementation assume normalization, not robust to standardization")
        print("Indeed, the output is currently passed through a sigmoid which crushes it between 0 and 1")

        self.threshold_length = model_kwargs["threshold_length"]
        self.n_autoregression = 1 # needed to save and load the right indices/models
        self.autoregressive = self.data.columns # Again needed
        self.hidden_size = model_kwargs["hidden_size"]
        self.NN = model_kwargs["NN"]

        # Build the input indices to link to each component, i.e. define for each component which
        # indices to take in the input
        self._build_components_inputs_indices()

        # Training, validation and testing indices separation
        self.train_test_validation_separation(validation_percentage=model_kwargs["validation_percentage"],
                                              test_percentage=model_kwargs["test_percentage"],
                                              threshold_length=model_kwargs["threshold_length"])

        # Build the model itself, using the custom LSTM module - if 'NN' is True, then a neraul network
        # is created after the LSTM to build the output
        self.model = LSTM_old(device=self.device,
                          components_inputs_indices=self.components_inputs_indices,
                          hidden_size=model_kwargs["hidden_size"],
                          num_layers=model_kwargs["num_layers"],
                          NN=model_kwargs["NN"],
                          hidden_sizes=model_kwargs["hidden_sizes"],
                          output_size=model_kwargs["output_size"])

        # Define the losses (one for each component) and the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_kwargs["learning_rate"])

        # Try to load an existing model
        if model_kwargs["load_model"]:
            self.load_model()
        self.model = self.model.to(self.device)

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

        # Loop over the sensors and components to fill the lists of indices for each component
        for index, sensor in enumerate(self.data.columns):
            for component in self.components:
                if sensor in self.components_inputs_dict[component]:
                    self.components_inputs_indices[component].append(index)

    def get_sequences(self, threshold_length: int = 96) -> None:
        """
        Function to prepare sequence of data, i.e. consecutive timestamps where there are no missing
        values. The threshold length avoids the sequences being too long (Truncated Back Propagation
        Through Time) to improve training.
        """

        # Use the global function to get all indices with no missing values
        self.predictable_indices = self.get_predictable_indices(n_autoregression=1)

        # Get the jumps, i.e. places where missing values intervene - the difference between jumps
        # corresponds to the length of data without missing values
        jumps = list(np.where(np.diff(self.predictable_indices) > 1)[0] + 1)

        # Prepare the list to store sequences
        sequences = []

        # Loop over the sequences
        for beginning, end in zip([0] + jumps, jumps + [len(self.predictable_indices)-1]):

            # Check the sequence length: if it is higher than the threshold, break it down
            if end - beginning > threshold_length - 1:
                temp_beg = beginning
                temp_end = beginning + threshold_length

                # Iterate with a sliding window and recall all sequences
                while temp_end <= end:
                    sequences.append([self.predictable_indices[temp_beg] - 1] +
                                     self.predictable_indices[temp_beg: temp_end])
                    temp_beg += 1
                    temp_end += 1

            # Else the sequence is stored as is
            else:
                sequences.append([self.predictable_indices[beginning] - 1] +
                                 self.predictable_indices[beginning: end])

        # Save the computations as an attribute
        self.sequences = sequences

    def train_test_validation_separation(self, validation_percentage: float = 0.2,
                                         test_percentage: float = 0.1, shuffle: bool = True,
                                         threshold_length: int = 96) -> None:
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
            threshold_length:       Maximal length of a sequence to train the LSTM

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

        # Create the sequences and copy them
        self.get_sequences(threshold_length=threshold_length)
        sequences = self.sequences.copy()

        # Shuffle if wanted
        if shuffle:
            np.random.shuffle(sequences)

        # Compute the cumulative length of the sequences
        len_sequences = np.cumsum([len(x) for x in sequences])

        # Given the total length of sequences, define aproximate separations between training
        # validation and testing sets
        train_validation_sep = int((1 - test_percentage - validation_percentage) * len_sequences[-1])
        validation_test_sep = int((1 - test_percentage) * len_sequences[-1])

        # Prepare the lists
        self.train_sequences = []
        self.validation_sequences = []
        self.test_sequences = []

        # Loop over sequences and save them in the right list
        for i, total in enumerate(len_sequences):
            # the second condition ensures to have at list on sequence in validation and testing
            if (total < train_validation_sep) & (i < len(len_sequences) - 2):
                self.train_sequences.append(sequences[i])
            # Same: conditions ensures at list one sequence in each set
            elif ((train_validation_sep <= total < validation_test_sep) & (i < len(len_sequences) - 1)) | (
                    i == len(len_sequences) - 2):
                self.validation_sequences.append(sequences[i])
            else:
                self.test_sequences.append(sequences[i])

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
            sequences = self.train_sequences.copy()
        elif "alidation" in iterator_type:
            sequences = self.validation_sequences.copy()
        elif "est" in iterator_type:
            sequences = self.test_sequences.copy()
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
            yield sequences[batch * batch_size:(batch + 1) * batch_size]

    def build_tensor_from_sequence(self, sequence: list, data=None):
        """
        Input and output generator from given indices corresponding to a batch. Here we overwrite the
        general function as LSTMs require sequential inputs, which are in 3 dimensions, not 2 like
        normal feedforward networks.

        Args:
            sequence:   Sequence to build the input from
            data:       Data to use to build the input from
        Returns:
            The input in the wanted form
        """

        if data is None:
            data = self.data

        # Transform the data from the DataFrame in a tensor to be handled later
        tensor = torch.FloatTensor(data.iloc[sequence, :].values)

        # Return the tensor
        return tensor.to(self.device)

    def build_input_output_from_sequences(self, sequences: list):
        """
        Input and output generator from given indices corresponding to a batch. Additionally,
        the initial hidden state is also computed, which correspond to the previous state
        of each component (i.e. typically the previous temperature)

        Args:
            sequences: sequences of the batch to prepare

        Returns:
            input_tensor:   Batch input of the model
            h_0:            Initial initial state of the model, a dict
            output_dict:    Targets of the model, a dict
        """

        # Ensure the given sequences are a list of list, not only one list
        if type(sequences[0]) == int:
            sequences = [sequences]

        # Iterate over the indices to build the input in the right form
        input_tensor_list = [self.build_tensor_from_sequence(sequence[:-1]) for sequence in sequences]

        # Create the final input tensor, in a way robust to batch of size 1
        # General case: stack the list of tensors together, using pad_sequence to handle variable
        # length sequences
        if len(sequences) > 1:
            input_tensor = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
        # For batch_size 1
        else:
            size = input_tensor_list[0].shape[0]
            input_tensor = input_tensor_list[0].view(1, size, -1)

        # Prepare the output and hidden state dict and loop over component
        output_dict = {component: [] for component in self.components}
        h_0 = {}
        for component in self.components:

            # Build lists of tensors corresponding to each sequence
            # If we want to predict differences
            if self.predict_differences:
                output_dict[component] = [torch.FloatTensor(self.differences[component][sequence[1:]].values)
                                              .to(self.device) for sequence in sequences]
                h_0[component] = [torch.FloatTensor([self.differences[component][sequence[0]]]).to(self.device)
                                  for sequence in sequences]
            # Otherwise just use the data
            else:
                output_dict[component] = [torch.FloatTensor(self.data[component][sequence[1:]].values).to(self.device)
                                          for sequence in sequences]
                h_0[component] = [torch.FloatTensor([self.data[component][sequence[0]]]).to(self.device)
                                  for sequence in sequences]

            # Again, build the final results by taking care of the batch_size=1 case
            # Pad the sequences to handle variable length
            if len(sequences) > 1:
                output_dict[component] = pad_sequence(output_dict[component], batch_first=True,
                                                      padding_value=0).unsqueeze(-1)
                h_0[component] = torch.stack(h_0[component]).unsqueeze(0)

            # Batch size of 1
            else:
                size = output_dict[component][0].shape[0]
                output_dict[component] = output_dict[component][0].view(1, size, -1)
                h_0[component] = h_0[component][0].view(1, len(sequences), 1)

            #output_dict[component] = output_dict[component].to(self.device)
            #h_0[component] = h_0[component].to(self.device)

        if self.hidden_size > 1:
            h_0 = None

        #input_tensor = input_tensor.to(self.device)

        # Return everything
        return input_tensor, h_0, output_dict

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
            # Evaluate each loss
            losses.append(self.losses[component](predictions[component], targets[component]))
            #losses.append(self.losses[component](predictions[component].to(self.device), targets[component].to(self.device)))

        # Return the sum of losses for PyTorch
        return sum(losses)

    def compute_loss(self, batch_sequences):
        """
        Custom function to compute the loss of a batch for that special NN model

        Args:
            batch_indices: The indices of the batch

        Returns:
            The loss
        """

        # Build the input tensor and the output data in the wanted form
        input_tensor, h_0, output_dict = self.build_input_output_from_sequences(batch_sequences)

        # Forward the batch
        predictions, _ = self.model(input_tensor, h_0=h_0)

        # Return the loss
        return self.evaluate_loss(predictions, output_dict)

    def warm_start(self, sequence: list, limit_back: int = 96):
        """
        Function looking back in time to compute a warm start for the LSTM prediction, i.e.
        computing the hidden and cell states at the beginning of the seuqence

        Args:
            sequence:  Sequence to predict
            limit_back: How far back do you want to look

        Returns:
            Both the predictions and the true observations over the horizon to facilitate an
            analysis down the line
        """

        # Put the model in evaluation mode
        self.model.eval()

        # Look back from where the sequence start to get past values. Then use these past values to build
        # up the hidden and cell states values that should arise at the beginning of the prediction
        low_limit = sequence[0]

        # Find the lowest index that is still predictable (i.e. no missing values)
        while (low_limit >= sequence[1] - limit_back) and low_limit in self.predictable_indices:
            low_limit -= 1

        # Build the sequence previous to the one we want to predict
        previous_sequence = np.arange(low_limit, sequence[1] + 1)

        if (self.verbose > 0) & (len(previous_sequence) < 16):
            print("One potentially bad prediction spotted")

        # Run the LSTM to compute the initial hidden and cell states
        input_tensor, h_0, _ = self.build_input_output_from_sequences(sequences=[previous_sequence])
        _, (h_current, c_current) = self.model(input_tensor, h_0=h_0)

        return h_current, c_current

    def predict_sequence(self, sequence: list, limit_back: int = 96):
        """
        Function making a prediction over a sequence.

        Args:
            sequence:  Sequence to predict
            limit_back: How far back do you want to look

        Returns:
            Both the predictions and the true observations over the horizon to facilitate an
            analysis down the line
        """

        # Warm start
        h_current, c_current = self.warm_start(sequence=sequence,
                                                 limit_back=limit_back)

        # Get the true data that the model will want to predict
        true_data = self.data.iloc[sequence, :]

        # Copy the true data and erase the values we want to predict over the horizon
        predictions = true_data.copy()
        predictions.loc[predictions.index[1:], self.components] = np.nan

        # Build the input and output for the wanted sequence
        input_tensor, h_0, output_dict = self.build_input_output_from_sequences(sequences=[sequence])

        # If the hidden_size is 1, then we directly want the LSTM to predict the temperature. This means
        # That we actually know the previous hidden state, which is computed as h_0 when the input is built
        if self.hidden_size == 1:
            h_current = h_0

        # Given the current hidden and cell state, predict the sequence
        output, _ = self.model(input_tensor,
                               h_0=h_current,
                               c_0=c_current)

        # If the model predicts differences, some manipulations are needed
        if self.predict_differences:

            # Get the components for which we want to predict differences
            columns = [x for x in self.components if x not in self.not_differences_components]

            # First step: put the predictions in a DataFrame
            out = pd.DataFrame(index=predictions.index[1:],
                               columns=columns)
            for col in columns:
                out[col] = output[col].squeeze().detach().numpy()

            # We can then use either the normalization or standardization parameters to normalize (i.e.
            # standardize) the difference that was predicted by the model
            if self.dataset.is_normalized:
                out = 0.8 * out / (self.dataset.max_[columns] - self.dataset.min_[columns])

            elif self.dataset.is_standardized:
                out = out.divide(self.dataset.std[columns])

            else:
                pass

            # Loop through the components of the model and augment the data with their prediction, adding
            # it to the previous one since the model predicts differences
            for i in range(len(sequence) - 1):
                predictions.loc[predictions.index[i + 1], columns] \
                    = predictions.loc[predictions.index[i], columns] + out.loc[out.index[i], columns].values

            # For the other components not in differences
            for component in self.not_differences_components:
                predictions.loc[predictions.index[1:], component] \
                    = output[component].squeeze().detach().numpy()

        # Otherwise, if we predict direct values, we can simply recall them
        else:
            for component in self.components:
                predictions.loc[predictions.index[1:], component] = output[component].squeeze().detach().numpy()

        # Scale both predictions and true data back to the original values - and only keep the data of
        # importance, i.e. the values of each component over the horizon
        if self.dataset.is_normalized:
            predictions = self.dataset.inverse_normalize(predictions[self.components])
            true_data = self.dataset.inverse_normalize(true_data[self.components])

        elif self.dataset.is_standardized:
            predictions = self.dataset.inverse_standardize(predictions[self.components])
            true_data = self.dataset.inverse_standardize(true_data[self.components])
        else:
            pass

        # Return both predictions and true values
        return predictions, true_data

    def plot_predictions(self, sequence: list, how_many: int = None, **kwargs) -> None:
        """
        Small function to plot predictions from a certain index into the future over the 'horizon'
        Args:
            sequence:   Sequence to predice
            how_many:   To tune  how many of the components to plot
            **kwargs:   Kwargs of the plots, see util.plots
        """

        # By default, plot each component
        if how_many is None:
            how_many = len(self.components)

        # Compute the predictions and get the true data
        predictions, true_data = self.predict_sequence(sequence=sequence)

        # Loop over the number of components wanted and plot the prediction vs the true observations
        for component in self.components[:how_many]:
            # Design the plot with custom helpers
            _plot_helpers(title=component, **kwargs)

            # Plot both informations
            plt.plot(predictions[component], label="Prediction")
            plt.plot(true_data[component], label="Observations")

            # Save or show the plot
            _save_or_show(legend=True, **kwargs)

    def predictions_analysis(self, **kwargs) -> None:
        """
        Function analyzing the prediction performances of the model in terms of MAE propagation
        over a given horizon. It takes 100 random indices from the data, and predicts over the
        'horizon' starting from there, and analyzing how the MAE evolves along the horizon

        Args:
            horizon:    Horizon to analyze
            **kwargs:   Kwargs of the plots, see util.plots
        """

        # Print the start
        print("\nAnalyzing up to 250 predictions...")

        # Get a copy of the validation indices and shuffle it
        sequences = self.validation_sequences.copy()
        np.random.shuffle(sequences)

        # Take long enough sequences to have interesting predictions
        sequences = [x for x in sequences if len(x) == self.threshold_length + 1]
        sequences = sequences[:250]

        # Build a dictionary of errors for each component and iterate over the indices to predict
        errors = {component: [] for component in self.components}
        for num, sequence in enumerate(sequences):

            # Use the model to predict over the horizon
            predictions, true_data = self.predict_sequence(sequence)

            # Store the MAE of each component
            for component in self.components:
                errors[component].append(compute_mae(predictions[component], true_data[component]).values)

            # Informative print
            if num % 10 == 9:
                print(f"{num + 1} predictions done")

        # Create dictionary to store the mean and max errors at each time step over the horizon and
        # fill it by iterating over the components
        errors_timesteps = {}
        for component in self.components:
            # Nested use of list comprehension to compute the mean and the max errors
            errors_timesteps[component] = [[x[i] for x in errors[component]] + [x[i+1] for x in errors[component]] +
                                           [x[i+2] for x in errors[component]] + [x[i+3] for x in errors[component]]
                                           for i in range(1, len(errors[component][0]), 4)]

        # Plot the mean errors using the usual helpers, looping over the components to have
        # all errors in one plot
        for component in self.components:
            ylabel = "Energy" if "energy" in component else "Degrees"
            _plot_helpers(title=f"Absolute error - {component}", xlabel="Prediction hour", ylabel=ylabel, **kwargs)
            plt.boxplot(errors_timesteps[component], notch=True, whis=[5, 95], showfliers=True)
            _save_or_show(**kwargs)

    def plot_consistency(self, sequence, room: int = 273):
        """
        Function to analyze the physical consistency of the model: plots:
         - the predicted temperatures and energy given the observed patterns of valves opening and closing
         - the predicted values in the case where the valves are kept open along the prediction horizon
         - the predicted values in the case where the valves are kept closed along the prediction horizon

         Args:
             sequence:  the sequence of inputs to analyze
             room:      which room to consider the analysis on
        """

        print("Warning: Working with normalization, not standardization")

        # Make predictions
        y, _ = self.predict_sequence(sequence)
        y1 = y[f"Thermal temperature measurement {room}"]
        y1.name = "Prediction"
        y1_energy = y["Thermal total energy"]
        y1_energy.name = "Prediction"

        # Recall te true opening and closing pattern
        temp = self.data.loc[self.data.index[sequence], f"Thermal valve {room}"].values

        # Put the valves to the closed state
        self.data.loc[self.data.index[sequence], f"Thermal valve {room}"] = 0.1
        # Compute the predictions in that case
        y, _ = self.predict_sequence(sequence)
        y2 = y[f"Thermal temperature measurement {room}"]
        y2.name = "All closed"
        y2_energy = y["Thermal total energy"]
        y2_energy.name = "All closed"

        # Put the valves to the opened state
        self.data.loc[self.data.index[sequence], f"Thermal valve {room}"] = 0.9
        # Compute the predictions in that case
        y, _ = self.predict_sequence(sequence)
        y3 = y[f"Thermal temperature measurement {room}"]
        y3.name = "All open"
        y3_energy = y["Thermal total energy"]
        y3_energy.name = "All open"

        # Bring back the original data so that further operations ar not impacted
        self.data.loc[self.data.index[sequence], f"Thermal valve {room}"] = temp

        # Plot the results
        plot_time_series([y2, y3, y1], ylabel="Degrees", title=f"Temperature room {room}")
        plot_time_series([y2_energy, y3_energy, y1_energy], ylabel="Energy", title="Full energy consumption of UMAR")

    def consistency_analysis(self, sequences=None, boxplots: bool = True, **kwargs):
        """
        Function to analyze the physical consistency of the model, i.e. make sure that the temperature of the room
        when the valves are open is higher (heating case) or lower (cooling case) than when the valves are closed.
        Additionally, we would like our predictions, which are always based on a sequence of opening and closing
        of the valves, to always lie in between those two extrema.
        This analysis is thus performed for each room (and for the energy), i.e. all the components, each of
        them separated into its cooling and heating behavior, with 3 comparisons in each case.

        The plots show aggregated values for each hour (i.e. binning 4 time steps together) to be more
        readable

        Args:
            sequences: the sequences to analyze (if None, then the usual validation sequences are chosen)
            kwargs:    plotting kwargs
        """

        # Define the sequences if none is given
        if sequences is None:
            print("Analyzing up to 500 predictions")
            sequences = self.validation_sequences.copy()
            np.random.shuffle(sequences)

            # Take long enough sequences to have interesting predictions
            sequences = [x for x in sequences if len(x) == self.threshold_length + 1]
            sequences = sequences[:500]

        # Define a dictionary for each type of error, i.e. heating and cooling errors, comparing
        # predictions, valves open and valves closed 2 by 2
        # We will record the 6 arising errors for each component
        cooling_errors_open = {component: [] for component in self.components}
        heating_errors_open = {component: [] for component in self.components}
        cooling_errors_closed = {component: [] for component in self.components}
        heating_errors_closed = {component: [] for component in self.components}
        cooling_errors_extrema = {component: [] for component in self.components}
        heating_errors_extrema = {component: [] for component in self.components}

        # Loop over the sequences to gather the errors
        for num, sequence in enumerate(sequences):

            # First, we need to define if we are in a heating or cooling situation. We thus scale the
            # data back and compare the temperature of the room (average) against the inlet temperature
            # If the former is higher, we are in a cooling situation, otherwise heating

            # Define the columns of interest
            columns = [f"Thermal temperature measurement {room}" for room in ROOMS] + ["Thermal inlet temperature"]
            # Scale the data if needed
            if self.dataset.is_normalized:
                temp_data = self.dataset.inverse_normalize(self.data.loc[self.data.index[sequence], columns])
            elif self.dataset.is_standardized:
                temp_data = self.dataset.inverse_standardize(self.data.loc[self.data.index[sequence], columns])
            else:
                temp_data = self.data.loc[self.data.index[sequence], columns]

            # Define in which case we are: here we allow some noise, using a threshold at 95 time steps
            if np.sum(
                    np.mean(temp_data.loc[:, [f"Thermal temperature measurement {room}" for room in ROOMS]].values,
                            axis=1) > temp_data.loc[:, "Thermal inlet temperature"]) > 95:
                case = "Cooling"
            elif np.sum(
                    np.mean(temp_data.loc[:, [f"Thermal temperature measurement {room}" for room in ROOMS]].values,
                            axis=1) < temp_data.loc[:, "Thermal inlet temperature"]) > 95:
                case = "Heating"
            else:
                case = "Other"
                print("Undecisive case")

            # For the meaningful cases, let's analyze our model
            if case != "Other":

                # First, use the model to predict the influence of the actual pattern of valves
                prediction, _ = self.predict_sequence(sequence)

                # Recall te true opening and closing pattern for later
                valves = [x for x in self.data.columns if "valve" in x]
                temp = self.data.loc[self.data.index[sequence], valves].values

                # Put the valves to the closed state
                self.data.loc[self.data.index[sequence], valves] = 0.1
                # Compute the predictions in that case
                all_closed, _ = self.predict_sequence(sequence)

                # Put the valves to the opened state
                self.data.loc[self.data.index[sequence], valves] = 0.9
                # Compute the predictions in that case
                all_open, _ = self.predict_sequence(sequence)

                # Bring back the original data so that further operations ar not impacted
                self.data.loc[self.data.index[sequence], valves] = temp

                # Loop over the components to gather the errors
                for component in self.components:

                    # Case separation
                    if case == "Cooling":
                        cooling_errors_open[component].append((all_open[component] - prediction[component]).values)
                        cooling_errors_closed[component].append(
                            (prediction[component] - all_closed[component]).values)
                        cooling_errors_extrema[component].append(
                            (all_open[component] - all_closed[component]).values)

                    elif case == "Heating":
                        heating_errors_open[component].append((prediction[component] - all_open[component]).values)
                        heating_errors_closed[component].append(
                            (all_closed[component] - prediction[component]).values)
                        heating_errors_extrema[component].append(
                            (all_closed[component] - all_open[component]).values)

            # Informative print
            if num % 5 == 4:
                print(f"{num + 1} predictions done")

        # Define the dictionaries retaining the errors by time steps
        cooling_errors_open_timesteps = {}
        heating_errors_open_timesteps = {}
        cooling_errors_closed_timesteps = {}
        heating_errors_closed_timesteps = {}
        cooling_errors_extrema_timesteps = {}
        heating_errors_extrema_timesteps = {}

        median_cooling_errors_open_timesteps = {}
        median_heating_errors_open_timesteps = {}
        median_cooling_errors_closed_timesteps = {}
        median_heating_errors_closed_timesteps = {}
        median_cooling_errors_extrema_timesteps = {}
        median_heating_errors_extrema_timesteps = {}

        cooling_errors_open_timesteps_75 = {}
        heating_errors_open_timesteps_75 = {}
        cooling_errors_closed_timesteps_75 = {}
        heating_errors_closed_timesteps_75 = {}
        cooling_errors_extrema_timesteps_75 = {}
        heating_errors_extrema_timesteps_75 = {}

        cooling_errors_open_timesteps_95 = {}
        heating_errors_open_timesteps_95 = {}
        cooling_errors_closed_timesteps_95 = {}
        heating_errors_closed_timesteps_95 = {}
        cooling_errors_extrema_timesteps_95 = {}
        heating_errors_extrema_timesteps_95 = {}

        # Loop over the components to transform the current lists of list of prediction errors to
        # lists of errors for each time step
        for component in self.components:
            # Nested use of list comprehension to compute errors for each time step
            # here we additionally cluster 4 time steps together at each time to make the plot more visible
            # I.e. we gather all the errors of each prediction hour together in on bin
            cooling_errors_open_timesteps[component] = [[x[i] for x in cooling_errors_open[component]] +
                                                        [x[i + 1] for x in cooling_errors_open[component]] +
                                                        [x[i + 2] for x in cooling_errors_open[component]] +
                                                        [x[i + 3] for x in cooling_errors_open[component]]
                                                        for i in
                                                        range(1, len(cooling_errors_open[component][0]), 4)]
            heating_errors_open_timesteps[component] = [[x[i] for x in heating_errors_open[component]] +
                                                        [x[i + 1] for x in heating_errors_open[component]] +
                                                        [x[i + 2] for x in heating_errors_open[component]] +
                                                        [x[i + 3] for x in heating_errors_open[component]]
                                                        for i in
                                                        range(1, len(heating_errors_open[component][0]), 4)]
            cooling_errors_closed_timesteps[component] = [[x[i] for x in cooling_errors_closed[component]] +
                                                          [x[i + 1] for x in cooling_errors_closed[component]] +
                                                          [x[i + 2] for x in cooling_errors_closed[component]] +
                                                          [x[i + 3] for x in cooling_errors_closed[component]]
                                                          for i in
                                                          range(1, len(cooling_errors_closed[component][0]), 4)]
            heating_errors_closed_timesteps[component] = [[x[i] for x in heating_errors_closed[component]] +
                                                          [x[i + 1] for x in heating_errors_closed[component]] +
                                                          [x[i + 2] for x in heating_errors_closed[component]] +
                                                          [x[i + 3] for x in heating_errors_closed[component]]
                                                          for i in
                                                          range(1, len(heating_errors_closed[component][0]), 4)]
            cooling_errors_extrema_timesteps[component] = [[x[i] for x in cooling_errors_extrema[component]] +
                                                           [x[i + 1] for x in cooling_errors_extrema[component]] +
                                                           [x[i + 2] for x in cooling_errors_extrema[component]] +
                                                           [x[i + 3] for x in cooling_errors_extrema[component]]
                                                           for i in
                                                           range(1, len(cooling_errors_extrema[component][0]), 4)]
            heating_errors_extrema_timesteps[component] = [[x[i] for x in heating_errors_extrema[component]] +
                                                           [x[i + 1] for x in heating_errors_extrema[component]] +
                                                           [x[i + 2] for x in heating_errors_extrema[component]] +
                                                           [x[i + 3] for x in heating_errors_extrema[component]]
                                                           for i in
                                                           range(1, len(heating_errors_extrema[component][0]), 4)]

            # Medians
            median_cooling_errors_open_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_open[component]], 50)
                for i in range(len(cooling_errors_open[component][0]))]
            median_heating_errors_open_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_open[component]], 50)
                for i in range(len(heating_errors_open[component][0]))]
            median_cooling_errors_closed_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_closed[component]], 50)
                for i in range(len(cooling_errors_closed[component][0]))]
            median_heating_errors_closed_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_closed[component]], 50)
                for i in range(len(heating_errors_closed[component][0]))]
            median_cooling_errors_extrema_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_extrema[component]], 50)
                for i in range(len(cooling_errors_extrema[component][0]))]
            median_heating_errors_extrema_timesteps[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_extrema[component]], 50)
                for i in range(len(heating_errors_extrema[component][0]))]

            # 75% percentiles
            cooling_errors_open_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_open[component]], 75)
                for i in range(len(cooling_errors_open[component][0]))]
            heating_errors_open_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_open[component]], 75)
                for i in range(len(heating_errors_open[component][0]))]
            cooling_errors_closed_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_closed[component]], 75)
                for i in range(len(cooling_errors_closed[component][0]))]
            heating_errors_closed_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_closed[component]], 75)
                for i in range(len(heating_errors_closed[component][0]))]
            cooling_errors_extrema_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_extrema[component]], 75)
                for i in range(len(cooling_errors_extrema[component][0]))]
            heating_errors_extrema_timesteps_75[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_extrema[component]], 75)
                for i in range(len(heating_errors_extrema[component][0]))]

            # 95% percentiles
            cooling_errors_open_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_open[component]], 95)
                for i in range(len(cooling_errors_open[component][0]))]
            heating_errors_open_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_open[component]], 95)
                for i in range(len(heating_errors_open[component][0]))]
            cooling_errors_closed_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_closed[component]], 95)
                for i in range(len(cooling_errors_closed[component][0]))]
            heating_errors_closed_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_closed[component]], 95)
                for i in range(len(heating_errors_closed[component][0]))]
            cooling_errors_extrema_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in cooling_errors_extrema[component]], 95)
                for i in range(len(cooling_errors_extrema[component][0]))]
            heating_errors_extrema_timesteps_95[component] = [
                np.nanpercentile([x[i] if x[i] > 0 else np.nan for x in heating_errors_extrema[component]], 95)
                for i in range(len(heating_errors_extrema[component][0]))]

        # List all the errors
        errors_ = [median_cooling_errors_open_timesteps, cooling_errors_open_timesteps_75,
                   cooling_errors_open_timesteps_95,
                   median_cooling_errors_closed_timesteps, cooling_errors_closed_timesteps_75,
                   cooling_errors_closed_timesteps_95,
                   median_cooling_errors_extrema_timesteps, cooling_errors_extrema_timesteps_75,
                   cooling_errors_extrema_timesteps_95,
                   median_heating_errors_open_timesteps, heating_errors_open_timesteps_75,
                   heating_errors_open_timesteps_95,
                   median_heating_errors_closed_timesteps, heating_errors_closed_timesteps_75,
                   heating_errors_closed_timesteps_95,
                   median_heating_errors_extrema_timesteps, heating_errors_extrema_timesteps_75,
                   heating_errors_extrema_timesteps_95]

        # list the labels to give to the curves
        labels = ["Open - Pred", "Pred - Closed", "Open - Closed",
                  "Pred - Open", "Closed - Pred", "Closed - Open"]

        # Define colors and line styles
        colors = ["blue", "green", "red"]  # , "black", "orange", "violet"]
        styles = ["solid", "dashed", "dotted"]

        # Define the list of errors for each component
        errors_list = [cooling_errors_open_timesteps, cooling_errors_closed_timesteps,
                       cooling_errors_extrema_timesteps,
                       heating_errors_open_timesteps, heating_errors_closed_timesteps,
                       heating_errors_extrema_timesteps, ]

        # If all the boxplots are wanted
        if boxplots:

            # Corresponding prints
            prints = ["Open vs Prediction Error", "Prediction vs Closed Error", "Open vs Closed Error",
                      "Prediction vs Open Error", "Closed vs Prediction Error", "Closed vs Open Error"]

            # Print all the information gathered
            for component in self.components:

                # Define the label of the y-axis
                ylabel = "Degrees" if component != "Thermal total energy" else "Energy"

                # Print the start of the analysis
                print(f"\n==========================================================\n{component}")
                print("==========================================================")

                # Loop over the 6 errors
                for num, errors in enumerate(errors_list):

                    # Print the case we are looking at
                    if num == 0:
                        print("\n----------------\nHeating Case")
                        print("----------------\n")
                    elif num == 3:
                        print("\n----------------\nCooling Case")
                        print("----------------\n")

                    print(prints[num])
                    # _plot_helpers(xlabel="Prediction time step", ylabel=ylabel, **kwargs)
                    # plt.boxplot(errors, notch=True, whis=[5, 95], showfliers=True)
                    # _save_or_show(**kwargs)

                    # Loop over the errors only keep the positive ones (negative errors mean that the model
                    # was physically sound) and count along the way the good predictions
                    count_good_predictions = 0
                    total_predictions = 0

                    # Define the list of bins of errors for the boxplot and loop along the errors to prepare them
                    error_bins = []
                    for i in range(len(errors[component])):

                        # Count the number of time predictions we have
                        number = len(errors[component][i])
                        # Compute the true errors (i.e. the positive ones)
                        true_errors = [x for x in errors[component][i] if x > 0]

                        # Update the counters
                        count_good_predictions += (number - len(true_errors))
                        total_predictions += number

                        # Append the true errors for the boxplot
                        if len(true_errors) > 0:
                            error_bins.append(true_errors)
                        else:
                            error_bins.append([])

                    # Print the results
                    print(f"{count_good_predictions} out of {total_predictions} comparisons were physically sound,"
                          f"{count_good_predictions / total_predictions * 100:.2f}%")

                    print(f"The error magnitude of the physically inconsistent steps is plotted below:")

                    # Boxplot of the true errors
                    _plot_helpers(xlabel="Prediction hour", ylabel=ylabel, **kwargs)
                    plt.boxplot(error_bins, notch=True, whis=[5, 95], showfliers=True)
                    _save_or_show(**kwargs)

        # Plot all informations in one plot, separating the heating and cooling cases

        # Prepare the plot
        fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16, 25))

        # Loop over the components to analyze the errors
        for i, component in enumerate(self.components[:-1]):
            for num, error in enumerate(errors_):
                if num % 3 == 0:
                    errors = errors_list[int(num / 3)]
                    count_wrong_predictions = 0
                    total_predictions = 0
                    # Compute the error percentage
                    for j in range(len(errors[component])):
                        # Count the number of time predictions we have
                        number = len(errors[component][j])
                        # Update the counters
                        count_wrong_predictions += len([x for x in errors[component][j] if x > 0])
                        total_predictions += number

                # Define the column, label, color and style of the plot
                column = 0 if num < 9 else 1
                label = f"{labels[int(num / 3)]} ({count_wrong_predictions / total_predictions * 100:.0f}%)" if num % 3 == 0 else None
                color = colors[int(num / 3)] if num < 9 else colors[int(num / 3) - 3]
                style = styles[num % 3]

                # Plot the mdeian, 75%, 95% percentile
                toplot = [0 if np.isnan(x) else x for x in error[component]]
                axes[i, column].plot(toplot, label=label, color=color, linestyle=style)

            # Define labels and legends
            axes[i, 0].set_ylabel(f"Room {component[-3:]} ($^\circ$C)", size=22)
            axes[i, 0].tick_params(axis='y', which='major', labelsize=15)
            axes[i, 0].legend(prop={'size': 15})
            axes[i, 1].legend(prop={'size': 15})
        # Set title and x label
        axes[4, 0].set_xlabel("Prediction time step", size=20)
        axes[4, 1].set_xlabel("Prediction time step", size=20)
        axes[0, 0].set_title("Cooling error", size=25)
        axes[4, 0].tick_params(axis='x', which='major', labelsize=15)
        axes[4, 1].tick_params(axis='x', which='major', labelsize=15)

        # Put it in the riht layout, save the figure and plot it
        plt.tight_layout()
        plt.savefig("Consistency.png", bbox_inches="tight")
        plt.show()

        # Plot separating all the 6 different cases
        # New labels
        labels = ["Cooling\nOpen - Pred", "Cooling\nPred - Closed", "Cooling\nOpen - Closed",
                  "Heating\nPred - Open", "Heating\nClosed - Pred", "Heating\nClosed - Open"]

        # Prepare the figure and loop over the components to analyze them
        fig, axes = plt.subplots(nrows=5, ncols=6, sharex=True, sharey=True, figsize=(36, 25))
        for i, component in enumerate(self.components[:-1]):
            # Recall the total wrong predictions, predictions and maximum error for each room
            wrong_predictions = 0
            predictions = 0
            maxima = []
            for num, error in enumerate(errors_):
                if num % 3 == 0:
                    errors = errors_list[int(num / 3)]
                    count_wrong_predictions = 0
                    total_predictions = 0
                    # Compute the error percentages
                    for j in range(len(errors[component])):
                        # Count the number of time predictions we have
                        number = len(errors[component][j])
                        # Update the counters
                        count_wrong_predictions += len([x for x in errors[component][j] if x > 0])
                        total_predictions += number

                    wrong_predictions += count_wrong_predictions
                    predictions += total_predictions

                if num % 3 == 2:
                    maxima.append(np.nanmax(error[component]))

                # Define the column label, color and line style
                column = int(num / 3)
                label = f"{count_wrong_predictions / total_predictions * 100:.0f}%" if num % 3 == 0 else None
                color = colors[int(num / 3)] if num < 9 else colors[int(num / 3) - 3]
                style = styles[num % 3]

                # Plot the info and legend
                toplot = [0 if np.isnan(x) else x for x in error[component]]
                axes[i, column].plot(toplot, label=label, color=color, linestyle=style)
                axes[i, column].legend(prop={'size': 20})

            # Set the labels
            axes[i, 0].set_ylabel(f"Room {component[-3:]} ($^\circ$C)", size=22)
            axes[i, 0].tick_params(axis='y', which='major', labelsize=15)

            # Print the statistics
            print(f"\n________________________________\n{component[7:]}:")
            print(f"  The model is physically inconsistent {wrong_predictions / predictions * 100:.0f}% of the time.")
            print(
                f"  When wrong, the model makes an error smaller than {np.nanmax(maxima):.2f} C with 95% probability.")

        # Set the labels and titles
        for j in range(6):
            axes[0, j].set_title(labels[j], size=25)
            axes[4, j].set_xlabel("Prediction time step", size=20)
            axes[4, j].tick_params(axis='x', which='major', labelsize=15)

        # Define the right layout, save the figure and plot it
        plt.tight_layout()
        plt.savefig("ConsistencyBis.png", bbox_inches="tight")
        plt.show()
