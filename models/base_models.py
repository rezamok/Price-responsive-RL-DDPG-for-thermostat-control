"""
File containing the basic models descriptions, i.e. a general model class
"""

import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from data_preprocessing.dataset import DataSet

from util.util import get_room, flat_list, intersect_lists
from util.plots import _save_or_show, _plot_helpers, plot_time_series
from models.helpers import name_factory

from parameters import  ROOMS
from models.parameters import model_kwargs

try:
    from torch.utils.tensorboard import SummaryWriter
    print("\nUsage of the TensorBoard is on!\n")
    USE_WRITER = True
except ImportError:
    USE_WRITER = False
    print("\nUsage of the TensorBoard is off!\n")


class BaseModel:
    """
    Base model class with common attributes and functions all models should have
    """

    def __init__(self, dataset: DataSet, data: pd.DataFrame,
                 differences: pd.DataFrame, model_kwargs: dict = model_kwargs):
        """
        Model initialization, assigning attributes that each model should have
        Args:
            dataset:        DataSet linked to the model
            data:           The actual data the model is to be or was fitted/trained with
            differences:    DataFrame of differences to use if you want to predict differences instead
                              of exact values
            model_kwargs:   Various parameters of the model
        """

        # Fix the seeds for reproduction
        self._fix_seeds(seed=model_kwargs["seed"])

        # Define the main attributes
        self.name = model_kwargs["model_name"]
        self.data = data

        # Battery models are much different than other models and are handled separately
        if self.name != "LinearBatteryModel":

            self.dataset = dataset
            self.differences = differences

            # Create the name associated to the model
            self.save_name = name_factory(data=self.dataset,
                                          model_kwargs=model_kwargs)

            # Save the parameters
            self.unit = model_kwargs["unit"]
            self.not_differences_components = model_kwargs["not_differences_components"]
            self.batch_size = model_kwargs["batch_size"]
            self.n_epochs = model_kwargs["n_epochs"]
            self.verbose = model_kwargs["verbose"]
            self.predict_differences = model_kwargs["predict_differences"]
            self.save_checkpoints = model_kwargs["save_checkpoints"]
            self.n_autoregression = model_kwargs["n_autoregression"]
            self.components = model_kwargs["components"]
            self.learning_rate = model_kwargs["learning_rate"]

            # Useful informations to keep in memory
            self.train_losses = []
            self.valid_losses = []

            # Define the Tensorboard writer
            if USE_WRITER:
                self.writer = SummaryWriter("runs/" + self.name)

            # Define the device (for colab)
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print("GPU acceleration on!")

            else:
                self.device = "cpu"
            self.save_path = model_kwargs["save_path"]

    def _fix_seeds(self, seed: int = None):
        """
        Function fixing the seeds for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_predictable_indices_for(self, sensor: str, n_autoregression: int = 1, overshoot: int = 0,
                                    condition=None, data=None) -> list:
        """
        Function getting the indices in the data for which sufficient autoregression terms exist
        For example, if we want to predict the model at 9h, we need past data, e.g. from
        8h on to input in the model.

        This is needed because the models are not robust to missing data, so we can only use (and
        thus do predictions) on data time series which don't contain any missing value.

        Usually used iteratively through the wrapper function 'get_predictable_indices'.

        Args:
            sensor:             Column to analyze
            n_autoregression:   Autoregression length (how steps back to look)
            overshoot:          Overshoot parameter: make predictions for several timesteps in future
                                 and not only 1
            data:               Use this to pass a predefined DataFrame to compute the indices for. If None
                                 will just use the data of the model.
            condition:          Set to "valve" to add the condition that the valves had to be opened during
                                 the interval under consideration, i.e. some action was taken (usually None)

        Returns:
            The indices for which sufficient previous terms are not missing
        """

        # First get the values of the Series to use the numpy library later
        if self.predict_differences & (sensor in self.components):
            data = self.differences[sensor].values
        else:
            if data is not None:
                data = data[sensor].values
            else:
                data = self.data[sensor].values

        # Check if we want to activate the condition (from an earlier version)
        cond = False
        if condition == "valve":
            if "valve" in sensor:
                cond = True

        # Get where the values exist
        not_nans_indices = np.where(~pd.isnull(data))[0]

        # Get the locations of the transfer between existing and missing values
        jumps = np.concatenate([[True], np.diff(not_nans_indices) != 1, [True]])

        # Helper value: last index of the data
        last = len(data) - 1

        # Get all the beginnings of a series of existing values - add 0 if the first value exists as well
        beginnings = list(set(jumps[:-1] * not_nans_indices) - set([0, last]))
        if ~np.isnan(data[0]):
            beginnings.append(0)
        if (~np.isnan(data[-1])) & np.isnan(data[-2]):
            beginnings.append(last)
        beginnings.sort()

        # Get all the ends of the previous series - add the last index if the last value exists
        ends = list(set(jumps[1:] * not_nans_indices) - set([0, last]))
        if ~np.isnan(data[last]):
            ends.append(last)
        if (~np.isnan(data[0])) & np.isnan(data[1]):
            ends.append(0)
        ends.sort()

        # We should have the same number of series beginning and ending
        assert len(beginnings) == len(ends), "Something went wrong"

        # Iterate through the streaks of indices with complete values
        indices = []
        for i in range(len(ends)):
            # If they are long enough: store the indices
            if ends[i] - beginnings[i] >= n_autoregression + overshoot:

                # If the valves condition applies, check that the valves were indeed opened at least once
                if cond:
                    for j in range(beginnings[i] + n_autoregression, ends[i] + 1 - overshoot):
                        if np.mean(data[j - n_autoregression: j + overshoot + 1]) > 0.1:
                            indices.append([j])

                # Otherwise just store the indices
                else:
                    indices.append(np.arange(beginnings[i] + n_autoregression, ends[i] + 1 - overshoot))

        # Return the indices, flattening the list of list created above
        return flat_list(indices)

    def get_predictable_indices(self, n_autoregression: int = 1, overshoot: int = 0, condition=None,
                                verbose: int = 0, case=None) -> list:
        """
        Function to get all the indices of a DataFrame for which sufficient autoregressive
        terms exist - by iterating over all the sensors and making returning the indices
        for which the past values are not missing.
        An overshoot parameter can also be added if we want to ensure no missing values in the future
        as well (for example if the model will run for several timesteps)

        Note that this can take a few minutes, which is why we save it once computed and try
        to load it before computing it.

        This function iterates over the columns of the data and uses the function
        'get_predictable_indices_for' and merges the results.

        Args:
            n_autoregression:   Autogression length
            overshoot:          Overshoot parameter: make predictions for several timesteps in future
                                 not only 1
            verbose:            vebose
            condition:          Set to "valve" to add the condition that the valves had to be opened during
                                 the interval under consideration, i.e. some action was taken (usually None)
            case:               Parameter to tune the name under which to save the computed indices (used
                                 typically to differentiate between the 'heating' and 'cooling' indices of the
                                 smae model)

        Returns:
            All the indices which we can predict for
        """

        # Compute the full name under which the indices should be saved
        # Deprecated: trick to handle the fact that two models wih different names had the same indices
        if "Energy" in self.name:
            self.name.replace("Energy", "Temperature")
            name = f"{self.dataset.start_date[:8]}__{self.dataset.end_date[:8]}__{self.unit}_{self.name}"
            self.name.replace("Temperature", "Energy")
        else:
            name = f"{self.dataset.start_date[:8]}__{self.dataset.end_date[:8]}__{self.unit}_{self.name}"

        # Add parameters to the name to make it unique
        if self.predict_differences:
            name += "_differences"
        if overshoot > 0:
            name += f"_{overshoot}"
        if condition is not None:
            name += condition
        if case is not None:
            name += f"_{case}"
        name += f"_{self.n_autoregression}_indices.pt"

        # Build the full path to the model
        full_path = os.path.join(self.save_path, name)

        print("Trying to load the predictable indices, where the data has no missing values...")
        try:
            # Check the existence of the model
            assert os.path.exists(full_path), f"The file {full_path} doesn't exist."

            # Load the checkpoint
            checkpoint = torch.load(full_path)

            # Put it into the model
            indices = checkpoint['indices']

            print("Found!")

        except AssertionError:

            print("Nothing found, building the indices - This could take a few minutes...")

            # Compute the indices working for the first sensor - check if it is autoregressive or
            # not as it changes the condition
            indices = self.get_predictable_indices_for(sensor=self.data.columns[0],
                                                       n_autoregression=n_autoregression,
                                                       overshoot=overshoot,
                                                       condition=condition,
                                                       data=self.data)

            # iterate over the rest of the sensors, check which indices are valid and intersect it with
            # ones working with the other sensors
            for sensor in self.data.columns[1:]:
                new_indices = self.get_predictable_indices_for(sensor=sensor,
                                                               n_autoregression=n_autoregression,
                                                               overshoot=overshoot,
                                                               condition=condition,
                                                               data=self.data)

                # Keep only the indices working for every sensor
                indices = intersect_lists(indices, new_indices)
                print(f"Number of indices working so far: {len(indices)}")

            # Save the built list to be able to load it later and avoid the computation
            torch.save({'indices': indices}, full_path)

        # Print information
        if verbose > 0:
            print(f"{len(indices)} indices long enough found!")

        # Return the indices
        return indices

    def save_torch_model(self, name: str, save_path: str = None):
        """
        Function to save a PyTorch model: Save the state of all parameters, as well as the one of the
        optimizer. We also recall the losses for ease of analysis.

        Args:
            name:       Name of the model
            save_path:  Where to save the model

        Returns
            Nothing, everything is done in place and stored in the parameters
        """

        if save_path is None:
            save_path = self.save_path

        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'valid_losses': self.valid_losses,
                    'train_sequences': self.train_sequences,
                    'validation_sequences': self.validation_sequences,
                    'test_sequences': self.test_sequences},
                   os.path.join(save_path, name + '.pt'))

    def load_torch_model(self, name: str, save_path: str = None):
        """
        Function to load a PyTorch model: reloading all that was saved

        Args:
            name:       Name of the model
            save_path:  Where to save the model

        Returns:
            Nothing, everything is done in place and stored in the parameters.
        """

        if save_path is None:
            save_path = self.save_path

        # Build the full path to the model and check its existence
        full_path = os.path.join(save_path, name + '.pt')
        assert os.path.exists(full_path), f"The file {full_path} doesn't exist."

        # Load the checkpoint
        checkpoint = torch.load(full_path, map_location=lambda storage, loc: storage)

        # Put it into the model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.valid_losses = checkpoint['valid_losses']

        # Check if we can also load sequences (to ensure that between 2 training sessions the training,
        # validation and testing data are not mixed)
        try:
            self.train_sequences = checkpoint['train_sequences']
            self.validation_sequences = checkpoint['validation_sequences']
            self.test_sequences = checkpoint['test_sequences']
        except KeyError:
            pass

    def load_model(self):
        """
        General function trying to load an existing model, using the 'load_torch_model' function.
        Returns nothing, everything is done in place and stored in the parameters.
        """

        print("\nTrying to load a trained model...")
        try:
            # Small deprecated tricks that were needed to change the name of the models to load
            if self.save_name[-5:] == "_Full":
                save_name = self.save_name[:-5]
            elif self.save_name[-17:] == "_Full_differences":
                save_name = self.save_name[:-17] + "_differences"
            else:
                save_name = self.save_name

            # Try to load the model
            self.load_torch_model(name=save_name)

            # Print the current status of the found model
            print(f"Found!\nThe model has been fitted for {len(self.train_losses)} epochs already, "
                  f"with loss {self.valid_losses[-1]: .5f}.")

            # Plot the losses if wanted
            if model_kwargs['show_plots']:
                self.plot_losses()

        # Otherwise, keep an empty model, return nothing
        except AssertionError:
            print(f"No existing model was found.")

    def plot_losses(self, **kwargs):
        """
        Function to plot the losses of the model
        """

        # Build the plot with the helpers: in general, the training loss can be high in the first
        # epochs, so let's customize the y axis limits to scale it right
        ylim_sup = 3 * self.valid_losses[-1]
        _plot_helpers(title="Losses", ylabel="Loss", xlabel="Epoch", ylim=(0, ylim_sup), **kwargs)

        # Plot the training and validation losses
        plt.plot(self.train_losses, label="Training")
        plt.plot(self.valid_losses, label="Validation")

        # Close and/or save the plot
        _save_or_show(legend=True, **kwargs)

    def fit(self, n_epochs: int = None, show_plot: bool = True, save_path: str = None) -> None:
        """
        General function fitting a model for several epochs, training and evaluating it on the data.

        Args:
            n_epochs:         Number of epochs to fit the model, if None this takes the default number
                                defined by the parameters
            show_plot:        Flag to set to False if you don't want to have the plot of losses shown
            save_path:        Path to save the data. If none uses the one stored as parameter

        Returns:
            Nothing
        """

        # If no special number of epochs is given, take the default one
        if n_epochs is None:
            n_epochs = self.n_epochs

        # If no special save pat is given, take the default one
        if save_path is None:
            save_path = self.save_path

        # Prepare the folder to save checkpoints
        if self.save_checkpoints:
            try:
                os.mkdir(os.path.join(save_path, self.save_name + "_Checkpoints"))
            except OSError:
                pass

        # Define a high starting loss
        best_loss = 1000

        # Assess the number of epochs the model was already trained on
        trained_epochs = len(self.train_losses)
        # Iterate over the epoch
        for epoch in range(trained_epochs, trained_epochs + n_epochs):

            # Start the training, define a list to retain the training losses along the way
            print("\nTraining starts!")
            self.model.train()
            train_losses = []
            
            self.adjust_learning_rate(epoch=epoch)

            # Create training batches and run through them
            for num_batch, batch_indices in enumerate(self.batch_iterator(iterator_type="train")):

                # Compute the loss of the batch and store it
                loss = self.compute_loss(batch_indices)

                # Compute the gradients and take one step using the optimizer
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_losses.append(float(loss))

                # Regularly print the current state of things
                if (self.verbose > 0) & (num_batch % 5 == 4):
                    print(f"Loss batch {num_batch + 1}: {float(loss): .5f}")

            # Compute the average loss of the training epoch and print it
            train_loss = sum(train_losses) / len(train_losses)
            print(f"Average training loss after {epoch + 1} epochs: {train_loss}")
            self.train_losses.append(train_loss)

            # Start the validation, again defining a list to recall the losses
            print("\nValidation starts!")
            self.model.eval()
            valid_losses = []

            # Create validation batches and run through them. Note that we use larger batches
            # to accelerate it a bit, and there is no need to shuffle the indices
            for num_batch, batch_indices in enumerate(self.batch_iterator(iterator_type="validation",
                                                                          batch_size=2 * self.batch_size,
                                                                          shuffle=False)):

                # compute the loss, in the torch.no_grad etting: we don't need the model to
                # compute and use gradients here, we are not training
                with torch.no_grad():
                    loss = self.compute_loss(batch_indices)
                    valid_losses.append(float(loss))

                    # Regularly print the current state of things
                    if (self.verbose > 0) & (num_batch % 2 == 1):
                        print(f"Loss batch {num_batch + 1}: {float(loss): .5f}")

            # Compute the average validation loss of the epoch and print it
            valid_loss = sum(valid_losses) / len(valid_losses)
            print(f"Average validation loss after {epoch + 1} epochs: {valid_loss}")
            self.valid_losses.append(valid_loss)

            # Save a checkpoint
            if self.save_checkpoints:
                self.save_torch_model(name=self.save_name + f"_epoch_{epoch}",
                                      save_path=os.path.join(save_path, self.save_name + "_Checkpoints"))
                if valid_loss < best_loss:
                    self.save_torch_model(name=self.save_name)
                    best_loss = valid_loss

            if USE_WRITER:
                self.writer.add_scalars("Losses", {"Training": train_loss, "Validation:": valid_loss}, epoch)

        if self.save_checkpoints:
            best_epoch = np.argmin([x for x in self.valid_losses])
            print(f"\nThe best model was obtained at epoch {best_epoch + 1}")
            #self.load_torch_model(name=self.save_name + f"_epoch_{best_epoch}",
             #                     save_path=os.path.join(save_path, self.save_name + "_Checkpoints"))
        # Save the final model
        self.save_torch_model(name=self.save_name)

        # Show plot
        if show_plot:
            self.plot_losses()
            
    def adjust_learning_rate(self, epoch):
        
        lr = self.learning_rate * (0.66 ** (epoch//15))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def disaggregate_rooms_energy(self, data=None, plot=True):
        """
        Function to disaggregate the total energy consumption into the individual rooms. For now
        only implemented for UMAR.
        It disaggregates the observed energy consumption proportionally to the flows through each valves,
        which are computed based on the mass flow designs.

        Args:
            data:   DataFrame containing at least the heating and cooling energy and all the valves information
            plot:   Flag to set to False if the plots are not wanted

        Returns:
            A DataFrame with the energy consumption of each room
        """

        # Take the entire model data if None is given
        if data is None:
            data = self.data

        # Copy the data to avoid problems
        df = data.copy()

        # Given and known mass flow design of each valve
        flows = {"Thermal valve 272": 217,
                 "Thermal valve 273": 217 * 3,  # there are 3 valves
                 "Thermal valve 274": 217,
                 "Thermal valve 275": 70,
                 "Thermal valve 276": 70}

        # Scale the data back
        if self.dataset.is_normalized:
            df = self.dataset.inverse_normalize(df)
        elif self.dataset.is_standardized:
            df = self.dataset.inverse_standardize(df)

        # Compute the total energy consumed
        total_energy = df["Thermal total energy"]

        # Get the valves columns
        valves = [x for x in data.columns if ("valve" in x)]

        # Use the flow of each valve and their opening to compute how much flows through each of them
        valves_flows = df[valves].multiply(flows)

        # Compute the total flow observed and plot it (scaled) against the energy measurement to check
        # visually if the decomposition makes sense
        total_flows = valves_flows.sum(axis=1)
        if plot:
            plot_time_series([pd.Series(total_flows, name="Total flow measurements") / 350,
                              pd.Series(total_energy, name="Total energy measured")])

        # Compute the proportion of the total flow (i.e. energy) of each room, and put the missing values to 0
        proportions = valves_flows.divide(total_flows, axis=0)
        proportions = proportions.fillna(0)

        # Compute the energy of each room
        room_energies = proportions.multiply(total_energy, axis=0)
        room_energies.rename(columns={f"Thermal valve {x}": f"Energy room {x}" for x in ROOMS}, inplace=True)

        # Plot the energies and return them
        if plot:
            plot_time_series(room_energies)
        return room_energies

##########################################
### Deprecated old version to keep
##########################################

import torch.nn.functional as F

class BaseModel_old:

    def __init__(self, dataset: DataSet, data: pd.DataFrame,
                 differences: pd.DataFrame, model_kwargs: dict = model_kwargs):
        """
        Model initialization, assigning attributes that each model should have
        Args:
            model_name:     Name of the model
            dataset:        DataSet linked to the model
            data:           The actual data the model was fitted/trained with
            differences:    DataFrame of differences to use if you ant to predict differences instead
                              of exact values
            model_kwargs:   Various parameters of the model
        """

        # Fix the seeds for reproduction
        self._fix_seeds(seed=model_kwargs["seed"])

        # Define the main attributes
        self.name = model_kwargs["model_name"]
        self.data = data

        if self.name != "LinearBatteryModel":

            self.dataset = dataset
            self.differences = differences

            # Create the name associated to the model
            self.save_name = name_factory(data=self.dataset,
                                          model_kwargs=model_kwargs)

            # Save the parameters
            self.unit = model_kwargs["unit"]
            self.components = model_kwargs["components"]
            self.not_differences_components = model_kwargs["not_differences_components"]
            self.batch_size = model_kwargs["batch_size"]
            self.n_epochs = model_kwargs["n_epochs"]
            self.verbose = model_kwargs["verbose"]
            self.predict_differences = model_kwargs["predict_differences"]
            self.save_checkpoints = model_kwargs["save_checkpoints"]
            self.n_autoregression = model_kwargs["n_autoregression"]

            # Useful informations to keep in memory
            self.train_losses = []
            self.valid_losses = []

            # Define the Tensorboard writer
            if USE_WRITER:
                self.writer = SummaryWriter("runs/" + self.name)

            # Define the device (for colab)
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print("GPU acceleration on!")

                # We also need to overwrite the save path to make it work with colab
                #self.save_path = "/content/drive/My Drive/DRL/saves/Models"
            else:
                self.device = "cpu"
            self.save_path = model_kwargs["save_path"]

            # Define the losses
            self._build_losses()

    def _fix_seeds(self, seed: int = None):
        """
        Function fixing the seeds for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _build_losses(self):
        """
        Little function creating one PyTorch loss for each component, currently a MSE loss
        """

        # Define a MSE loss for each component
        self.losses = {}
        for component in self.components:
            self.losses[component] = F.mse_loss

    def _build_components_inputs_dict(self):
        """
        Function to define which inputs impact which component. For example, the weather impacts
        the room temperature, the occupancy of a specific room impacts it, and so on.
        A dictionary is built where a list of sensor influencing each component is created.
        """

        # Define the dictionary and iterate over the components
        self.components_inputs_dict = {}

        print(self.data.columns)

        for component in self.components:

            # this takes both DFAB and UMAR cases, either the total energy, heating energy or cooling energy
            if "energy" in component:
                # In DFAB we have outlet temperature information, the rest are in both datasets
                sensors = [x for x in self.data.columns if ("valve" in x) | ("inlet" in x) | ("outlet" in x)]

            else:
                # All components are influenced by the weather, the time, the electricity consumption
                # and the total thermal energy consumed
                sensors = [x for x in self.data.columns if (("Time" in x) & ("month" not in x)) | ("Weather" in x) | ("outlet" in x)]
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
                    sensors += [x for x in self.data.columns if (room in x) & (x != component)]

                if self.unit == "UMAR":
                    sensors.append("Thermal inlet temperature")

            # Save the sensors in the dictionary
            self.components_inputs_dict[component] = sensors

    def get_predictable_indices_for(self, sensor: str, n_autoregression: int = 1, overshoot: int = 0,
                                    condition=None, data=None) -> list:
        """
        Function getting the indices for which sufficient autoregression terms exist
        For example, if we want to predict the model at 9h, we need past data, e.g. from
        8h on to input in the model

        Args:
            sensor:             Column to analyze
            n_autoregression:   Autoregression length
            overshoot:          Overshoot parameter: make predictions for several timesteps in future
                                    not only 1

        Returns:
            The indices for which sufficient previous terms are not missing
        """

        # First get the values of the Series to use numpy
        if self.predict_differences & (sensor in self.components):
            data = self.differences[sensor].values
        else:
            if data is not None:
                data = data[sensor].values
            else:
                data = self.data[sensor].values

        cond = False
        if condition == "valve":
            if "valve" in sensor:
                cond = True

        # Get where the values exist
        not_nans_indices = np.where(~pd.isnull(data))[0]

        # Get the locations of the transfer between existing and missing values
        jumps = np.concatenate([[True], np.diff(not_nans_indices) != 1, [True]])

        # Helper value: last index of the data
        last = len(data) - 1

        # Get all the beginnings of a series of existing values - add 0 if the first
        # value exists as well
        beginnings = list(set(jumps[:-1] * not_nans_indices) - set([0, last]))
        if ~np.isnan(data[0]):
            beginnings.append(0)
        if (~np.isnan(data[-1])) & np.isnan(data[-2]):
            beginnings.append(last)
        beginnings.sort()

        # Get all the ends of the previous series - add the last index if the last
        # value exists
        ends = list(set(jumps[1:] * not_nans_indices) - set([0, last]))
        if ~np.isnan(data[last]):
            ends.append(last)
        if (~np.isnan(data[0])) & np.isnan(data[1]):
            ends.append(0)
        ends.sort()

        # We should have the same number of series beginning and ending
        assert len(beginnings) == len(ends), "Something went wrong"

        # Iterate through the streaks of values
        indices = []
        for i in range(len(ends)):
            # If they are long enough: store the indices
            if ends[i] - beginnings[i] >= n_autoregression + overshoot:
                if cond:
                    for j in range(beginnings[i] + n_autoregression, ends[i] + 1 - overshoot):
                        if np.mean(data[j-n_autoregression: j+overshoot+1]) > 0.1:
                            indices.append([j])
                else:
                    indices.append(np.arange(beginnings[i] + n_autoregression, ends[i] + 1 - overshoot))

        # Return the indices, flattening the list
        return flat_list(indices)

    def get_predictable_indices(self, n_autoregression: int = 1, overshoot: int = 0, condition=None,
                                verbose: int = 0, data=None) -> list:
        """
        Function to get all the indices of a DataFrame for which sufficient autoregressive
        terms exist - by iterating over all the sensors and making returning the indices
        for which the past values are not missing.

        Note that this can take a few minutes, which is why we save it once computed and try
        to load it before computing it

        Args:
            n_autoregression:   Autogression length
            overshoot:          Overshoot parameter: make predictions for several timesteps in future
                                    not only 1
            verbose:            vebose

        Returns:
            All the indices which we can predict for
        """

        # Compute the full name (short version)
        name = f"{self.dataset.start_date[:8]}__{self.dataset.end_date[:8]}__{self.unit}_{self.name}"

        if self.predict_differences:
            name += "_differences"

        if overshoot > 0:
            name += f"_{overshoot}"

        if condition is not None:
            name += condition

        if data is not None:
            if len(data) > 0:
                if np.max(data.iloc[:, -1]) == 0.9:
                    name += "Heating"
                else:
                    name += "Cooling"

        name += f"_{self.n_autoregression}_indices.pt"

        # Build the full path to the model
        full_path = os.path.join(self.save_path, name)

        print("Trying to load the predictable indices, where the data has no missing values...")
        try:
            # Check the existence of the model
            assert os.path.exists(full_path), f"The file {full_path} doesn't exist."

            # Load the checkpoint
            checkpoint = torch.load(full_path)

            # Put it into the model
            indices = checkpoint['indices']

            print("Found!")

        except AssertionError:

            print("Nothing found, building the indices - This could take a few minutes...")

            # Compute the indices working for the first sensor - check if it is autoregressive or
            # not as it changes the condition

            indices = self.get_predictable_indices_for(sensor=self.data.columns[0],
                                                       n_autoregression=n_autoregression,
                                                       overshoot=overshoot,
                                                       condition=condition,
                                                       data=data)

            # iterate over the rest of the sensors, check which indices are valid and intersect it with
            # ones working with the other sensors
            for sensor in self.data.columns[1:]:
                new_indices = self.get_predictable_indices_for(sensor=sensor,
                                                               n_autoregression=n_autoregression,
                                                               overshoot=overshoot,
                                                               condition=condition,
                                                               data=data)

                # Keep only the indices working for every sensor
                indices = intersect_lists(indices, new_indices)

            # Save the built list to be able to load it later and avoid the computation
            torch.save({'indices': indices}, full_path)

        # Print information
        if verbose > 0:
            print(f"{len(indices)} sequences long enough found!")

        # Return the indices
        return indices

    def save_torch_model(self, name: str, save_path: str = None):
        """
        Function to save a PyTorch model: Save the state of all parameters, as well as the one of the
        optimizer. We also recall the losses for ease of analysis

        Args:
            name:       Name of the model
            save_path:  Where to save the model
        """

        if save_path is None:
            save_path = self.save_path

        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'valid_losses': self.valid_losses,
                    'train_sequences': self.train_sequences,
                    'validation_sequences': self.validation_sequences,
                    'test_sequences': self.test_sequences},
                   os.path.join(save_path, name + '.pt'))

    def load_torch_model(self, name: str, save_path: str = None):
        """
        Function to load a PyTorch model: reloading all that was saved

        Args:
            name:       Name of the model
            save_path:  Where to save the model

        Returns:
            The downloaded model
        """

        if save_path is None:
            save_path = self.save_path

        # Build the full path to the model and check its existence
        full_path = os.path.join(save_path, name + '.pt')
        assert os.path.exists(full_path), f"The file {full_path} doesn't exist."

        # Load the checkpoint
        checkpoint = torch.load(full_path, map_location=lambda storage, loc: storage)

        # Put it into the model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.valid_losses = checkpoint['valid_losses']
        try:
            self.train_sequences = checkpoint['train_sequences']
            self.validation_sequences = checkpoint['validation_sequences']
            self.test_sequences = checkpoint['test_sequences']
        except KeyError:
            pass

    def load_model(self, epoch=None):
        """
        General function trying to load an existing model
        """

        # Print the status and try to load the model
        print("\nTrying to load a trained model...")
        try:
            if self.save_name[-5:] == "_Full":
                save_name = self.save_name[:-5]
            elif self.save_name[-17:] == "_Full_differences":
                save_name = self.save_name[:-17]+"_differences"
            else:
                save_name = self.save_name
            self.load_torch_model(name=save_name)

            # Print the current status of the found model
            print(f"Found!\nThe model has been fitted for {len(self.train_losses)} epochs already, "
                  f"with loss {self.valid_losses[-1]: .5f}.")

            # Plot the losses if wanted
            if model_kwargs['show_plots']:
                self.plot_losses()

        # Otherwise, keep an empty model, return nothing
        except AssertionError:
            print(f"No existing model was found.")

    def plot_losses(self, **kwargs):
        """
        Function to plot the losses of the model
        """

        # Build the plot with the helpers: in general, the training loss can be high in the first
        # epochs, so let's customize the y axis limits to scale it right
        ylim_sup = 3 * self.valid_losses[-1]
        _plot_helpers(title="Losses", ylabel="Loss", xlabel="Epoch", ylim=(0, ylim_sup), **kwargs)

        # Plot the training and validation losses
        plt.plot(self.train_losses, label="Training")
        plt.plot(self.valid_losses, label="Validation")

        # Close and/or save the plot
        _save_or_show(legend=True, **kwargs)

    def fit(self, n_epochs: int = None, show_plot: bool = True, save_path: str = None) -> None:
        """
        General function fitting a model for several epochs, training and evaluating it on the data.

        Args:
            n_epochs:         Number of epochs to fit the model, if None this takes the default number
                                defined by the parameters
            show_plot:        Flag to set to False if you don't want to have the plot of losses shown
        """

        # If no special number of epochs is given, take the default one
        if n_epochs is None:
            n_epochs = self.n_epochs

        if self.save_checkpoints:
            if save_path is None:
                save_path = self.save_path
            try:
                os.mkdir(os.path.join(save_path, self.save_name+"_Checkpoints"))
            except OSError:
                pass

        best_loss = 1000

        # Assess the number of epochs the model was already trained on
        trained_epochs = len(self.train_losses)
        # Iterate over the epoch
        for epoch in range(trained_epochs, trained_epochs+n_epochs):

            # Start the training, define a list to retain the training losses along the way
            print("\nTraining starts!")
            self.model.train()
            train_losses = []

            # Create training batches and run through them
            for num_batch, batch_indices in enumerate(self.batch_iterator(iterator_type="train")):

                # Compute the loss of the batch and store it
                loss = self.compute_loss(batch_indices)
                train_losses.append(loss)

                # Regularly print the current state of things
                if (self.verbose > 0) & (num_batch % 10 == 9):
                    print(f"Loss batch {num_batch + 1}: {loss: .5f}")

                # Compute the gradients and take one step using the optimizer
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Compute the average loss of the training epoch and print it
            train_loss = sum(train_losses) / len(train_losses)
            print(f"Average training loss after {epoch+1} epochs: {train_loss}")
            self.train_losses.append(train_loss)

            # Start the validation, again defining a list to recall the losses
            print("\nValidation starts!")
            self.model.eval()
            valid_losses = []

            # Create validation batches and run through them. Note that we use larger batches
            # to accelerate it a bit, and there is no need to shuffle the indices
            for num_batch, batch_indices in enumerate(self.batch_iterator(iterator_type="validation",
                                                                          batch_size=2*self.batch_size,
                                                                          shuffle=False)):

                # compute the loss, in the torch.no_grad etting: we don't need the model to
                # compute and use gradients here, we are not training
                with torch.no_grad():
                    loss = self.compute_loss(batch_indices)
                    valid_losses.append(loss)

                    # Regularly print the current state of things
                    if (self.verbose > 0) & (num_batch % 5 == 4):
                        print(f"Loss batch {num_batch + 1}: {loss: .5f}")

            # Compute the average validation loss of the epoch and print it
            valid_loss = sum(valid_losses)/len(valid_losses)
            print(f"Average validation loss after {epoch+1} epochs: {valid_loss}")
            self.valid_losses.append(valid_loss)

            # Save a checkpoint
            if self.save_checkpoints:
                self.save_torch_model(name=self.save_name + f"_epoch_{epoch}",
                                      save_path=os.path.join(save_path, self.save_name + "_Checkpoints"))
                if valid_loss < best_loss:
                    self.save_torch_model(name=self.save_name)
                    best_loss = valid_loss

            if USE_WRITER:
                self.writer.add_scalars("Losses", {"Training": train_loss, "Validation:": valid_loss}, epoch)

        if self.save_checkpoints:
            best_epoch = np.argmin([x.numpy() for x in self.valid_losses])
            print(f"\nThe best model was obtained at epoch {best_epoch+1}")
            self.load_torch_model(name=self.save_name + f"_epoch_{best_epoch}",
                                  save_path=os.path.join(save_path, self.save_name + "_Checkpoints"))
        # Save the final model
        self.save_torch_model(name=self.save_name)

        # Show plot
        if show_plot:
            self.plot_losses()

    def disaggregate_rooms_energy(self, data=None, plot=True):
        """
        Function to disaggregate the total energy consumption into the individual rooms. For now
        only implemented for UMAR.
        It disaggregates the observed energy consumption proportionally to the flows through each valves,
        which are computed based on the mass flow designs.

        Args:
            data:   DataFrame containing at least the heating and cooling energy and all the valves information
            plot:   Flag to set to False if the plots are not wanted

        Returns:
            A DataFrame with the energy consumption of each room
        """

        # Take the entire model data if None is given
        if data is None:
            data = self.data

        # Copy the data to avoid problems
        df = data.copy()

        # Given and known mass flow design of each valve
        flows = {"Thermal valve 272": 217,
                 "Thermal valve 273": 217 * 3, # there are 3 valves
                 "Thermal valve 274": 217,
                 "Thermal valve 275": 70,
                 "Thermal valve 276": 70}

        # Scale the data back
        if self.dataset.is_normalized:
            df = self.dataset.inverse_normalize(df)
        elif self.dataset.is_standardized:
            df = self.dataset.inverse_standardize(df)

        # Compute the total energy consumed
        total_energy = df["Thermal total energy"]

        # Get the valves columns
        valves = [x for x in data.columns if ("valve" in x)]

        # Use the flow of each valve and their opening to compute how much flows through each of them
        valves_flows = df[valves].multiply(flows)

        # Compute the total flow observed and plot it (scaled) against the energy measurement to check
        # visually if the decomposition makes sense
        total_flows = valves_flows.sum(axis=1)
        if plot:
            plot_time_series([pd.Series(total_flows, name="Total flow measurements") / 350,
                              pd.Series(total_energy, name="Total energy measured")])

        # Compute the proportion of the total flow (i.e. energy) of each room, and put the missing values to 0
        proportions = valves_flows.divide(total_flows, axis=0)
        proportions = proportions.fillna(0)

        # Compute the energy of each room
        room_energies = proportions.multiply(total_energy, axis=0)
        room_energies.rename(columns={f"Thermal valve {x}": f"Energy room {x}" for x in ROOMS}, inplace=True)

        # Plot the energies and return them
        if plot:
            plot_time_series(room_energies)
        return room_energies
