"""
File containing the base class of models, with general functions
"""

import os
import pandas as pd
import math
import time

import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from data.util import prepare_data

from util.util import model_save_name_factory, format_elapsed_time
from util.plots import _plot_helpers, _save_or_show


class BaseModel:
    """
    Base model class with common attributes and functions all models should have
    """

    def __init__(self, data_kwargs: dict, model_kwargs: dict):
        """
        Model initialization, assigning attributes that each model should have.

        Args:
            data_kwargs:    Various parameters of the data, see 'parameters.py'
            model_kwargs:   Various parameters of the model, see 'parameters.py'
        """

        # Define the main attributes
        self.name = model_kwargs["name"]
        self.rooms = model_kwargs['room_models']
        self.model_kwargs = model_kwargs

        # Create the name associated to the model
        self.save_name = model_save_name_factory(data_kwargs=data_kwargs, model_kwargs=model_kwargs)

        if not os.path.isdir(self.save_name):
            os.mkdir(self.save_name)

        # Fix the seeds for reproduction
        self._fix_seeds(seed=model_kwargs["seed"])

    def _fix_seeds(self, seed: int = None):
        """
        Function fixing the seeds for reproducibility.

        Args:
            seed:   Seed to fix everything
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class TorchModel(BaseModel):
    """
    Class of models using PyTorch
    """

    def __init__(self, data_kwargs: dict, model_kwargs: dict, Y_columns: list, X_columns: list = None):
        """
        Initialize a model.

        Args:
            data_kwargs:    Parameters of the data, see 'parameters.py'
            model_kwargs:   Parameters of the models, see 'parameters.py'
            Y_columns:      Name of the columns that are to be predicted
            X_columns:      Sensors (columns) of the input data
        """

        super().__init__(data_kwargs=data_kwargs, model_kwargs=model_kwargs)

        self.batch_size = model_kwargs["batch_size"]
        self.shuffle = model_kwargs["shuffle"]
        self.n_epochs = model_kwargs["n_epochs"]
        self.verbose = model_kwargs["verbose"]
        self.learning_rate = model_kwargs["learning_rate"]
        self.decrease_learning_rate = model_kwargs["decrease_learning_rate"]
        self.predict_differences = model_kwargs["predict_differences"]
        self.heating = model_kwargs["heating"]
        self.cooling = model_kwargs["cooling"]
        self.warm_start_length = model_kwargs["warm_start_length"]
        self.minimum_sequence_length = model_kwargs["minimum_sequence_length"]
        self.maximum_sequence_length = model_kwargs["maximum_sequence_length"]
        self.overlapping_distance = model_kwargs["overlapping_distance"]
        self.validation_percentage = model_kwargs["validation_percentage"]
        self.test_percentage = model_kwargs["test_percentage"]
        self.module = model_kwargs["module"]

        # Prepare the data
        self.dataset = prepare_data(data_kwargs=data_kwargs, predict_differences=self.predict_differences,
                                    Y_columns=Y_columns, X_columns=X_columns, verbose=self.verbose)

        self.model = None
        self.optimizer = None
        self.loss = None
        self.train_losses = []
        self.validation_losses = []
        self._validation_losses = []
        self.test_losses = []
        self.discount_factors_heating = []
        self.discount_factors_cooling = []
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.times = []
        self.heating_sequences, self.cooling_sequences = None, None
        self.train_sequences = None
        self.validation_sequences = None
        self.test_sequences = None

        # To use the GPU when available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("\nGPU acceleration on!")
        else:
            self.device = "cpu"
            self.save_path = model_kwargs["save_path"]

    @property
    def X(self):
        return self.dataset.X

    @property
    def Y(self):
        return self.dataset.Y

    @property
    def columns(self):
        return self.dataset.data.columns

    @property
    def differences_Y(self):
        return self.dataset.differences_Y

    def get_column(self, names):
        """
        Small helper function to get the indices of columns with specific names
        """
        if type(names) == str:
            names = [names]

        indices = []
        for i, column in enumerate(self.dataset.X_columns):
            for name in names:
                if name in column:
                    indices.append(i)

        # Return a number if there is only one index
        if len(indices) == 1:
            return indices[0]
        else:
            return indices

    def _create_sequences(self, X: pd.DataFrame = None, Y: pd.DataFrame = None, inplace: bool = False):
        """
        Function to create tuple designing the beginning and end of sequences of data we can predict.
        This is needed because PyTorch models don't work with NaN values, so we need to only path
        sequences of data that don't contain any.

        Args:
            X:          input data
            Y:          output data, i.e. labels
            inplace:    Flag whether to do it in place or not

        Returns:
            The created sequences if not inplace.
        """

        # Take the data of the current model if nothing is given
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        # Get the places of NaN values not supported by PyTorch models
        nans = list(set(np.where(np.isnan(X))[0]) | set(np.where(np.isnan(Y))[0]))

        # List of indices that present no nans
        indices = np.arange(len(X))
        not_nans_indices = np.delete(indices, nans)
        last = len(indices) - 1

        sequences = []

        if len(not_nans_indices) > 0:
            # Get the "jumps", i.e. where the the nan values appear
            jumps = np.concatenate([[True], np.diff(not_nans_indices) != 1, [True]])

            # Get the beginnings of all the sequences, correcting extreme values and adding 0 if needed
            beginnings = list(not_nans_indices[np.where(jumps[:-1])[0]])
            if 0 in beginnings:
                beginnings = beginnings[1:]
            if last in beginnings:
                beginnings = beginnings[:-1]
            if (0 in not_nans_indices) and (1 in not_nans_indices):
                beginnings = [0] + beginnings

            # Get the ends of all the sequences, correcting extreme values and adding the last value if needed
            ends = list(not_nans_indices[np.where(jumps[1:])[0]])
            if 0 in ends:
                ends = ends[1:]
            if last in ends:
                ends = ends[:-1]
            if (last in not_nans_indices) and (last - 1 in not_nans_indices):
                ends = ends + [last]

            # We should have the same number of series beginning and ending
            assert len(beginnings) == len(ends), "Something went wrong"

            # Bulk of the work: create starts and ends of sequences tuples
            for beginning, end in zip(beginnings, ends):
                # Add sequences from the start to the end, jumping with the wanted overlapping distance and ensuring
                # the required warm start length and minimum sequence length are respected
                sequences += [(beginning + self.overlapping_distance * x,
                               min(beginning + self.warm_start_length + self.maximum_sequence_length
                                   + self.overlapping_distance * x, end))
                    for x in range(math.ceil((end - beginning - self.warm_start_length
                                              - self.minimum_sequence_length) / self.overlapping_distance))]

        if inplace:
            self.sequences = sequences
        else:
            return sequences

    def get_sequences(self, X: pd.DataFrame = None, Y: pd.DataFrame = None) -> list:
        """
        Function to get tuple designing the beginning and end of sequences of data we can predict.
        This is needed because PyTorch models don't work with NaN values, so we need to only path
        sequences of data that don't contain any.

        If no sequences exist, it creates them.

        Args:
            X:          input data
            Y:          output data, i.e. labels

        Returns:
            All the sequences we can predict
        """

        # Create the corresponding name
        name = os.path.join(self.save_name, "sequences.pt")

        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        if self.verbose > 0:
            print("\nTrying to load the predictable sequences, where the data has no missing values...")

        try:
            # Check the existence of the model
            assert os.path.exists(name), f"The file {name} doesn't exist."
            # Load the checkpoint
            checkpoint = torch.load(name)
            # Put it into the model
            heating_sequences = checkpoint["heating_sequences"]
            cooling_sequences = checkpoint["cooling_sequences"]

            if self.verbose > 0:
                print("Found!")

        except AssertionError:
            if self.verbose > 0:
                print("Nothing found, building the sequences - This could take a few minutes...")

            # Create the sequences
            if self.heating:
                X_ = X.copy()
                X_[np.where(X_[:, self.get_column("Case")] < 0.5)[0]] = np.nan
                heating_sequences = self._create_sequences(X=X_, Y=Y)
            else:
                heating_sequences = []

            if self.cooling:
                X_ = X.copy()
                X_[np.where(X_[:, self.get_column("Case")] > 0.5)[0]] = np.nan
                cooling_sequences = self._create_sequences(X=X_, Y=Y)
            else:
                cooling_sequences = []

            # Save the built list to be able to load it later and avoid the computation
            torch.save({"heating_sequences": heating_sequences, "cooling_sequences": cooling_sequences}, name)

        if self.verbose > 0:
            print(f"Number of sequences for the model {self.name}: {len(heating_sequences)} heating sequences and " f"{len(cooling_sequences)} cooling sequences.")

        # Return the sequences
        return heating_sequences, cooling_sequences

    def train_test_validation_separation(self, validation_percentage: float = 0.2, test_percentage: float = 0.0) -> None:
        """
        Function to separate the data into training and testing parts. The trick here is that
        the data is not actually split - this function actually defines the sequences of
        data points that are in the training/testing part.

        Args:
            validation_percentage:  Percentage of the data to keep out of the training process
                                    for validation
            test_percentage:        Percentage of data to keep out of the training process for
                                    the testing

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

        # Prepare the lists
        self.train_sequences = []
        self.validation_sequences = []
        self.test_sequences = []

        if self.verbose > 0:
            print("Creating training, validation and testing data")

        for sequences in [self.heating_sequences, self.cooling_sequences]:
            if len(sequences) > 0:
                # Given the total number of sequences, define aproximate separations between training
                # validation and testing sets
                train_validation_sep = int((1 - test_percentage - validation_percentage) * len(sequences))
                validation_test_sep = int((1 - test_percentage) * len(sequences))

                # Little trick to ensure training, validation and test sequences are completely distinct
                while True:
                    if (sequences[train_validation_sep - 1][1] < sequences[train_validation_sep][0]) | (train_validation_sep == 1):
                        break
                    train_validation_sep -= 1
                if test_percentage > 0.:
                    while True:
                        if (sequences[validation_test_sep - 1][1] < sequences[validation_test_sep][0]) | (validation_test_sep == 1):
                            break
                        validation_test_sep -= 1

                # Prepare the lists
                self.train_sequences += sequences[:train_validation_sep]
                self.validation_sequences += sequences[train_validation_sep:validation_test_sep]
                self.test_sequences += sequences[validation_test_sep:]

    def fit(self, n_epochs: int = None, show_plot: bool = True) -> None:
        """
        General function fitting a model for several epochs, training and evaluating it on the data.

        Args:
            n_epochs:         Number of epochs to fit the model, if None this takes the default number
                                defined by the parameters
            show_plot:        Flag to set to False if you don't want to have the plot of losses shown

        Returns:
            Nothing
        """

        self.times.append(time.time())

        if self.verbose > 0:
            print("\nTraining starts!")

        # If no special number of epochs is given, take the default one
        if n_epochs is None:
            n_epochs = self.n_epochs

        # Define the best loss, taking the best existing one or a very high loss
        best_loss = np.min(self.validation_losses) if len(self.validation_losses) > 0 else np.inf

        # Assess the number of epochs the model was already trained on to get nice prints
        trained_epochs = len(self.train_losses)

        for epoch in range(trained_epochs, trained_epochs + n_epochs):

            if self.verbose > 0:
                print(f"\nTraining epoch {epoch + 1}...")

            # Start the training, define a list to retain the training losses along the way
            self.model.train()
            train_losses = []

            # Adjust the learning rate if wanted
            if self.decrease_learning_rate:
                self.adjust_learning_rate(epoch=epoch)

            # Create training batches and run through them, using the batch_iterator function, which has to be defined
            # independently for each subclass, as different types of data are handled differently
            for num_batch, batch_sequences in enumerate(self.batch_iterator(iterator_type="train")):

                # Compute the loss of the batch and store it
                loss = self.compute_loss(batch_sequences)

                # Compute the gradients and take one step using the optimizer
                loss.backward()
                #for p in self.model.named_parameters():
                #    if (p[1].grad is not None):
                #        print(p[0], ":", p[1].grad.norm())
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_losses.append(float(loss))

                # Regularly print the current state of things
                if (self.verbose > 1) & (num_batch % 5 == 4):
                    print(f"Loss batch {num_batch + 1}: {float(loss): .5f}")

            # Compute the average loss of the training epoch and print it
            train_loss = sum(train_losses) / len(train_losses)
            print(f"Average training loss after {epoch + 1} epochs: {train_loss}")
            self.train_losses.append(train_loss)

            # Start the validation, again defining a list to recall the losses
            print(f"Validation epoch {epoch + 1}...")
            validation_losses = []
            _validation_losses = []

            # Create validation batches and run through them. Note that we use larger batches
            # to accelerate it a bit, and there is no need to shuffle the indices
            for num_batch, batch_indices in enumerate(self.batch_iterator(iterator_type="validation", batch_size=2 * self.batch_size, shuffle=False)):

                # Compute the loss, in the torch.no_grad setting: we don't need the model to
                # compute and use gradients here, we are not training
                if 'PiNN' not in self.name:
                    self.model.eval()
                    with torch.no_grad():
                        loss = self.compute_loss(batch_indices)
                        validation_losses.append(float(loss))
                        # Regularly print the current state of things
                        if (self.verbose > 1) & (num_batch % 2 == 1):
                            print(f"Loss batch {num_batch + 1}: {float(loss): .5f}")

                else:
                    self.model.train()
                    loss = self.compute_loss(batch_indices)
                    validation_losses.append(float(loss))
                    # Regularly print the current state of things
                    if (self.verbose > 1) & (num_batch % 2 == 1):
                        print(f"Loss batch {num_batch + 1}: {float(loss): .5f}")
                    self.model.eval()
                    with torch.no_grad():
                        loss = self._compute_loss(batch_indices)
                        _validation_losses.append(float(loss))
                        # Regularly print the current state of things
                        if (self.verbose > 1) & (num_batch % 2 == 1):
                            print(f"Loss batch {num_batch + 1}: {float(loss): .5f}")

            # Compute the average validation loss of the epoch and print it
            validation_loss = sum(validation_losses) / len(validation_losses)
            self.validation_losses.append(validation_loss)
            if self.verbose > 0:
                print(f"Average validation loss after {epoch + 1} epochs: {validation_loss}")

            if 'PiNN' in self.name:
                _validation_loss = sum(_validation_losses) / len(_validation_losses)
                self._validation_losses.append(_validation_loss)
                if self.verbose > 0:
                    print(f"Average accuracy validation loss after {epoch + 1} epochs: {_validation_loss}")

            # Timing information
            self.times.append(time.time())
            if self.verbose > 0:
                print(f"Time elapsed for the epoch: {format_elapsed_time(self.times[-2], self.times[-1])}"
                      f" - for a total training time of {format_elapsed_time(self.times[0], self.times[-1])}")

            # Save parameters
            try:
                self.discount_factors_heating.append(float(self.model.discount_factor_heating))
                self.discount_factors_cooling.append(float(self.model.discount_factor_cooling))
            except AttributeError:
                pass
            if self.module == 'PCNNTestQuantiles':
                self.a.append([[float(x._parameters['log_weight']) for x in a] for a in self.model.a])
                self.b.append([[float(x._parameters['log_weight']) for x in b] for b in self.model.b])
                self.c.append([[float(x._parameters['log_weight']) for x in c] for c in self.model.c])
                self.d.append([[float(x._parameters['log_weight']) for x in d] for d in self.model.d])
            else:
                try:
                    self.a.append([float(x._parameters['log_weight']) for x in self.model.a])
                    self.b.append([float(x._parameters['log_weight']) for x in self.model.b])
                    self.c.append([float(x._parameters['log_weight']) for x in self.model.c])
                    self.d.append([float(x._parameters['log_weight']) for x in self.model.d])
                except TypeError:
                    self.a.append(float(self.model.a._parameters['weight']))
                    self.b.append(float(self.model.b._parameters['weight']))
                    try:
                        self.c.append(float(self.model.c._parameters['weight']))
                    except:
                        self.c.append([float(x._parameters['weight']) for x in self.model.c])
                    self.d.append(float(self.model.d._parameters['weight']))
                except AttributeError:
                    pass

            # Save last and possibly best model
            self.save(name_to_add="last", verbose=0)

            if validation_loss < best_loss:
                self.save(name_to_add="best", verbose=1)
                best_loss = validation_loss

        if self.verbose > 0:
            best_epoch = np.argmin([x for x in self.validation_losses])
            print(f"\nThe best model was obtained at epoch {best_epoch + 1} after training for " f"{trained_epochs + n_epochs} epochs")

        # Show plot
        if show_plot:
            self.plot_losses()

    def adjust_learning_rate(self, epoch: int) -> None:
        """
        Custom function to decrease the learning rate along the training

        Args:
            epoch:  Epoch of the training

        Returns:
            Nothing, modifies the optimizer in place
        """

        #lr = self.learning_rate * (1 / (np.sqrt(epoch/2 + 1)))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.997

    def batch_iterator(self, iterator_type: str = "train", batch_size: int = None, shuffle: bool = True) -> None:
        """
        Function to create batches of the data with the wanted size, either for training or testing. This
        needs to be implemented independently for each subclass, as it depends on how the data is handled.

        Args:
            iterator_type:  To know if this should handle training or testing data
            batch_size:     Size of the batches
            shuffle:        Flag to shuffle the data before the batches creation

        Returns:
            Nothing, yields the batches to be then used in an iterator
        """

        raise NotImplementedError

    def compute_loss(self, sequences: list):
        """
        Custom function to compute the loss of a batch of sequences.

        Args:
            sequences: The sequences in the batch

        Returns:
            The loss
        """

        raise NotImplementedError

    def save(self, name_to_add: str = None):
        """
        Function to save a PyTorch model: Save the state of all parameters, as well as the one of the
        optimizer. We also recall the losses for ease of analysis.

        Args:
            name_to_add:    Something to save a unique model

        Returns
            Nothing, everything is done in place and stored in the parameters
        """

        raise NotImplementedError

    def load(self, load_last: bool = False):
        """
        Function trying to load an existing model, by default the best one if it exists. But for training purposes,
        it is possible to load the last state of the model instead.

        Args:
            load_last:  Flag to set to True if the last model checkpoint is wanted, instead of the best one

        Returns:
             Nothing, everything is done in place and stored in the parameters.
        """

        raise NotImplementedError

    def plot_losses(self, **kwargs):
        """
        Function to plot the losses of the model
        """

        # Build the plot with the helpers: in general, the training loss can be high in the first
        # epochs, so let's customize the y axis limits to scale it right
        ylim_sup = 3 * self.validation_losses[-1]
        _plot_helpers(title=f"{self.name}_losses", ylabel="Loss", xlabel="Epoch", ylim=(0, ylim_sup), **kwargs)

        # Plot the training and validation losses
        plt.plot(np.arange(1, len(self.train_losses) + 1), self.train_losses, label="Training")
        plt.plot(np.arange(1, len(self.train_losses) + 1), self.validation_losses, label="Validation")

        # Close and/or save the plot
        _save_or_show(legend=True, save_name=f"Model_losses_{self.name}", **kwargs)
