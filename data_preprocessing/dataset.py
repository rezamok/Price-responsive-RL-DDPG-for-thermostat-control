"""
File of the basic DataSet class with methods to handle the data. It manages the downloading
and saving of the data locally according to the sensors defined in "data_from_NEST.py"

Some cleaning processes are also included (and some more in the preprocessing file)

At the end of the file, the global function to the data is given
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from typing import List
from rest.client import DataStruct
from util.util import  flat_list, construct_list_of, save_data, load_data
from util.plots import histogram

from data_preprocessing.preprocessing import preprocessing_pipeline

from parameters import ROOMS, UNIT

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2020-01-15"


class DataSet(object):
    """
    Main class to handle the data, merging data sets, standardizing or normalizing it
    """

    save_path = os.path.join("saves", "Data_preprocessed") if sys.platform == "win32" else \
        os.path.join("..", "saves", "Data_preprocessed")

    def __init__(self, data, name=None, names=None, controllable=None, controllable_NEST=None, informations=None,
                 unit=UNIT, interval: int = 15, save_data: bool = True, force_preprocess: bool = False,
                 is_standardized: bool = False, mean: pd.Series = pd.Series(dtype=np.float64),
                 std: pd.Series = pd.Series(dtype=np.float64), is_normalized: bool = False,
                 min_: pd.Series = pd.Series(dtype=np.float64),
                 max_: pd.Series = pd.Series(dtype=np.float64), verbose: int = 0):
        """
        Class intialization. Most arguments are usually initialized as empty and defined/used
        using some function below.

        Main arguments:
            data:           Either a DataStruct to load and preprocess the data from the
                            NEST database or an already preprocessed DataSet
            name:           The name to give the DataSet
            names:          Dictionary of names to give the columns of the DataFrame
            interval:       Interval to which smooth the data if the preprocessing is needed
            controllable:   list of indices or name of controllable columns/sensors

        Creates a DataSet with a DataFrame as "data" argument containing everything, as well as some
        other helping arguments
        """

        # Set the name of the dataset
        if name is not None:
            self.name = name
        elif type(data) == DataStruct:
            self.name = data.name
        else:
            raise NotImplementedError("Give a name to the data set")

        # Create or load the data
        # If a DataStruct is given
        if type(data) == DataStruct:

            # If we want to overwrite the data by preprocessing it with new parameters
            if force_preprocess:
                if verbose > 0:
                    print("Preprocessing the data...")
                self.data = preprocessing_pipeline(data=data,
                                                   name=self.name,
                                                   names=names,
                                                   interval=interval,
                                                   verbose=verbose)
                if save_data:
                    if verbose > 0:
                        print("Saving the data...")
                    self.save_dataframe(unit=unit,
                                        start_date=data.start_date,
                                        end_date=data.end_date)
                    if verbose > 0:
                        print("Data saved successfully!")

            else:
                # Try to load the preprocessed data
                try:
                    if verbose > 0:
                        print(f"Trying to load the {self.name} data...")
                    self.load_dataframe(unit=unit,
                                        start_date=data.start_date,
                                        end_date=data.end_date)
                    if verbose > 0:
                        print("Data downloaded sucessfully!")
                # If it fails, preprocess it
                except AssertionError:
                    if verbose > 0:
                        print("There is no preprocessed data.\nPreprocessing the data...")
                    self.data = preprocessing_pipeline(data=data,
                                                       name=self.name,
                                                       names=names,
                                                       interval=interval,
                                                       verbose=verbose)
                    if save_data:
                        if verbose > 0:
                            print("Saving the data...")
                        self.save_dataframe(unit=unit,
                                            start_date=data.start_date,
                                            end_date=data.end_date)
                        if verbose > 0:
                            print("Data saved successfully!")

        # Otherwise, if a preprocessed DataFrame is passed in argument,
        # copy it to avoid alterating the original DataFrame
        elif type(data) == pd.DataFrame:
            self.data = data.copy()

            # Save if wanted
            if save_data:
                if verbose > 0:
                    print("Saving the data...")
                self.save_dataframe(unit=unit,
                                    start_date=self.data.index[0].strftime("%Y-%m-%d"),
                                    end_date=self.data.index[-1].strftime("%Y-%m-%d"))
                if verbose > 0:
                    print("Data saved successfully!")

        # Otherwise not implemented
        else:
            raise ValueError(f"Data type {type(data)} is not supported")

        # Define the starting and ending date of the dataset
        self.start_date = self.data.index[0].strftime("%y-%m-%d %Hh%M")
        self.end_date = self.data.index[-1].strftime("%y-%m-%d %Hh%M")

        # Define the controllable column(s) as a list
        self.controllable = construct_list_of(controllable)

        # Define the actual controllable loads at NEST as a list
        self.controllable_NEST = construct_list_of(controllable_NEST)

        # Define the informative columns as a list
        self.informations = construct_list_of(informations)

        # The rest are measurements
        self.measurements = [sensor for sensor in self.data.columns \
                             if sensor not in self.controllable + self.controllable_NEST + self.informations]

        # Arguments of the standardization (empty until the dataset is explicitly standardized)
        self.is_standardized = is_standardized
        self.mean = mean
        self.std = std

        # Arguments of the normalization (empty until the dataset is explicitly normalized)
        self.is_normalized = is_normalized
        self.min_ = min_
        self.max_ = max_

        # To keep in memory
        self.interval = interval

    def _get_sensors_with(self, name):
        """
        Helper function to return the columns containing sensors with a particular name
        e.g. all sensors of the "Battery"
        """

        if type(name) == str:
            return [sensor for sensor in self.data.columns if name in sensor]
        elif type(name) == list:
            sensors = []
            for element in name:
                sensors.append([sensor for sensor in self.data.columns if element in sensor])
            return flat_list(sensors)
        else:
            raise ValueError(f"{type(name)} is not supported")

    def _get_measurements_with(self, name):
        """
        Helper function to return the columns containing measurements with a particular name
        e.g. all measurements of the "Battery"
        """

        if type(name) == str:
            return [sensor for sensor in self.measurements if name in sensor]
        elif type(name) == list:
            sensors = []
            for element in name:
                sensors.append([sensor for sensor in self.measurements if element in sensor])
            return flat_list(sensors)
        else:
            return self.measurements

    def _get_controllable_with(self, name):
        """
        Helper function to return the columns containing controllable sensors with a particular
        name e.g. all controllable loads of the "Battery"
        """

        if type(name) == str:
            return [sensor for sensor in self.controllable if name in sensor]
        elif type(name) == list:
            sensors = []
            for element in name:
                sensors.append([sensor for sensor in self.controllable if element in sensor])
            return flat_list(sensors)
        else:
            return self.controllable

    def _get_controllable_NEST_with(self, name):
        """
        Helper function to return the columns containing controllable sensors in NEST with
        a particular name, e.g. all controllable loads of the "Battery"
        """

        if type(name) == str:
            return [sensor for sensor in self.controllable_NEST if name in sensor]
        elif type(name) == list:
            sensors = []
            for element in name:
                sensors.append([sensor for sensor in self.controllable_NEST if element in sensor])
            return flat_list(sensors)
        else:
            return self.controllable_NEST

    def _get_informations_with(self, name):
        """
        Helper function to return the columns containinginformations with
        a particular name, e.g. all informations on the "Occupancy"
        """

        if type(name) == str:
            return [sensor for sensor in self.informations if name in sensor]
        elif type(name) == list:
            sensors = []
            for element in name:
                sensors.append([sensor for sensor in self.informations if element in sensor])
            return flat_list(sensors)
        else:
            return self.informations

    def get_sensors(self, name):
        """
        Helper function returning a slice of the data corresponding to the wanted sensors, e.g.
        all sensors corresponding to the "Battery".
        """

        return self.data[self._get_sensors_with(name)]

    def get_measurements(self, name=None):
        """
        Helper function returning a slice of the data corresponding to the wanted measurements,
        e.g. all measurements of the "Battery". If no name is provided, all measurements are
        returned
        """

        if name is not None:
            return self.data[self._get_measurements_with(name)]
        else:
            return self.data[self.measurements]

    def get_controllable(self, name=None):
        """
        Helper function returning a slice of the data corresponding to the wanted controllable
        sensors, e.g. all the controllable loads of the "Battery". If no name is provided,
        all controllable loads are returned
        """

        if name is not None:
            return self.data[self._get_controllable_with(name)]
        else:
            return self.data[self.controllable]

    def get_controllable_NEST(self, name=None):
        """
        Helper function returning a slice of the data corresponding to the wanted controllable
        sensors in NEST, e.g. all the controllable loads of the "Battery". If no name is provided,
        all controllable loads are returned
        """

        if name is not None:
            return self.data[self._get_controllable_NEST_with(name)]
        else:
            return self.data[self.controllable_NEST]

    def get_informations(self, name=None):
        """
        Helper function returning a slice of the data corresponding to the wanted informations,
        e.g. all the informations on the "Occupancy". If no name is provided,
        all informations are returned
        """

        if name is not None:
            return self.data[self._get_informations_with(name)]
        else:
            return self.data[self.informations]

    def get_controllable_and_measurements(self, name=None):
        """
        Helper function returning a slice of the data corresponding to the wanted controllable
        sensors and measurements, e.g. all the controllable loads and measurements of the "Battery".
        If no name is provided, all controllable loads and measurements are returned
        """

        if name is not None:
            return self.data[self._get_measurements_with(name) + self._get_controllable_with(name)]
        else:
            return self.data[self.measurements + self.controllable]

    def concatenate(self, other, name=None, inplace: bool = False, save_data: bool = False):
        """
        Function to concatenate, i.e. merge, two datasets, which means that
        the new sensors are appended as new columns, and the other arguments are transmitted
        or concatenated

        Args:
            name:       Name of the new DataSet
            other:      other DataSet
            inplace:    if true, the new DataSet will be saved in place of the current one,
                        otherwise a new DataSet is returned
            save_data:  Boolean to save the data

        Returns:
            A new DataSet containing the information of both original DataSets if inplace=False

        """

        # Several sanity checks that match the current implementation
        # Check that both datasets span the same dates
        if self.start_date != other.start_date:
            print(f"Warning: the DataSets have different starting dates {self.start_date} and {other.start_date}")
        if self.end_date != other.end_date:
            print(f"Warning: the DataSets have different ending dates {self.end_date} and {other.end_date}")
        if len(self.data) != len(other.data):
            print("Warning: the DataSets have different length")

        # Check the status of the datasets to merge, ensuring both are equal
        assert self.is_standardized == other.is_standardized, f"One dataset is standardized, not the other"
        assert self.is_normalized == other.is_normalized, f"One dataset is normalized, not the other"

        # Bulk of the work, append the columns of the second dataset to the first one
        data = self.data.copy().join(other.data.copy(), how='outer')

        # Merge or concatenate all the parameters
        # General arguments
        # Give it an automatic name if none is provided
        if name is None:
            name = self.name + "+" + other.name
        else:
            name = name

        # Update the controllable and measurement columns
        controllable = self.controllable + other.controllable
        controllable_NEST = self.controllable_NEST + other.controllable_NEST
        informations = self.informations + other.informations
        measurements = self.measurements + other.measurements

        # Standardization and normalization parameters
        mean = pd.concat([self.mean, other.mean])
        std = pd.concat([self.std, other.std])
        min_ = pd.concat([self.min_, other.min_])
        max_ = pd.concat([self.max_, other.max_])

        # If inplace is wanted, replace the arguments of the current dataset
        if inplace:
            self.data = data
            self.name = name
            self.controllable = controllable
            self.controllable_NEST = controllable_NEST
            self.informations = informations
            self.measurements = measurements
            self.mean = mean
            self.std = std
            self.min_ = min_
            self.max_ = max_
            self.start_date = self.data.index[0].strftime("%y-%m-%d %Hh%M")
            self.end_date = self.data.index[-1].strftime("%y-%m-%d %Hh%M")

        # Otherwise, return a new dataset
        else:
            return DataSet(data=data,
                           name=name,
                           controllable=controllable,
                           controllable_NEST=controllable_NEST,
                           informations=informations,
                           is_standardized=self.is_standardized,
                           mean=mean,
                           std=std,
                           is_normalized=self.is_normalized,
                           min_=min_,
                           max_=max_,
                           save_data=save_data)

    def save_dataframe(self, name=None, unit=UNIT, start_date=None, end_date=None,
                       save_path=save_path) -> None:
        """
        Function to save the dataframe after the preprocessing to avoid having to redo
        all the preprocessing each time

        Args:
            name:       Name of the DataFrame
            unit:       Corresponding unit of the data
            start_date: Starting date of the file to save
            end_date:   Ending date of the file to save
            save_path:  Where to save it
        """

        # Define the starting and ending date if not provided
        if start_date is None:
            start_date = self.start_date
            end_date = self.end_date

        if name is None:
            name = self.name

        # Bulk of the work
        save_data(data=self.data,
                  unit=unit,
                  start_date=start_date,
                  end_date=end_date,
                  name=name,
                  save_path=save_path)

    def load_dataframe(self, name=None, unit=UNIT, start_date=None, end_date=None,
                       save_path=save_path) -> None:
        """
        Function to load a preprocessed dataframe if it exists to avoid having to redo
        all the preprocessing each time

        Args:
            name:       Name of the DataFrame
            unit:       Corresponding unit of the data
            start_date: Starting date of the file to load
            end_date:   Ending date of the file to load
            save_path:  Where to load it
        """

        # Get the right starting and ending date
        if start_date is None:
            start_date = self.start_date
            end_date = self.end_date

        if name is None:
            name = self.name

        if torch.cuda.is_available():
            print("GPU acceleration on!")

            # We also need to overwrite the save path to make it work with colab
            #save_path = "/content/drive/My Drive/DRL/saves/Data_preprocessed"

        # Load the data and change the index to timestamps
        self.data = load_data(start_date=start_date,
                              end_date=end_date,
                              unit=unit,
                              name=name,
                              save_path=save_path)

    def data_analysis(self, name: str = None, bins: int = 1000, show: bool = True, verbose: int = 0) -> None:
        """
        Very small and basic function to analyze the missing values in the data

        Args:
            name:       Name the sensors to analyze should have (if None: all the sensors)
            bins:       Number of bins of the histogram plot
            show:       To show the plot
            verbose:    Verbose
        """

        # First get the data with the wanted sensors
        data = self.get_controllable_and_measurements(name=name)

        # General information
        if verbose > 0:
            if name is not None:
                print(f"\n{name} data analysis:")
            print(f"We have {len(data)} datapoints, {len(data.dropna(axis=0, how='any'))}"
                  f" of which have no missing value.")

        # Define some variables to capture the missing values location and number for each sensor
        nans_location = []
        nans_number = pd.Series(index=data.columns, dtype=int)

        # Iterate to retrieve the information from all sensors
        for sensor in data.columns:
            nans_location.append(data.index[np.where(pd.isnull(data[sensor]))[0]])
            nans_number[sensor] = len(np.where(pd.isnull(data[sensor]))[0])

        # Flatten the list of nans
        nans_location = flat_list(nans_location)

        if verbose > 1:
            # Plot if wanted
            if show:
                # Define the title
                if name is not None:
                    title = f"Missing {name} data analysis"
                else:
                    title = "Missing data analysis"

                histogram(nans_location,
                          bins=bins,
                          xdates=(data.index[-1] - data.index[0]).days,
                          ylabel="Number of missing values",
                          title=title)
            if verbose > 2:
                print("Number of missing values for each sensor")
                print(nans_number)

    def full_data_analysis(self, bins=1000, show=True, verbose: int = 0):
        """
        Small helper function to perform the data analysis in the DFAB case

        Args:
            bins:       Bins for the histogram
            show:       To plot the histogram or not
            verbose:    Verbose
        """

        # Global analysis
        self.data_analysis(bins=bins,
                           show=show,
                           verbose=1)

        # Let's analyze the battery, the HP and the thermal data and finally, let's take
        # a look at the weather and the thermal data together, as they are interdependent
        to_analyze = ['Battery', ['HP', 'Thermal', 'Weather']]
        for name in to_analyze:
            self.data_analysis(name=name,
                               bins=bins,
                               show=False,
                               verbose=verbose)

    def clean_data(self, bins=1000, show=True, threshold: int = 2, verbose: int = 0):
        """
        Function to clean the data.
        It currently remove timesteps where all the sensors have missing values and interpolates
        linearly missing streaks of "threshold" steps or less

        Args:
            threshold:  Maximum number of missing values in a row that can be interpolated
            bins:       Bins for the histogram
            show:       To plot the histogram or not
            verbose:    Verbose

        Returns:
            Nothing, the data is changed in place
        """

        # Delete useless time steps where all sensors are missing
        n_datapoints = len(self.data)
        self.data = self.data.dropna(axis=0, how='all')
        if verbose > 0:
            print(f"\nDropped {n_datapoints - len(self.data)} datapoints were all values were missing")

        # Interpolate sufficiently short streaks of missing values
        count = 0
        # This only makes sense for weather and temperature measurements
        for sensor in self._get_sensors_with(['Weather', 'Thermal temperature measurement']):

            # Get the location of the missing data and correct it if there is any
            nans_location = np.where(pd.isnull(self.data[sensor]))[0]
            if len(nans_location) > 0:

                # Transform the data in True-False, where True means a jump and False means
                # two missing values where consecutive
                #transformed_data = np.concatenate([[True], np.diff(nans_location) != 1, [True]])
                transformed_data = np.diff(nans_location)

                # Get the indices of the True values, i.e. the indices of the jumpy between
                # missing values streaks
                non_zero_indices = np.flatnonzero(transformed_data)

                # Compute the length of the missing streaks looking at the difference between
                # jumps - 2 jumps in a row means there was only one missing value in between
                missing_streak_length = np.diff(non_zero_indices)

                # Get the indices of the streaks we can interpolate, those below the threshold
                streaks_to_interpolate = np.where(missing_streak_length <= threshold)[0]

                # Interpolation
                for streak in streaks_to_interpolate:

                    # Get the values at the beginning and at the end of the missing streak
                    beginning = self.data[sensor][nans_location[non_zero_indices[streak]] - 1]
                    end = self.data[sensor][nans_location[non_zero_indices[streak]] + missing_streak_length[streak]]

                    # Fill in and count the missing values by linear interpolation
                    for i in range(missing_streak_length[streak]):
                        count += 1
                        self.data.iloc[nans_location[non_zero_indices[streak]] + i, self.data.columns.get_loc(sensor)] =\
                            beginning + (i+1) * (end-beginning) / (missing_streak_length[streak]+1)

        if verbose > 0:
            print(f"Interpolated {count} datapoints.")

        # Analyze the final data
        #self.full_data_analysis(bins, show, verbose)

    def standardize(self, data=None):
        """
        Function to standardize the dataset, i.e. put it to zero mean and 1 std
        The mean and std of each column (sensor) is kept in memory to reverse the
        operation and to be able to apply them to other datasets
        """

        inplace = False

        if data is None:
            data = self.data

            # First, check that no normalization or standardization was already performed
            assert self.is_normalized == False, f"The data is already normalized!"
            assert self.is_standardized == False, f"The data is already standardized!"

            # Set the standardized flag to true
            self.is_standardized = True
            inplace = True

        # Define and keep the means and stds in memory
        mean = data.mean()
        std = data.std()

        # Substract the mean
        data = data.subtract(mean)

        # Little trick to handle constant data (zero std --> cannot divide)
        # Only consider non zero variance columns
        non_zero_std = np.where(data.std().values > 1e-10)[0]
        # If there is a zero variance column, it is actually useless since it is constant
        if len(non_zero_std) < len(data.columns):
            print(f"Warning, 0 std for columns {np.where(data.std().values < 1e-10)[0]}, really useful?")

        # Divide by the std where possible
        data.iloc[:, non_zero_std] = data.iloc[:, non_zero_std].divide(std[non_zero_std])

        if inplace:
            self.data = data
            self.mean = mean
            self.std = std
        else:
            return data, mean, std

    def inverse_standardize(self, data=None):
        """
        Function to reverse the standardization to get the original scales back. If no data is provided,
        the entire dataset is scaled back.
        """

        # First sanity check: the data is already standardized
        assert self.is_standardized, f"The data is not standardized!"

        # If no data is provided, take the entire DataFrame of the DataSet
        if data is None:
            data = self.data
            # Change the normalized flag back
            self.is_standardized = False

        # Otherwise, copy the data to avoid issues
        else:
            if type(data) == pd.Series:
                data = pd.DataFrame(data=data, index=data.index, columns=[data.name])
            else:
                data = data.copy()

        # Get the places where the variance is not zero and multiply back
        # (the other columns were ignored)
        non_zero_std = np.where(self.std[data.columns].values > 1e-10)[0]
        data.iloc[:, non_zero_std] = data.iloc[:, non_zero_std].multiply(self.std[data.columns][non_zero_std])

        # Add the mean back
        data = data.add(self.mean[data.columns])

        data.iloc[:, np.where(self.std[data.columns].values <= 1e-10)[0]] = 0

        # Return the scaled data
        return data

    def normalize(self, data=None):
        """
        Function to normalize the dataset, i.e. scale it by the min and max values
        The min and max of each column (sensor) is kept in memory to reverse the
        operation and to be able to apply them to other datasets

        There is an additional trick here, the data is actually scaled between
        0.1 and 0.9 instead of the classical 0-1 scaling to avoid saturation
        This is supposed to help the learning
        """
        inplace = False

        if data is None:
            data = self.data

            # First, check that no normalization or standardization was already performed
            assert not self.is_standardized, f"The data is already standardized!"
            assert not self.is_normalized, f"The data is already normalized!"

            # Set the standardized flag to true
            self.is_normalized = True
            inplace = True

        # Define and save the min and max of each column
        max_ = data.max()
        min_ = data.min()

        # Little trick to handle constant data (zero std --> cannot divide)
        # Only consider non zero variance columns
        non_zero_div = np.where(max_ - min_ > 1e-10)[0]
        # If there is a zero variance column, it is actually useless since it is constant
        if len(non_zero_div) < len(data.columns):
            print(f"Warning, columns {np.where(max_ - min_ < 1e-10)[0]} are constant, really useful?")

        # Scale the data between 0.1 and 0.9
        data.iloc[:, non_zero_div] = 0.8 * (data.iloc[:, non_zero_div] - min_[non_zero_div]) / \
                                          (max_[non_zero_div] - min_[non_zero_div]) + 0.1

        data.iloc[:, np.where(max_ - min_ <= 1e-10)[0]] = 0.5

        if inplace:
            self.data = data
            self.min_ = min_
            self.max_ = max_
        else:
            return data, min_, max_

    def inverse_normalize(self, data=None):
        """
        Function to reverse the normalization to get the original scales back. If no data is provided,
        The entire data is scaled back.
        """

        # If no data is provided, take the entire DataFrame of the DataSet
        if data is None:
            data = self.data
            # First sanity check: the data is already normalized
            assert self.is_normalized, f"The data is not normalized!"
            # Change the normalized flag back
            self.is_normalized = False

        # Otherwise, copy the data to avoid issues
        else:
            if type(data) == pd.Series:
                data = pd.DataFrame(data=data, index=data.index, columns=[data.name])
            else:
                data = data.copy()

        # Get the places where the variance is not zero and multiply back
        # (the other columns were ignored)
        non_zero_div = np.where(self.max_[data.columns] - self.min_[data.columns] > 1e-10)[0]

        # Can get back to the original scale back
        data.iloc[:, non_zero_div] = (data.iloc[:, non_zero_div] - 0.1). \
                                         multiply(self.max_[data.columns][non_zero_div] -
                                                  self.min_[data.columns][non_zero_div]) / 0.8
        data.iloc[:, non_zero_div] = data.iloc[:, non_zero_div].add(self.min_[data.columns][non_zero_div])

        # Return the scaled data
        return data


def create_occupancy_data(start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE, interval: int = 15,
                          name: str = "Occupancy", save_data: bool = True, verbose: int = 0) -> DataSet:
    """
    Function to create a DataFrame containing the DFAB occupancy information.
    The dates when each room was occupied was filled in by hand (see NEST_data.py) based on the
    Excel sheet provided.

    For rooms that could be shared by several occupants, the total number of people with access to
    it is summed to create the data, so that a value of 3 for example means that 3 people had
    access to that room.

    Args:
        start_date: Starting date of the experiment running
        end_date:   Ending date of the experiment running
        interval:   Interval at which the data has been subsampled
        name:       Name of the data
        save_data:  Boolean to save the data
        verbose:    Verbose

    Returns:
        A DataFrame with the occupancy of each room between the starting and ending dates
    """

    # Try to load the DataFrame if it already exists
    try:
        if verbose > 0:
            print(f"Trying to load the {name} data...")

        occupancy_data = load_data(start_date=start_date,
                                   end_date=end_date,
                                   name=name)

        if verbose > 0:
            print("Data downloaded sucessfully!")

        # If we managed to download it, then there is no need to save it again
        save_data = False

    # If it fails, create the data
    except AssertionError:
        if UNIT == "DFAB":
            if verbose > 0:
                print(f"There is no {name} data.\nPreprocessing the data...")

            # Import the dates each bedroom was occupied - start and end
            from NEST_data.DFAB import occupancy_start_472, occupancy_start_571, \
                occupancy_start_574, occupancy_end_472, occupancy_end_571, occupancy_end_574

            # Build a time series containing with timestamps with the right interval from the
            # starting date to the ending one (this will be the index of the DataFrame)
            # Start with the starting date
            times = [pd.to_datetime(start_date, format="%Y-%m-%d")]
            # Iterate adding a new time stamp with the right interval until the ending date
            while times[-1] < pd.to_datetime(end_date, format="%Y-%m-%d"):
                times.append(times[-1] + pd.Timedelta(interval, 'm'))

            # Define the DataFrame with the right index and all the rooms in columns
            # We also initialize the data as zeroes everywhere (inoccupied)
            occupancy_data = pd.DataFrame(index=times, columns=[f"Occupancy {room}" for room in ROOMS], data=0)

            # For each bedroom, put a '1' when somebody was using it
            occupancy_data.loc[occupancy_start_472:occupancy_end_472, "Occupancy 472"] = 1
            occupancy_data.loc[occupancy_start_571:occupancy_end_571, "Occupancy 571"] = 1
            occupancy_data.loc[occupancy_start_574:occupancy_end_574, "Occupancy 574"] = 1

            # For the corridors (between 2 bedrooms), sum the value of the 2 bedrooms, as several
            # people can use them
            occupancy_data["Occupancy 573"] = occupancy_data["Occupancy 571"] + occupancy_data["Occupancy 574"]
            occupancy_data["Occupancy 474"] = occupancy_data["Occupancy 472"] + occupancy_data["Occupancy 476"]

            # Finally, the common room is used by the people living in all 4 bedrooms
            occupancy_data["Occupancy 371"] = occupancy_data[[f"Occupancy {room}" for \
                                                              room in ['472', '476', '571', '574']]].sum(axis=1)

    # Create a dataset with the data and return it
    return DataSet(data=occupancy_data,
                   name=name,
                   save_data=save_data,
                   informations=list(occupancy_data.columns),
                   verbose=verbose)


def create_full_dataset(datasets: List[DataSet], name: str) -> DataSet:
    """
    Create a full dataset out of several other ones

    Args:
        datasets:   list of DataSets to concatenate together
        name:       Name of the new dataset

    Returns:
        The full dataset
    """

    # Concatenate all the datasets using the custom Class function
    dataset = datasets[0]
    for other in datasets[1:]:
        dataset = dataset.concatenate(other=other, inplace=False)

    # Redefine the name of the dataset
    dataset.name = name

    # Return it
    return dataset


def create_time_data(dataset: DataSet, interval: int = 15, verbose: int = 0) -> DataSet:
    """
    Add a column corresponding to the month, weekday and the time interval to the data set
    The sin/cos transformation is applied to introduce smoothness and the fact that 23pm is
    'close' to 1am (same for the months)

    Args:
        dataset:    The DataSet to modify
        interval:   Interval of the data
        verbose:    Verbose
    """

    if verbose > 0:
        print("Adding the time information")

    # Month computation
#    dataset.data['Time month sin'] = np.sin(np.pi * dataset.data.index.month / 6)
#    dataset.data['Time month cos'] = np.cos(np.pi * dataset.data.index.month / 6)

    # Day of the week computation
#    dataset.data['Time weekday'] = dataset.data.index.weekday
    # Time interval during the day
    dataset.data['Time interval sin'] = np.sin(2 * np.pi * (60 * dataset.data.index.hour + dataset.data.index.minute)
                                               / 1440)
    dataset.data['Time interval cos'] = np.cos(2 * np.pi * (60 * dataset.data.index.hour + dataset.data.index.minute)
                                               / 1440)

    # Save it as "information" data for the DataSet
#    dataset.informations += ['Time month sin', 'Time month cos', 'Time weekday', 'Time interval sin',
#                             'Time interval cos']

    return dataset


def prepare_data(start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE, name: str = "Full_Data",
                 unit: str = UNIT, interval: int = 15, missing_values_threshold: int = 2, verbose: int = 0) -> DataSet:
    """
    Main function that puts together a DataSet containing everything

    Args:
        start_date:                 Date when to start
        end_date:                   Date when to end
        name:                       Name of the created DataSet
        unit:                       Which unit the data is from
        interval:                   Interval at which to subsample the data
        missing_values_threshold:   To clean the data, maximum length of missing value streaks to
                                    interpolate (the rest are ignored and let empty)
        verbose:                    Verbose to print information on what is going on

    Returns:
        The full prepared DataSet
    """
    # Import all the constants that are needed, i.e. all the wanted sensors with their
    # names and what is controllable
    from NEST_data.Weather import weather_ids, weather_name, weather_names
    from NEST_data.Battery import battery_ids, battery_name, battery_names, battery_controllable, \
        battery_controllable_NEST
    if unit == "DFAB":
        from NEST_data.DFAB import dhw_hp_ids, dhw_hp_name, dhw_hp_names, dhw_hp_controllable, \
            dhw_hp_controllable_NEST
        from NEST_data.DFAB import electricity_ids, electricity_name, electricity_names
        from NEST_data.DFAB import valves_ids, valves_name, valves_names, valves_controllable
        from NEST_data.DFAB import thermal_ids, thermal_name, thermal_names, thermal_controllable_NEST
    elif unit == "UMAR":
        from NEST_data.UMAR import dhw_hp_name, dhw_hp_names, dhw_hp_controllable, dhw_hp_ids, \
            dhw_hp_controllable_NEST
        from NEST_data.UMAR import electricity_name, electricity_names, electricity_ids
        from NEST_data.UMAR import valves_name, valves_names, valves_ids, valves_controllable
        from NEST_data.UMAR import thermal_name, thermal_names, thermal_ids, thermal_controllable_NEST
        from NEST_data.UMAR import rooms_name, rooms_names, rooms_ids

    # Try to load the preprocessed data
    try:
        if verbose > 0:
            print(f"Trying to load the full data...")
        data = load_data(name="Full",
                                    unit=unit,
                                    start_date=start_date,
                                    end_date=end_date)
        dataset = DataSet(data,
                          name="Full",
                          save_data=False,
                          controllable=battery_controllable+dhw_hp_controllable+valves_controllable,
                          controllable_NEST=battery_controllable_NEST+dhw_hp_controllable_NEST+
                                            thermal_controllable_NEST,
                          informations=[x for x in data.columns if ("Time" in x) | ("Occupancy" in x)])
        if verbose > 0:
            print("Data downloaded sucessfully!")
    # If it fails, preprocess it
    except AssertionError:

        # Create a list to save all the created DataSets
        datasets = []

        # Create and preprocess the weather data
        weather_data = DataStruct(id_list=weather_ids,
                                  name=weather_name,
                                  start_date=start_date,
                                  end_date=end_date)

        datasets.append(DataSet(weather_data,
                                names=weather_names,
                                interval=interval,
                                verbose=verbose))

        # Create and preprocess the battery data
        try:
            battery_data = DataStruct(id_list=battery_ids,
                                      name=battery_name,
                                      start_date=start_date,
                                      end_date=end_date)

            datasets.append(DataSet(battery_data,
                                    names=battery_names,
                                    controllable=battery_controllable,
                                    controllable_NEST=battery_controllable_NEST,
                                    interval=interval,
                                    verbose=verbose))
        except AssertionError:
            print("The battery data couldn't be downloaded for these dates")

            # Create the occupancy data
            datasets.append(create_occupancy_data(start_date=start_date,
                                                  end_date=end_date,
                                                  interval=interval,
                                                  verbose=verbose))

        # Create and preprocess the additional room data of UMAR
        rooms_data = DataStruct(id_list=rooms_ids,
                                 name=rooms_name,
                                 start_date=start_date,
                                 end_date=end_date)

        datasets.append(DataSet(rooms_data,
                                names=rooms_names,
                                interval=interval,
                                verbose=verbose))

        # Create and preprocess the domestic hot water heat pump data
        dhw_hp_data = DataStruct(id_list=dhw_hp_ids,
                                 name=dhw_hp_name,
                                 start_date=start_date,
                                 end_date=end_date)

        datasets.append(DataSet(dhw_hp_data,
                                names=dhw_hp_names,
                                controllable=dhw_hp_controllable,
                                controllable_NEST=dhw_hp_controllable_NEST,
                                interval=interval,
                                verbose=verbose))

        # Create and preprocess the electricity data
        electricity_data = DataStruct(id_list=electricity_ids,
                                      name=electricity_name,
                                      start_date=start_date,
                                      end_date=end_date)

        datasets.append(DataSet(electricity_data,
                                names=electricity_names,
                                interval=interval,
                                verbose=verbose))

        # Create and preprocess the data corresponding to the valves
        valves_data = DataStruct(id_list=valves_ids,
                                 name=valves_name,
                                 start_date=start_date,
                                 end_date=end_date)

        datasets.append(DataSet(valves_data,
                                names=valves_names,
                                controllable=valves_controllable,
                                interval=interval,
                                verbose=verbose))

        # Create and preprocess the data corresponding to the thermal control
        thermal_data = DataStruct(id_list=thermal_ids,
                                  name=thermal_name,
                                  start_date=start_date,
                                  end_date=end_date)

        datasets.append(DataSet(thermal_data,
                                names=thermal_names,
                                controllable_NEST=thermal_controllable_NEST,
                                interval=interval,
                                verbose=verbose))

        # Concatenate everything into one single DataSet
        dataset = create_full_dataset(datasets=datasets,
                                      name=name)

        # Add the time information
        dataset = create_time_data(dataset=dataset,
                                   interval=interval,
                                   verbose=verbose)

        dataset.clean_data(threshold=missing_values_threshold,
                           verbose=verbose)

        print("\nDataSet created successfully!")

        if verbose > 0:
            print("Saving the dataset...")
        save_data(data=dataset.data,
                  name="Full",
                  unit=unit,
                  start_date=start_date,
                  end_date=end_date)
        if verbose > 0:
            print("Dataset saved successfully!")

    # Return the full dataset
    return dataset
