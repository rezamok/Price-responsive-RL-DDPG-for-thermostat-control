import pandas as pd
import numpy as np

from rest.client import DataStruct
from data_preprocessing.dataset import DataSet, prepare_data
from data_preprocessing.preprocessing import apply_gaussian_filters, clean_constant_streaks

from parameters import ROOMS, data_kwargs
from util.plots import plot_time_series


def disaggregate_rooms_energy(data, plot=True):
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

    # Copy the data to avoid problems
    df = data.copy()

    # Given and known mass flow design of each valve
    flows = {"Thermal valve 272": 217,
             "Thermal valve 273": 217 * 3,  # there are 3 valves
             "Thermal valve 274": 217,
             "Thermal valve 275": 70,
             "Thermal valve 276": 70}

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
    #proportions = valves_flows.divide(total_flows, axis=0)
    #proportions = proportions.fillna(0)

    nonzeros = np.where(total_flows > 0)[0]
    proportions = valves_flows.copy()
    proportions.iloc[nonzeros, :] = valves_flows.iloc[nonzeros, :].divide(total_flows[nonzeros], axis=0)

    # Compute the energy of each room
    room_energies = proportions.multiply(total_energy, axis=0)
    room_energies.rename(columns={f"Thermal valve {x}": f"Energy room {x}" for x in ROOMS}, inplace=True)

    # Plot the energies and return them
    if plot:
        plot_time_series(room_energies)

    return room_energies


def prepare_UMAR_data(data_kwargs: dict = data_kwargs):
    """
    Pipeline of actions to take to prepare the data for a model of UMAR

    Args:
        data_kwargs:    data arguments (start and end date, interval, ...)

    Returns:
        A preprocessed dataset ready to be put into a model
    """

    print("Preparing the data...")

    # Use the custom function to load and prepare the full dataset from the NEST data
    dataset = prepare_data(start_date=data_kwargs['start_date'],
                           end_date=data_kwargs['end_date'],
                           name=data_kwargs['name'],
                           interval=data_kwargs['interval'],
                           missing_values_threshold=data_kwargs['missing_values_threshold'],
                           verbose=data_kwargs['verbose'])

    # Only keep the measurements, controls and general informations
    dataset.data = dataset.data[dataset.controllable + dataset.measurements + dataset.informations]

    # We can drop the battery information in that case, as it is independent
    if data_kwargs["small_model"]:
        dataset.data.drop(columns=[x for x in dataset.data.columns if ("Battery" in x) | ("HP" in x) | ("brightness" in x) | ("relative" in x) | ("humidity" in x) | ("window" in x) | ("wind" in x) | ("weekday" in x)], inplace=True)
    else:
        dataset.data.drop(columns=[x for x in dataset.data.columns if ("Battery" in x) | ("HP" in x)], inplace=True)

    # Compute the total energy out of the heating and cooling energies
    dataset.data["Thermal total energy"] = dataset.data["Thermal heating energy"] - \
                                           dataset.data["Thermal cooling energy"]
    dataset.data.drop(columns=["Thermal heating energy", "Thermal cooling energy"], inplace=True)

    # Small outlier correction (now corrected in the preprocessing phase but needed for older data)
    dataset.data[(dataset.data["Thermal inlet temperature"] > 50)] = np.nan

    # Get the columns of the temperature measurements and smooth it out to ease learning,
    # applying gaussian filters of size 2
    temperature_columns = [column for column in dataset.data.columns if "temperature measurement" in column]

    thresholds = {column: 12 * 60 / dataset.interval for column in temperature_columns}
    dataset.data = clean_constant_streaks(df=dataset.data,
                                          sensors=temperature_columns,
                                          thresholds=thresholds)

    sigmas = {column: 2 for column in temperature_columns}
    dataset.data = apply_gaussian_filters(df=dataset.data,
                                          sensors=temperature_columns,
                                          sigmas=sigmas)

    # Get the temperature measurements
    temperatures = dataset.data[temperature_columns].copy()
    temperatures = temperatures.iloc[:-1, :]

    # Create new DataFrame and compute the temperature differences
    differences = pd.DataFrame(columns=temperature_columns)
    for column in differences.columns:
        # Compute the differences
        difference = np.diff(dataset.data[column].values)

        # Data cleaning: Delete data where the temperature changes of more than 0.5 degrees in 15 minutes
        difference[(difference < -0.5) | (difference > 0.5)] = np.nan
        temperatures[(difference < -0.5) | (difference > 0.5)] = np.nan

        # Store the differences in the correct DataFrame
        differences[column] = difference

    # Store the new values of the temperatures (with the deleted data where jumps happened)
    # and interpolate short missing values streaks
    dataset.data.loc[temperatures.index, temperature_columns] = temperatures
    dataset.data.interpolate(method="linear", limit=4, axis=0, inplace=True)

    # Clean data inconsistencies, i.e. when the valves are open but no energy is used or
    # when the valves are closed but energy is used
    # Sum all the valves openings to see when at least one valve was open
    columns = [x for x in dataset.data.columns if "valve" in x]
    valves = np.sum(dataset.data[columns], axis=1)
    # Get all the indices where it physically doesn't make sense
    inconsistencies = np.where(((np.abs(dataset.data["Thermal total energy"].values) > 0.025) & (valves == 0)) |
                               ((np.abs(dataset.data["Thermal total energy"].values) < 0.025) & (valves > 0)))[0]

    print(f"Dropped {len(inconsistencies)} inconsistent points")
    # Delete that data
    dataset.data.loc[dataset.data.index[inconsistencies], "Thermal total energy"] = np.nan

    dataset.data = dataset.data.join(disaggregate_rooms_energy(dataset.data, plot=False), how='outer')

    dataset.data["Case"] = (dataset.data[temperature_columns].mean(axis=1) < dataset.data[
        "Thermal inlet temperature"]) * 1 * 2 - 1

    inconsistencies = np.where(((dataset.data["Case"] == 0) & (dataset.data["Thermal total energy"] > 0)) |
                               ((dataset.data["Case"] == 1) & (dataset.data["Thermal total energy"] < 0)))[0]
    print(f"Dropped {len(inconsistencies)} inconsistent points")

    dataset.data.loc[dataset.data.index[inconsistencies], "Case"] = np.nan

    dataset.data.drop(columns=["Thermal total energy"], inplace=True)

    # Save the differences and interpolate them
    if data_kwargs["predict_differences"]:
        dataset.differences = differences
        dataset.differences.interpolate(method="linear", limit=4, axis=0, inplace=True)

    # If standardization is wanted
    if data_kwargs['to_standardize']:
        dataset.standardize()
        # Make sure we don't try to normalize it later
        data_kwargs['to_normalize'] = False

        # Standardize the differences as well and recall the mean and std
        if data_kwargs["predict_differences"]:
            dataset.differences, dataset.mean_differences, dataset.std_differences = \
                dataset.standardize(dataset.differences)

    # Else, normalization is usually done
    elif data_kwargs['to_normalize']:
        dataset.normalize()
        # Normalize the differences as well and recall the min and max
        if data_kwargs["predict_differences"]:
            dataset.differences, dataset.min_differences, dataset.max_differences = \
                dataset.normalize(dataset.differences)

    # Print the result and return it
    print("\nData ready!")
    return dataset


class NESTData:
    """
    Class to handle data from NEST
    """

    def __init__(self, data_kwargs: dict = data_kwargs, new_version=True):

        # Define a dataset and normalize / standardize it
        if new_version:
            self.dataset = prepare_UMAR_data(data_kwargs=data_kwargs)
        else:
            self.dataset = prepare_NEST_data(data_kwargs=data_kwargs)

        self.data = self.dataset.data.copy()
        self.predict_differences = data_kwargs["predict_differences"]
        if self.predict_differences:
            self.differences = self.dataset.differences
        else:
            self.differences = None

    # Some properties inherited from the DataSet class
    @property
    def start_date(self):
        return self.dataset.start_date

    @property
    def end_date(self):
        return self.dataset.end_date

    @property
    def is_normalized(self):
        return self.dataset.is_normalized

    @property
    def mean(self):
        return self.dataset.mean

    @property
    def std(self):
        return self.dataset.std

    @property
    def is_standardized(self):
        return self.dataset.is_standardized

    @property
    def min_(self):
        return self.dataset.min_

    @property
    def max_(self):
        return self.dataset.max_


## Battery Data

def transform_data(data):
    """
    Function to transform the data of the battery into the needed form for the model, i.e.
    from SoC and power measurements to SoC differences and power averages.

    This will be used to describe the new SoC when an agent decides to input a certain
    power to the battery.

    Args:
        data:       The data to transform

    Returns:
        new_data:   The new DataFrame
    """

    # Define an new DataFrame
    new_data = data.iloc[:-1, :].copy()

    # Compute the SoC differences between intervals
    new_data["Delta SoC"] = np.diff(data["Battery SoC"])

    # Compute the power averages during the intervals
    new_data["Average power"] = (data["Battery power measurement"][:-1].values +
                                 data["Battery power measurement"][1:].values) / 2

    # Drop the missing values and return the new data
    new_data.dropna(how='any', inplace=True)
    return new_data


class BatteryData:
    """
    Special class to handle the Battery data, as its behavior is independent from the rest of
    the NEST
    """

    def __init__(self, data_kwargs):
        """
        Initialize the data using the DataSet class from the data_preprocessing part

        Args:
            Classical arguments, the same as for the dataset class
        """

        from NEST_data.Battery import battery_ids, battery_name, battery_names, battery_controllable,\
            battery_controllable_NEST

        # Create the DataStruct used to access the NEST database
        battery_data = DataStruct(id_list=battery_ids,
                                  name=battery_name,
                                  start_date=data_kwargs["start_date"],
                                  end_date=data_kwargs["end_date"])

        # Create the DataSet, but we are here only interested in the data itself
        self.dataset = DataSet(battery_data,
                               names=battery_names,
                               controllable=battery_controllable,
                               controllable_NEST=battery_controllable_NEST,
                               interval=data_kwargs["interval"],
                               verbose=data_kwargs["verbose"])

        # Transform the data into the form used in the models
        self.data = transform_data(self.dataset.data)

    # Useful properties to keep in mind
    @property
    def start_date(self):
        return self.dataset.start_date

    @property
    def end_date(self):
        return self.dataset.end_date



def prepare_NEST_data(data_kwargs: dict = data_kwargs):
    """
    Pipeline of actions to take to prepare the data for a model

    Args:
        data_kwargs:    data arguments (start and end date, interval, ...)
        model_kwargs:   model arguments (n_autoregression, ...)
    """

    print("Preparing the data...")

    print("WARNING: Deprecated for differences doesn't work")

    # Use the custom function to load and prepare the full dataset from the NEST data
    dataset = prepare_data(start_date=data_kwargs['start_date'],
                           end_date=data_kwargs['end_date'],
                           name=data_kwargs['name'],
                           interval=data_kwargs['interval'],
                           missing_values_threshold=data_kwargs['missing_values_threshold'],
                           verbose=data_kwargs['verbose'])

    # Only keep the measurements, controls and general informations
    dataset.data = dataset.data[dataset.controllable + dataset.measurements + dataset.informations]

    # We can drop the battery information in that case, as it is independent
    dataset.data.drop(columns=[x for x in dataset.data.columns if "Battery" in x], inplace=True)
    if data_kwargs["unit"] == "DFAB":
        dataset.data.drop(columns=["Electricity sum of loads", "Electricity total measurement"], inplace=True)

    # Compute the total energy
    dataset.data["Thermal total energy"] = dataset.data["Thermal heating energy"] -\
                                           dataset.data["Thermal cooling energy"]
    dataset.data.drop(columns=["Thermal heating energy", "Thermal cooling energy"], inplace=True)

    # Modify the data if we want to predict differences and not the values themselves
    if data_kwargs["predict_differences"]:

        # Create new DataFrame
        differences = pd.DataFrame(columns=data_kwargs["components"])
        for column in differences.columns:

            # Check if there are some components (energy components typically) for which we don't
            # want to predict differences but the actual value
            if column not in data_kwargs["not_differences_components"]:
                # Compute the differences
                differences[column] = np.diff(dataset.data[column].values)
            else:
                differences[column] = dataset.data[column].values[1:]

        # Save the data
        dataset.differences = differences

    # If standardization is wanted
    if data_kwargs['to_standardize']:
        dataset.standardize()
        # Make sure we don't try to normalize it later
        data_kwargs['to_normalize'] = False

    # Else, normalization is usually done
    elif data_kwargs['to_normalize']:
        dataset.normalize()

    return dataset


def prepare_UMAR_data_old(data_kwargs: dict = data_kwargs):
    """
    Pipeline of actions to take to prepare the data for a model of UMAR

    Args:
        data_kwargs:    data arguments (start and end date, interval, ...)

    Returns:
        A preprocessed dataset ready to be put into a model
    """

    print("Preparing the data...")

    # Use the custom function to load and prepare the full dataset from the NEST data
    dataset = prepare_data(start_date=data_kwargs['start_date'],
                           end_date=data_kwargs['end_date'],
                           name=data_kwargs['name'],
                           interval=data_kwargs['interval'],
                           missing_values_threshold=data_kwargs['missing_values_threshold'],
                           verbose=data_kwargs['verbose'])

    # Only keep the measurements, controls and general informations
    dataset.data = dataset.data[dataset.controllable + dataset.measurements + dataset.informations]

    # We can drop the battery information in that case, as it is independent
    dataset.data.drop(columns=[x for x in dataset.data.columns if "Battery" in x], inplace=True)
    if data_kwargs["unit"] == "DFAB":
        dataset.data.drop(columns=["Electricity sum of loads", "Electricity total measurement"], inplace=True)

    # Compute the total energy out of the heating and cooling energies
    dataset.data["Thermal total energy"] = dataset.data["Thermal heating energy"] - \
                                           dataset.data["Thermal cooling energy"]
    dataset.data.drop(columns=["Thermal heating energy", "Thermal cooling energy"], inplace=True)

    # Small outlier correction (now corrected in the preprocessing phase but needed for older data)
    dataset.data[(dataset.data["Thermal inlet temperature"] > 50)] = np.nan

    # Clean data inconsistencies, i.e. when the valves are open but no energy is used or
    # when the valves are closed but energy is used
    # Sum all the valves openings to see when at least one valve was open
    columns = [x for x in dataset.data.columns if "valve" in x]
    valves = np.sum(dataset.data[columns], axis=1)
    # Get all the indices where it physically doesn't make sense
    inconsistencies = np.where(((np.abs(dataset.data["Thermal total energy"].values) > 0.025) & (valves == 0)) |
                               ((np.abs(dataset.data["Thermal total energy"].values) < 0.025) & (valves > 0)))[0]

    print(f"Dropped {len(inconsistencies)} inconsistent points")
    # Delete that data
    dataset.data.loc[dataset.data.index[inconsistencies], "Thermal total energy"] = np.nan

    # Get the columns of the temperature measurements and smooth it out to ease learning,
    # applying gaussian filters of size 2
    temperature_columns = [column for column in dataset.data.columns if "temperature measurement" in column]

    sigmas = {column: 2 for column in temperature_columns}
    dataset.data = apply_gaussian_filters(df=dataset.data,
                                          sensors=temperature_columns,
                                          sigmas=sigmas)

    dataset.data["Case"] = (dataset.data[temperature_columns].mean(axis=1) < dataset.data[
        "Thermal inlet temperature"]) * 1 * 2 - 1

    # Get the temperature measurements
    temperatures = dataset.data[temperature_columns].copy()
    temperatures = temperatures.iloc[:-1, :]

    # Create new DataFrame and compute the temperature differences
    differences = pd.DataFrame(columns=temperature_columns)
    for column in differences.columns:
        # Compute the differences
        difference = np.diff(dataset.data[column].values)

        # Data cleaning: Delete data where the temperature changes of more than 0.5 degrees in 15 minutes
        difference[(difference < -0.5) | (difference > 0.5)] = np.nan
        temperatures[(difference < -0.5) | (difference > 0.5)] = np.nan

        # Store the differences in the correct DataFrame
        differences[column] = difference

    # Store the new values of the temperatures (with the deleted data where jumps happened)
    # and interpolate short missing values streaks
    dataset.data.loc[temperatures.index, temperature_columns] = temperatures
    dataset.data.interpolate(method="linear", limit=4, axis=0, inplace=True)

    # Save the differences and interpolate them
    if data_kwargs["predict_differences"]:
        dataset.differences = differences
        dataset.differences.interpolate(method="linear", limit=4, axis=0, inplace=True)

    # If standardization is wanted
    if data_kwargs['to_standardize']:
        dataset.standardize()
        # Make sure we don't try to normalize it later
        data_kwargs['to_normalize'] = False

        # Standardize the differences as well and recall the mean and std
        if data_kwargs["predict_differences"]:
            dataset.differences, dataset.mean_differences, dataset.std_differences = \
                dataset.standardize(dataset.differences)

    # Else, normalization is usually done
    elif data_kwargs['to_normalize']:
        dataset.normalize()
        # Normalize the differences as well and recall the min and max
        if data_kwargs["predict_differences"]:
            dataset.differences, dataset.min_differences, dataset.max_differences = \
                dataset.normalize(dataset.differences)

    # Print the result and return it
    print("\nData ready!")
    return dataset


class NESTData_old:
    """
    Class to handle data from NEST
    """

    def __init__(self, data_kwargs: dict = data_kwargs, new_version=True):

        # Define a dataset and normalize / standardize it
        if new_version:
            self.dataset = prepare_UMAR_data_old(data_kwargs=data_kwargs)
        else:
            self.dataset = prepare_NEST_data(data_kwargs=data_kwargs)

        self.data = self.dataset.data.copy()
        self.predict_differences = data_kwargs["predict_differences"]
        if self.predict_differences:
            self.differences = self.dataset.differences
        else:
            self.differences = None

    # Some properties inherited from the DataSet class
    @property
    def start_date(self):
        return self.dataset.start_date

    @property
    def end_date(self):
        return self.dataset.end_date

    @property
    def is_normalized(self):
        return self.dataset.is_normalized

    @property
    def mean(self):
        return self.dataset.mean

    @property
    def std(self):
        return self.dataset.std

    @property
    def is_standardized(self):
        return self.dataset.is_standardized

    @property
    def min_(self):
        return self.dataset.min_

    @property
    def max_(self):
        return self.dataset.max_