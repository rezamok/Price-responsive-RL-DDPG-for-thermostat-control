"""
File to handle the data collection and preprocessing from the NEST database
"""

import numpy as np
import pandas as pd
import scipy.ndimage

from typing import Tuple, List

from rest.client import DataStruct

from parameters import UNIT


def preprocessing_pipeline(data: DataStruct, name=None, names: dict=None, interval: int = 15,
                           linear_interpolation: bool = False, keep: str = 'last',
                           verbose: int = 0) -> pd.DataFrame:
    """
    Perform all the preprocessings described below in one pipeline:
        Creates a DataFrame with all the sensor measurements at each time step
        Smooth the data to correspond to the wanted intervals

    Args:
        datas:                  DataStruct for the NEST client
        name:                   Name of the data to create
        names:                  Dict of the name to give to the sensors, corresponding to NumericId
                                if None keeps the original names of the DQL database
        interval:               Interval to which the data should be smoothed
        linear_interpolation:   Flag to set to true if the missing values are to be interpolated linearly
        keep:                   How to handle double datas for one minute
                                'first', 'last' or False (to drop that timestamp)
        verbose:                verbose

    Returns:
        DataFrame with the preprocessed timeseries
    """

    # Get the data, either locally or downloading it from the NEST
    datas = data.get_data()

    # Create a DataFrame containing all the measurements at 1 minute time steps
    df = create_full_data(datas=datas,
                          names=names,
                          linear_interpolation=linear_interpolation,
                          keep=keep,
                          verbose=verbose)

    # Preprocess and clean the data
    if name == 'Weather':
        df = preprocess_weather_full(df)

    elif name == 'Battery':
        df = preprocess_battery_full(df)

    elif name == 'DHW_HP':
        df = preprocess_dhw_hp_full(df)

    elif name == 'Electricity':
        df = preprocess_electricity_full(df)

    elif name == 'Valves':
        df = preprocess_valves_full(df)

    elif name == 'Thermal':
        df = preprocess_thermal_full(df)

    elif name == "Rooms":
        df = preprocess_rooms_full(df)

    else:
        print(f"Warning: Unexpected name {name} given to the data - no special preprocessing applied")

    # Smooth it to the wanted intervals
    df = correct_interval(df=df,
                          interval=interval,
                          verbose=verbose)
    return df


def create_dataframe(data: Tuple[np.ndarray, np.ndarray], column: str, linear_interpolation: bool = False,
                     keep= 'last', verbose: int = 1) -> pd.DataFrame:
    """
    Function to put the collected data in a DataFrame form, with the timestamps as index and the
    sensor values in the column, with the sensor's description as header.

    This function assumes measurements are taken each minute and round the timestamps to
    minute values accordingly.
    There is the possibility to add NaNs ot to use linear interpolation to fill the missing values

    Args:
        data:                   Tuple of numpy arrays containing the data loaded from the NEST database
        column:                 Name to give to the sensor
        linear_interpolation:   Flag to set to True if linear interpolation is wanted
        keep:                   How to handle double datas for one minute
                                'first', 'last' or False (to drop that timestamp)
        verbose:                Verbose

    Returns:
        DataFrame with all the measured values
    """

    # Create a DataFrame with the given data, putting the timestamps as index
    #df = pd.DataFrame(data[0], index=data[1], columns=[column])
    # Round to the minutes
    #df.index = df.index.round('T')

    rounded_times = np.array(data[1] + np.timedelta64(30, 's'), dtype='datetime64[m]')
    index = pd.date_range(rounded_times[0], rounded_times[-1], freq='T')
    df = pd.DataFrame(index=index, columns=[column])

    df.loc[rounded_times, column] = pd.Series(data[0], index=rounded_times)

    # Drop the duplicates - here keeps the last one
    org_len = len(df)
    df = df.loc[~df.index.duplicated(keep=keep)]
    new_len = len(df)

    if verbose > 1:
        print(f"{org_len-new_len} duplicated timestamps were removed.")

    # Compute some values around the time steps observed in the data
    t_diffs = df.index[1:] - df.index[:-1]
    max_t_diff = np.max(t_diffs)
    mean_t_diff = np.timedelta64(t_diffs.mean())

    # Ensure the time steps correspond more or less to 1 minute
    #assert (np.timedelta64(45, 's') <= mean_t_diff) & (mean_t_diff <= np.timedelta64(75, 's')), \
       # "The are some time intervals far from one minute"

    # Interpolation of missing time steps if needed
    # Linear interpolation

    # Count missing values
    missing_values = 0

    if max_t_diff > np.timedelta64(1, 'm'):

        if linear_interpolation:

            # Find where time steps are missing
            while len(np.where(t_diffs > np.timedelta64(1, 'm'))[0]) > 0:

                # Where the (first) gap is
                hole = np.where(t_diffs > np.timedelta64(1, 'm'))[0][0]

                # Identify the time gap, starting and ending values
                diff = np.timedelta64(df.index[hole + 1] - df.index[hole], 'm').astype(int)
                start = df.iloc[hole].values
                end = df.iloc[hole + 1, :].values

                # Increment the count of missing values
                missing_values += diff - 1

                # Interpolate if needed, insert the new value
                for j in range(diff - 1):
                    # Define the new index (timestamp) to insert in the data
                    ind = df.index[hole] + np.timedelta64(j + 1, 'm')

                    # Linear interpolation of the corresponding value
                    val = (end - start) / diff * (j + 1) + start

                    # Insert the new value in the DataFrame by splitting it at the wanted
                    # index and then concatenate it around the new value
                    to_add = pd.DataFrame({df.columns[0]: val}, index=[ind])

                    df = pd.concat([df[:hole + j + 1], to_add, df[hole + j + 1:]])

                # Identify the new differences
                t_diffs = df.index[1:] - df.index[:-1]

    # Print the verbose
    #if verbose > 1:
        #print(f"{missing_values} missing values were found, and the corresponding timestamps added.")

    # Check that the interpolation worked
    t_diffs = df.index[1:] - df.index[:-1]
    #assert np.max(t_diffs) == np.timedelta64(1, 'm'), \
        #"The implemented interpolation failed"

    # Check that we don't have double values after the rounding
    assert np.min(t_diffs) == np.timedelta64(1, 'm'), \
        "There is a double value at the same time - correction is to be implemented"

    # Return the created DataFrame
    return df


def create_full_data(datas, names: dict = None, linear_interpolation: bool = False,
                     keep = 'last', verbose: int = 0):
    """
    Function to create a full DataFrame from the datas downloaded from the NEST database

    Args:
        datas:                  Datas downloaded from the NEST database
        names:                  Dict of the name of the sensors, if None will keep the original
                                name of the SQL database
        linear_interpolation:   Set to true if linear interpolation is wanted to fill
                                the missing values
        keep:                   How to handle double datas for one minute
                                'first', 'last' or False (to drop that timestamp)
        verbose:                Verbose

    Returns:
        df: DataFrame with the timestamps as indices and the corresponding measured
            values corresponding to each sensor
    """

    # Extract the data and metadata from the NESt data
    data, metadata = datas

    # When we want to change the names of the sensors
    if type(names) == dict:
        column = names[metadata[0]['numericId']]

    # Otherwise take the original names
    else:
        column = metadata[0]['description']

    # Create a first DataFrame with the first sensor (one value each minute)
    df = create_dataframe(data=data[0],
                          column=column,
                          linear_interpolation=linear_interpolation,
                          keep=keep,
                          verbose=verbose)
    # Iterate over the sensors to add their values in another column
    for i in range(1, len(data)):
        # When we want to change the names of the sensors
        if type(names) == dict:
            column = names[metadata[i]['numericId']]

        # Otherwise take the original names
        else:
            column = metadata[i]['description']

        # Keep orinigal names
        df2 = create_dataframe(data=data[i],
                               column=column,
                               linear_interpolation=linear_interpolation,
                               keep=keep,
                               verbose=verbose)

        # Sanity checks: ensure both sensors have been measured similarly
        # According to the current implementations
        if df.index[0] != df2.index[0]:
            print(f"Warning: DataFrame have not the same starting date {df.index[0]} and {df2.index[0]}")
        if df.index[-1] != df2.index[-1]:
            print(f"Warning: DataFrame have not the same ending date { df.index[-1]} and {df2.index[-1]}")
        if len(df) != len(df2):
            print("Warning: DataFrame not the same length")

        # Add the column at the end of the data, merging the indices
        df = df.join(df2, how='outer', rsuffix='2')

    # Return the created DataFrame
    return df


def delete_constant_streaks(df: pd.DataFrame, sensor: str, threshold: int) -> pd.DataFrame:
    """
    Function to remove sequences of data if it keeps constant for too long

    Args:
        df:         DataFrame to clean
        sensor:     Column to clean
        threshold:  Max length of constant data to keep

    Returns:
        the corrected DataFrame
    """
    # Transform it into an array to use numpy
    data = df[sensor].values

    # Label all the places where the difference is not 0 as True
    # Wrap it between two other True values for later use below
    transformed_data = np.concatenate([[True], np.diff(data) != 0.0, [True]])

    # Transform it into indices: create a lit of indices where this is True
    # i.e. a list of indices where the difference is not zero
    non_zero_indices = np.flatnonzero(transformed_data)

    # Use the 'diff' function again to get the length of the constant streaks
    # between two True values
    constant_length = np.diff(non_zero_indices)

    # Get the streaks where the constant length are above the threshold
    streaks_to_delete = np.where(constant_length > threshold)[0]

    # Delete the corresponding values in the DataFrame
    for streak in streaks_to_delete:
        df[sensor][non_zero_indices[streak]: non_zero_indices[streak] + constant_length[streak]] = np.nan

    return df


def clean_constant_streaks(df: pd.DataFrame, sensors: List, thresholds: dict) -> pd.DataFrame:
    """
    Function to remove sequences of data if it keeps constant for too long, for different sensors
    with different threshold values

    Args:
        df:         DataFrame to clean
        sensors:    Columns to clean
        thresholds: Max length of constant data to keep in each case

    Returns:
        the corrected DataFrame
    """

    # For each needed sensor needed, delete long constant streaks
    for sensor in sensors:
        df = delete_constant_streaks(df=df,
                                     sensor=sensor,
                                     threshold=thresholds[sensor])

    # Return the DataFrame
    return df


def delete_extreme_values(df: pd.DataFrame, sensor: str, limit: float) -> pd.DataFrame:
    """
    Function to delete some known extreme values, when the value jumps up and down from one
    minute to the next

    Args:
        df:     DataFrame to clean
        sensor: The column to clean in the DataFrame
        limit:  The limit of the "authorized" jumps

    Returns:
        new_df: the cleaned DataFrame
    """

    # First copy the data because we will modify it
    new_df = df.copy()

    # Transform it in a numpy array to use numpy later and keep only the known points
    data = df[sensor].values
    no_nans = data[~pd.isnull(data)]

    # Compute the jumps from one time step to the next
    diffs = np.insert(np.diff(no_nans), 0, 0)

    # Get the extreme jumps, i.e. the ones above the limit
    extremes_indices = np.where(np.abs(diffs) >= limit)[0]

    # Transform the indices back into the actual time indices of the dataframe
    indices = df.index[~pd.isnull(data)][extremes_indices]

    # Make sure we indeed have a jump over one minute (especially not longer)
    extreme_places = np.where(np.diff(indices) == np.timedelta64(1, "m"))[0]

    # Delete the corresponding data and return the new DataFrame
    new_df.loc[indices[extreme_places], sensor] = np.nan
    return new_df


def clean_extreme_values(df: pd.DataFrame, sensors: List, limits: dict) -> pd.DataFrame:
    """
    Function to remove extreme values, for different sensors
    with different limit values defining the limit

    Args:
        df:         DataFrame to clean
        sensors:    Columns to clean
        limits:     Maximum "jump" authorized by each sensor

    Returns:
        The corrected DataFrame
    """

    # For each needed sensor needed, delete long constant streaks
    for sensor in sensors:
        df = delete_extreme_values(df=df,
                                   sensor=sensor,
                                   limit=limits[sensor])

    # Return the DataFrame
    return df


def gaussian_filter_ignoring_nans(df: pd.DataFrame, sensor: str, sigma: float = 2.0, no_zeros: bool = False)\
        -> np.ndarray:
    """
    Applies 1-dimensional Gaussian Filtering ignoring occurrences of NaNs.

    From:
        https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

    Adapted to ignore zeros as well

    Args:
        df:     The DataFrame to process.
        sensor: The column to clean
        sigma:  Gaussian filter standard deviation
        no_zeros:   Flag to ignore zeros (i.e. let them at zero)

    Returns:
        Filtered time series.
    """

    nans = pd.isnull(df[sensor])
    if no_zeros:
        zeros = (df[sensor] == 0)

    v = df[sensor].copy()
    v[nans] = 0
    vv = scipy.ndimage.filters.gaussian_filter1d(v.values.tolist(), sigma=sigma)

    w = 0 * df[sensor].copy() + 1
    w[nans] = 0
    if no_zeros:
        w[zeros] = 0
    ww = scipy.ndimage.filters.gaussian_filter1d(w.values.tolist(), sigma=sigma)

    z = vv / ww
    z[nans] = np.nan
    if no_zeros:
        z[zeros] = 0

    df[sensor] = z
    return df


def apply_gaussian_filters(df: pd.DataFrame, sensors, sigmas: dict, no_zeros: bool = False) -> pd.DataFrame:
    """
    Function to remove extreme values, for different sensors
    with different limit values defining the limit

    Args:
        df:         DataFrame to process
        sensors:    Columns to process
        sigmas:     Dict containing the wanted standard deviation of the Gaussian kernels
        no_zeros:   Flag to ignore zeros (i.e. let them at zero)

    Returns:
        Corrected DataFrame
    """

    # For each needed sensor needed, delete long constant streaks
    for sensor in sensors:
        df = gaussian_filter_ignoring_nans(df=df,
                                           sensor=sensor,
                                           sigma=sigmas[sensor],
                                           no_zeros=no_zeros)

    # Return the DataFrame
    return df


def preprocess_weather_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess the weather data, i.e. clean the data

    Args:
        df: DataFrame containing all the weather sensors

    Returns:
        The modified DataFrame
    """

    # Clean values that were constant for too long, i.e. 20 hours for the irradiation
    # and 30 minutes for the rest
    # First create a dictionary with 30 minutes for each sensor
    sensors = df.columns
    thresholds = {sensor: 30 for sensor in sensors}
    # Then modify the threshold in the irradiation case
    thresholds['Weather solar irradiation'] = 20*60

    # Clean the data from constant values
    df = clean_constant_streaks(df=df,
                                sensors=sensors,
                                thresholds=thresholds)

    # Ensure the irradiation values make sense
    df["Weather solar irradiation"] = df["Weather solar irradiation"].clip(lower=0.0)

    # Remove extreme values in the pressures measurements, i.e. when it changes of more
    # than 40 bars
    df = delete_extreme_values(df=df,
                               sensor="Weather relative air pressure",
                               limit=40)

    # Smooth the data with a small Gaussian filter
    sigmas = {sensor: 2 for sensor in sensors}
    df = apply_gaussian_filters(df=df,
                                sensors=sensors,
                                sigmas=sigmas)

    # Return the DataFrame
    return df


def preprocess_battery_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess the battery data, i.e. clean the data

    Args:
        df: DataFrame containing all the battery sensors

    Returns:
        The modified DataFrame
    """

    # Clean values that were constant for too long, i.e. 24 hours for the SoC
    # and 6 hours for the power measurement
    thresholds = {'Battery SoC': 24*60,
                  'Battery power measurement': 6*60}
    sensors = thresholds.keys()

    # Clean the data from constant values
    df = clean_constant_streaks(df=df,
                                sensors=sensors,
                                thresholds=thresholds)

    # Ensure the SoC has valid values
    df["Battery SoC"] = df["Battery SoC"].clip(0, 100)

    # Return the DataFrame
    return df


def preprocess_dhw_hp_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess the domestic hot water heat pump data, i.e. clean the data

    Args:
        df: DataFrame containing all the heat pump sensors

    Returns:
        The modified DataFrame
    """

    # Some additional preprocessing is needed for UMAR
    if UNIT == "UMAR":
        # Clip negative values that don't make sense
        df["HP thermal consumption"] = df["HP thermal consumption"].clip(0)

        # Compute the electricity consumption of the pump
        from NEST_data.UMAR import dhw_hp_cop
        df["HP electricity consumption"] = df["HP thermal consumption"] / dhw_hp_cop

        # We can forget the thermal consumption
        df.drop(columns=["HP thermal consumption"], inplace=True)

        # Forget extremely low boiler tempereatures
        df["HP Boiler temperature"][np.where(df["HP Boiler temperature"] < 51)[0]] = np.nan

    # Clean values that were constant for too long, i.e. 10 days for the electricity
    # consumption and 30 minutes for the temperature
    thresholds = {'HP electricity consumption': 10 * 24 * 60,
                  'HP Boiler temperature': 30}
    sensors = thresholds.keys()

    # Clean the data from constant values
    df = clean_constant_streaks(df=df,
                                sensors=sensors,
                                thresholds=thresholds)

    # Return the DataFrame
    return df


def preprocess_electricity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess the electricity data, i.e. sum up all the known loads and create
    the total consumption (adding the measured one and all the automation loads - because
    some automation loads are not measured by the main meter)

    Args:
        df: DataFrame containing all the electrical sensors

    Returns:
        Nothing, the DataFrame is modified in place
    """

    if UNIT == "DFAB":
        # Create the sum of all the known loads
        individual_loads = ['HVAC consumption', 'Kitchen consumption',
                            'Lights and powerplugs consumption', 'Reserve electricity']

        # Clip those loads, they cannot be negative
        df[individual_loads] = df[individual_loads].clip(lower=0)

        # Sum them up
        df['Electricity sum of loads'] = df[individual_loads].sum(axis=1)

        # Create the sum of all the loads: take all the automation loads and add them to the
        # total measurement
        automation_loads = ['Emergency mains electricity consumption', 'Alarm electricity consumption',
                            'Fire alarm electricity consumption']
        total_loads = ['Electricity total measurement'] + automation_loads
        df['Electricity total consumption'] = df[total_loads].sum(axis=1)

        # Drop the individual measurements that aren't useful anymore
        df.drop(columns=individual_loads + automation_loads, inplace=True)

    elif UNIT == "UMAR":

        # Scale the PV production from DFAB and make sure it isn't negative
        from NEST_data.UMAR import pv_scale_factor
        df['Electricity PV production'] /= pv_scale_factor
        df['Electricity PV production'] = df['Electricity PV production'].clip(upper=0)

        # Compute the total electricity consumption
        df['Electricity total consumption'] = df.sum(axis=1)

        # Drop the other columns
        df.drop(columns=['Electricity PV production', 'Electricity total measurement'], inplace=True)

    # Return the DataFrame
    return df


def preprocess_electricity_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess the electricity data, i.e. create the wanted loads and clean
    the data

    Args:
        df: DataFrame containing all the electrical sensors

    Returns:
        The modified DataFrame
    """

    # Specific preprocessing to create the wanted loads
    df = preprocess_electricity(df)
    df[df["Electricity total consumption"] < -100] = np.nan

    # Clean values that were constant for more than 30 days
    sensors = df.columns
    thresholds = {sensor: 30 for sensor in sensors}
    df = clean_constant_streaks(df=df,
                                sensors=sensors,
                                thresholds=thresholds)

    # Smooth the data with a small Gaussian filter
    sigmas = {sensor: 1 for sensor in sensors}
    df = apply_gaussian_filters(df=df,
                                sensors=sensors,
                                sigmas=sigmas)

    # Return the DataFrame
    return df


def preprocess_valves_DFAB(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess the data from the valves of the floor heating in DFAB
    Some rooms have several valves, this function puts all the valves of any room together

    Args:
        df: DataFrame with all the valves as columns

    Returns:
        The modified DataFrame
    """

    # Get the columns (i.e. sensors) corresponding to each room with multiple valves
    columns_371 = []
    columns_472 = []
    columns_476 = []
    columns_571 = []
    columns_574 = []
    # Go through the columns and check which ones correspond to the wanted room
    for col in df.columns:
        if '371' in col:
            columns_371.append(col)
        elif '472' in col:
            columns_472.append(col)
        elif '476' in col:
            columns_476.append(col)
        elif '571' in col:
            columns_571.append(col)
        elif '574' in col:
            columns_574.append(col)
        else:
            pass

    # For each room, sum the valves creating a new column and delete the individuel
    # valves as they are not useful
    df['Thermal valves 371'] = df[columns_371].sum(axis=1)
    df.drop(columns=columns_371, inplace=True)
    df['Thermal valves 472'] = df[columns_472].sum(axis=1)
    df.drop(columns=columns_472, inplace=True)
    df['Thermal valves 476'] = df[columns_476].sum(axis=1)
    df.drop(columns=columns_476, inplace=True)
    df['Thermal valves 571'] = df[columns_571].sum(axis=1)
    df.drop(columns=columns_571, inplace=True)
    df['Thermal valves 574'] = df[columns_574].sum(axis=1)
    df.drop(columns=columns_574, inplace=True)

    # Return the DataFrame
    return df


def preprocess_valves_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to fully preprocess the valves: create the wanted columns
    and then clean the data

    Args:
        df: DataFrame with the valves data

    Returns:
        Modified DataFrame
    """

    # Specific preprocessing to add the valves in each room for the Nussbaum system in DFAB
    if UNIT == "DFAB":
        df = preprocess_valves_DFAB(df)

        # Clean values that were constant for more than 30 days
        sensors = df.columns
        thresholds = {sensor: 30*24*60 for sensor in sensors}
        df = clean_constant_streaks(df=df,
                                    sensors=sensors,
                                    thresholds=thresholds)

    # Return the DataFrame
    return df


def preprocess_thermal_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess the thermal data, i.e. clean the data of the room temperatures

    Args:
        df: DataFrame containing all the thermal sensors

    Returns:
        The modified DataFrame
    """

    if UNIT == "DFAB":
        # Clean values that were constant for too long, i.e. 24 hours
        sensors = df.columns
        thresholds = {sensor: 24 * 60 for sensor in sensors}

        # Clean the data from constant values
        df = clean_constant_streaks(df=df,
                                    sensors=sensors,
                                    thresholds=thresholds)

        # Remove extreme values in the measurements, when it change of more than 2 degrees
        # First get all the corresponding columns and create the dictionary
        measurement_columns = []
        for col in df.columns:
            if ('measurement' in col) | ('inlet' in col) | ('outlet' in col):
                measurement_columns.append(col)
        limits = {sensor: 1.5 for sensor in measurement_columns}
        # Clean the data of the extreme values
        df = clean_extreme_values(df=df,
                                  sensors=measurement_columns,
                                  limits=limits)

        # Apply a Gaussian filter to the measurements, with std 5
        sigmas = {sensor: 5 if "measurement" in sensor else 1 for sensor in measurement_columns}

        # Also apply a flter to the thermal total energy
        measurement_columns.append("Thermal total energy")
        sigmas["Thermal total energy"] = 1

        df = apply_gaussian_filters(df=df,
                                    sensors=measurement_columns,
                                    sigmas=sigmas)

    elif UNIT == "UMAR":

        # In that case the heating and cooling agents are separated - their consumptions cannot be negative
        df["Thermal heating energy"] = df["Thermal heating energy"].clip(0)
        df["Thermal cooling energy"] = df["Thermal cooling energy"].clip(0)

        # Clean the unexpectedly low temperature measurements
        sensors = [x for x in df.columns if "measurement" in x]
        for sensor in sensors:
            df[sensor][np.where(df[sensor] < 12)[0]] = np.nan
        df["Thermal inlet temperature"][np.where(df["Thermal inlet temperature"] < 10)[0]] = np.nan
        df["Thermal inlet temperature"][np.where(df["Thermal inlet temperature"] > 50)[0]] = np.nan

        # Add the inlet temperature measurements to the sensors to clean
        sensors += ["Thermal inlet temperature"]

        # Clean the data from constant values
        thresholds = {sensor: 24 * 60 for sensor in sensors}
        df = clean_constant_streaks(df=df,
                                    sensors=sensors,
                                    thresholds=thresholds)

        # Apply a Gaussian filter to the measurements, with std 5
        sigmas = {sensor: 5 if "measurement" in sensor else 1 for sensor in sensors}

        # Also apply filters to the thermal total energy
        sensors.append("Thermal heating energy")
        sensors.append("Thermal cooling energy")
        sigmas["Thermal heating energy"] = 1
        sigmas["Thermal cooling energy"] = 1

        # Apply all fiters
        df = apply_gaussian_filters(df=df,
                                    sensors=sensors,
                                    sigmas=sigmas)

    # Return the DataFrame
    return df


def preprocess_rooms_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess the additional data of the rooms, i.e. brightness, humidity, window openeings,...

    Args:
        df: DataFrame containing all the thermal sensors

    Returns:
        The modified DataFrame
    """

    if UNIT == "UMAR":

        df["Room 273 windows"] = df["Room 273 window a"] + df["Room 273 window b"]
        df.drop(columns=["Room 273 window a", "Room 273 window b"], inplace=True)

        df["Room 272 brightness"][np.where(df["Room 272 brightness"] > 2000)[0]] = np.nan
        df["Room 273 brightness"][np.where(df["Room 273 brightness"] > 15000)[0]] = np.nan
        df["Room 274 brightness"][np.where(df["Room 274 brightness"] > 2000)[0]] = np.nan

        df["Room 272 humidity"][np.where(df["Room 272 humidity"] < 12)[0]] = np.nan
        df["Room 273 humidity"][np.where(df["Room 273 humidity"] < 12)[0]] = np.nan
        df["Room 274 humidity"][np.where(df["Room 274 humidity"] < 12)[0]] = np.nan

        df["Room 272 humidity"][np.where(df["Room 272 humidity"] > 85)[0]] = np.nan
        df["Room 273 humidity"][np.where(df["Room 273 humidity"] > 85)[0]] = np.nan
        df["Room 274 humidity"][np.where(df["Room 274 humidity"] > 85)[0]] = np.nan

        illuminance_sensors = [x for x in df.columns if "brightness" in x]
        humidity_sensors = [x for x in df.columns if "humidity" in x]
        sensors = illuminance_sensors + humidity_sensors

        limits = {sensor: 5 for sensor in humidity_sensors}
        limits["Room 272 brightness"] = 100
        limits["Room 273 brightness"] = 5000
        limits["Room 274 brightness"] = 100

        df = clean_extreme_values(df=df,
                                  sensors=sensors,
                                  limits=limits)

        # Clean the data from constant values
        thresholds = {sensor: 24 * 60 for sensor in sensors}
        df = clean_constant_streaks(df=df,
                                    sensors=sensors,
                                    thresholds=thresholds)

        # Apply a Gaussian filter to the measurements, with std 5
        sigmas = {sensor: 5 if "humidity" in sensor else 1 for sensor in sensors}
        df = apply_gaussian_filters(df=df,
                                    sensors=sensors,
                                    sigmas=sigmas)

    else:
        raise ValueError(f"Do you really want to process additional room informations?")

    return df


def correct_interval(df: pd.DataFrame, interval: int = 15, verbose: int = 0) -> pd.DataFrame:
    """
    Creates a DataFrame with the wanted time steps.

    Temperatures are taken as snapshots and thus computed by averaging the measurements around
    each wanted time step. For example for the output at 9:15, the function computes the average
    of the values at 9:08, 9:09, ..., 9:21, 9:22

    For energy, weather or valves information, the average over the next interval is computed,
    i.e. over the values at 9:15, 9:16, ..., 9:28, 9:29

    The reason behind this difference is that given current temperature informations, we want
    to know them at the next time step by acting over the coming interval (using some
    energy and controlling the valves).

    Note: this is the latest version of the function, an old versions is below in case of problems.
    Both should yield the same result. This one looks a bit tedious but is actually orders of magnitude
    faster due to the use of the pandas magic.

    Args:
        df:         The DataFrame of measurements for each minute
        interval:   The wanted time step
        verbose:    Verbose

    Returns:
        The modified DataFrame with the wanted intervals in index
    """

    if verbose > 0:
        print(f"Correcting the data to have {interval} minutes intervals...")

    # Ensure the interval is large enough
    assert np.timedelta64(interval, 'm') >= np.timedelta64(interval, 'm'), \
        "Wanted interval is too short compared to the sampling interval!"

    # Nothing to do if the interval is already correct
    if interval == 1:
        return df

    else:

        # Get the 'temperature' sensors, i.e. the ones we want a current snapshot of
        temperature_sensors = [sensor for sensor in df.columns if ("temperature" in sensor) | ("SoC" in sensor)
                               if "outside" not in sensor]

        # The rest of the sensors, i.e. the one we are interested over the next interval
        other_sensors = [sensor for sensor in df.columns if ("temperature" not in sensor) & ("SoC" not in sensor)]
        if "Weather outside temperature" in df.columns:
            other_sensors.append("Weather outside temperature")

        # Ensure all columns are considered
        assert np.sum([(x in temperature_sensors) | (x in other_sensors) for x in df.columns]) == len(df.columns),\
        f"Not all columns are considered!"

        # Copy the data to avoid issues
        temperatures = df[temperature_sensors].copy()
        intervals = df[other_sensors].copy()

        # Calculate the half interval as a helper
        half_interval = int(interval / 2)

        # Get when the first and last value arise with respect to the interval length, as it impacts
        # the computations
        start = df.index[0].minute % interval
        end = temperatures.index[-1].minute % interval

        # Deal with the 'temperature' sensors. For each of the indices, average the values half an interval before
        # and after to define the new value and store it in a new DataFrame
        if len(temperatures.columns) > 0:

            # If the data starts in the first half of the interval: the first index must be rounded down
            if start <= half_interval:

                # Define the indices at which the intervals are
                temp_indices = np.arange(0, len(temperatures), interval)[1:] - start

                # Create the new dataframe with the wanted index and columns
                temperature_df = pd.DataFrame(columns=temperatures.columns,
                                              index=pd.Index([df.index[0] - pd.Timedelta(minutes=start)]).append(
                                                  temperatures.index[temp_indices]))

                # Fill the first value
                temperature_df.loc[df.index[0] - pd.Timedelta(minutes=start), :] = temperatures.head(
                    half_interval - start + 1).mean()

            # If the data starts in the second half: this changes all the indices
            else:
                # Define the indices at which the intervals are in that case
                temp_indices = np.arange(0, len(temperatures), interval)[1:-1] + interval - start

                # Define the new dataframe accordingly
                temperature_df = pd.DataFrame(columns=temperatures.columns,
                                              index=pd.Index([df.index[0] + pd.Timedelta(minutes=interval - start)]).
                                              append(temperatures.index[temp_indices]))

                # Fill in the first value
                temperature_df.loc[df.index[0] + pd.Timedelta(minutes=interval - start), :] = temperatures.head(
                    interval - start + half_interval + 1).mean()

            # Bulk of the work: use pandas magic and some trick to compute all the values at once (except the
            # first one (already above) and last one which are special cases)
            # Forget the last indices - special case
            temp_indices = temp_indices[:-1]

            # Define a 3D array to store everything we need: for each step in the interval store all the values
            temp_temperatures = np.zeros((interval, len(temp_indices), len(temperatures.columns)))
            temp_temperatures[0, :, :] = temperatures.iloc[temp_indices - half_interval, :].values
            for i in range(1, interval):
                temp_temperatures[i, :, :] = temperatures.iloc[temp_indices - half_interval + i, :].values

            # Average each interval
            temperature_df.loc[temperatures.index[temp_indices], :] = np.nanmean(temp_temperatures, axis=0)

            # Special case of the last index
            # If the data ends in the first half of an interval: round it down
            if end <= half_interval:
                temperature_df.loc[temperatures.index[-1] - pd.Timedelta(minutes=end), :] = \
                    temperatures.iloc[-(half_interval + end + 1):, :].mean()

            # Else round it up
            else:
                temperature_df.loc[df.index[-1] + pd.Timedelta(minutes=interval - end), :] = \
                    temperatures.iloc[-(end - half_interval):, :].mean()

        # Deal with the 'other' sensors. For each of the indices, average the values of the coming interval
        # and after to define the new value and store it in a new DataFrame
        if len(intervals.columns) > 0:

            # Define the indices at which the intervals begin and the new dataframe
            int_indices = np.arange(0, len(intervals), interval)[1:] - start
            intervals_df = pd.DataFrame(columns=intervals.columns,
                                        index=pd.Index([df.index[0] - pd.Timedelta(minutes=start)]).append(
                                            intervals.index[int_indices]))

            # Forget the last index and compute the first index, which are special cases
            int_indices = int_indices[:-1]
            intervals_df.loc[df.index[0] - pd.Timedelta(minutes=start), :] = intervals.head(interval - start).mean()

            # Bulk of the work: use pandas magic and some trick to compute all the values at once (except the
            # first one (already above) and last one which are special cases)

            # Again define a 3D array to store everything
            temp_intervals = np.zeros((interval, len(int_indices), len(intervals.columns)))
            temp_intervals[0, :, :] = intervals.iloc[int_indices, :].values
            for i in range(1, interval):
                temp_intervals[i, :, :] = intervals.iloc[int_indices + i, :].values

            # Average all the values over each interval
            intervals_df.loc[intervals.index[int_indices], :] = np.nanmean(temp_intervals, axis=0)

            # Last value computation
            intervals_df.loc[df.index[-1] - pd.Timedelta(minutes=end), :] = intervals.iloc[-end - 1:, :].mean()

        # Return the created DataFrame, depending on the type of sensors observed
        if len(temperatures.columns) > 0:
            if len(intervals.columns) > 0:
                return temperature_df.join(intervals_df, how='outer')
            else:
                return temperature_df
        else:
            return intervals_df


def correct_interval_OLD(df: pd.DataFrame, interval: int = 15, verbose: int = 0) -> pd.DataFrame:
    """
    Creates a DataFrame with the wanted time step by averaging the measurements
    around each wanted time steps

    For example for the output at 9:15, the function computes the average
    of the values at 9:08, 9:09, ..., 9:21, 9:22

    Args:
        df:         The DataFrame of measurements for each minute
        interval:   The wanted time step
        verbose:    Verbose

    Returns:
        The modified DataFrame with the wanted intervals in index
    """

    if verbose > 0:
        print(f"Correcting the data to have {interval} minutes intervals...\nThis can take a few minutes")

        # Ensure the interval is large enough
    assert np.timedelta64(interval, 'm') >= np.timedelta64(interval, 'm'), \
        "Wanted interval is too short compared to the sampling interval!"

    # Nothing to do if the interval is already correct
    if interval == 1:
        return df

    else:
        temperature_sensors = [sensor for sensor in df.columns if ("temperature" in sensor) | ("SoC" in sensor)
                               if "outside" not in sensor]
        other_sensors = [sensor for sensor in df.columns if ("temperature" not in sensor) & ("SoC" not in sensor)]
        if "Weather outside temperature" in df.columns:
            other_sensors.append("Weather outside temperature")

        # Ensure all columns are considered
        assert np.sum([(x in temperature_sensors) | (x in other_sensors) for x in df.columns]) == len(df.columns),\
        f"Not all columns are considered!"

        temperatures = df[temperature_sensors].copy()
        intervals = df[other_sensors].copy()

        # Calculate the half interval as a helper
        half_interval = int(interval / 2)

        # Create the new DataFrames
        temperature_df = pd.DataFrame(columns=temperatures.columns)
        intervals_df = pd.DataFrame(columns=intervals.columns)

        # Compute the first value - this depends when the first data points was measured
        # Get when the first value arises, i.e. at which point in the interval
        start = df.index[0].minute % interval

        # If the data starts in the first half of the interval: round it down
        if start <= half_interval:
            # This means a new Timestamp is created
            temperature_df.loc[df.index[0] - pd.Timedelta(minutes=start), :] = temperatures.head(half_interval - start + 1).mean()
            # Define the indices at which the intervals are
            temp_indices = np.arange(0, len(temperatures), interval)[1:] - start

        # If the data starts in the second half: round it up
        else:
            temperature_df.loc[df.index[0] + pd.Timedelta(minutes=interval - start), :] = temperatures.head(
                interval - start + half_interval + 1).mean()
            # Define the indices at which the intervals are in that case
            temp_indices = np.arange(0, len(temperatures), interval)[1:-1] + interval - start

        intervals_df.loc[df.index[0] - pd.Timedelta(minutes=start), :] = intervals.head(interval - start).mean()
        int_indices = np.arange(0, len(intervals), interval)[1:] - start

        # For each of the indices, average the values half an interval before and after to define
        # the value during that interval and store it in the new DataFrame
        for ind in temp_indices:
            temperature_df.loc[temperatures.index[ind], :] = temperatures.iloc[ind - half_interval: ind + half_interval + 1, :].mean()

        for ind in int_indices:
            intervals_df.loc[intervals.index[ind], :] = intervals.iloc[ind: ind + interval, :].mean()

        # Last value computation, again depending on when the last value was measured,
        # i.e. at which point in the interval
        end = temperatures.index[-1].minute % interval

        # If the data ends in the first half of an interval: round it down
        if end <= half_interval:
            temperature_df.loc[temperatures.index[-1] - pd.Timedelta(minutes=end), :] = temperatures.iloc[-(half_interval + end + 1):, :].mean()

        # Else round it up, creating a new Timestamp
        else:
            temperature_df.loc[df.index[-1] + pd.Timedelta(minutes=interval - end), :] = \
                temperatures.iloc[-(end - half_interval):, :].mean()

        # Return the created DataFrame
        return temperature_df.join(intervals_df, how='inner')
