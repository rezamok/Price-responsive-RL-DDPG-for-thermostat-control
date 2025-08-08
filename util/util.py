"""
Various constants and functions used in many places
"""
import os
import sys
os.chdir(r'C:/Users/remok/OneDrive - Danmarks Tekniske Universitet/Skrivebord/RL TRV/Models/Price-Responsive RL/util')
from typing import List

import pandas as pd
import numpy as np

from parameters import ROOMS, UNIT

FIGURES_SAVE_PATH = os.path.join("saves", "Figures") if sys.platform == "win32" \
    else os.path.join("..", "saves", "Figures")


def login_from_file(file_name: str) -> List[str]:
    """
    Loads login information from a file.
    """
    assert os.path.isfile(file_name), f"File: {file_name} not found!"
    with open(file_name, "r") as f:
        return [l.rstrip() for l in f if l.rstrip() != ""]


def flat_list(x: list):
    """
    Flattens nested list to one unique list
    """
    return [z for y in x for z in y]


def intersect_lists(l1: list, l2: list):
    """
    Returns elements in both lists
    """
    return [x for x in l1 if x in l2]


def construct_list_of(argument):
    """
    Helper function to construct a list of arguments to make sure the iterators work.

    Args:
        argument: the argument to build a list from

    Returns:
        The arguments in a list
    """

    if argument is not None:
        # If there is only one element given as input, put it in a list to make sure
        # the iterators work
        if type(argument) == str:
            return [argument]

        # Else save it as is if it has already the right format
        elif type(argument) == list:
            return argument

        # For now not implemented
        else:
            raise ValueError(f"Type {type(argument)} is not supported for the argument {argument}")

    # If nothing is given, initialize it as an empty list
    else:
        return []


def build_list_of(*args):
    """
    Function to build a list out of different elements (list or strings)
    """

    list_ = []
    for arg in args:
        list_.append(construct_list_of(arg))

    return flat_list(list_)


def save_data(data: pd.DataFrame, start_date: str, end_date: str, name: str, unit: str = UNIT,
              save_path: str = "../saves/Data_preprocessed") -> None:
    """
    Function to save a dataframe

    Args:
        start_date: Starting date of the file to save
        end_date:   Ending date of the file to save
        unit:       Unit corresponding to the model
        save_path:  Where to save it
    """

    # Bulk of the work
    data.to_csv(f"{save_path}/{unit}__{start_date}__{end_date}__{name}.csv")


def load_data(start_date: str, end_date: str, name: str, unit: str = UNIT,
              save_path: str = "../saves/Data_preprocessed") -> pd.DataFrame:
    """
    Function to load a dataframe if it exists

    Args:
        start_date: Starting date of the file to load
        end_date:   Ending date of the file to load
        unit:       Unit corresponding to the model
        save_path:  Where to load it
    """

    # Build the full path and check its existence
    full_path = f"{save_path}/{unit}__{start_date}__{end_date}__{name}.csv"
    assert os.path.exists(full_path), f"The file {full_path} doesn't exist."

    # Load the data and change the index to timestamps
    data = pd.read_csv(full_path, index_col=[0])
    data.index = pd.to_datetime(data.index)

    return data


def get_room(sensor: str):
    """
    Small helper function to know in which room a sensor is (i.e. from a name see which number is
    present)

    Args:
        sensor: the name of the sensor
    """

    for room in ROOMS:
        if room in sensor:
            return room


def compute_mae(predictions, true_data):
    """
    Computes the Mean Absolute Error between the predictions and the truc data
    """

    return np.abs(predictions - true_data)
