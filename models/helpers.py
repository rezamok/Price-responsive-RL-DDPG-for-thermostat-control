"""
Some helpers for the models
"""

import os
import sys
import _pickle as pickle


def name_factory(data, model_kwargs):
    """
    Function to create helpful and somewhat unique names to easily save and load the wanted models
    This uses the starting and ending date of the data used to fit the model, as well as another
    part in "model_name" that is specific to each model, representing the model type and possibly
    some hyperparameters' choices

    Args:
        data:       Data the model was fitted to
        model_kwargs: Specific name depending on the model type and hyperparameters

    Returns:
        A full name to save the model as
    """

    # First check that the data structure does indeed contain a starting date information
    assert data.start_date is not None, "The model must be linked to the right data type"

    if type(model_kwargs) == str:
        name = model_kwargs

    elif type(model_kwargs) == dict:
        name = f"{model_kwargs['unit']}_{model_kwargs['model_name']}"
        if model_kwargs["predict_differences"]:
            name += "_differences"

        if model_kwargs["model_name"] in ["NNModel", "NNExtractionModel"]:
            name += f"_{model_kwargs['n_autoregression']}"

            hidden_names = ["_" + str(x) for x in model_kwargs["hidden_sizes"]]
            for hidden_name in hidden_names:
                name += hidden_name

            if model_kwargs['model_name'] == "NNExtractionModel":
                feature_names = ["_" + str(x) for x in model_kwargs["feature_extraction_sizes"]]
                for feature_name in feature_names:
                    name += feature_name

        elif model_kwargs["model_name"] in ["LSTMExtractionModel", "LSTMModel"]:
            name += f"_{model_kwargs['hidden_size']}"

            if model_kwargs["model_name"] == "LSTMExtractionModel":
                name += f"_{model_kwargs['n_autoregression']}"

            if model_kwargs['NN']:
                hidden_names = ["_" + str(x) for x in model_kwargs["hidden_sizes"]]
                for hidden_name in hidden_names:
                    name += hidden_name

    # Then build the full name (only retaining the first 8 characters of the dates, which
    # puts them in the format yy-mm-dd
    return f"{data.start_date[:8]}__{data.end_date[:8]}__{name}"

save_path = os.path.join("saves", "Models") if sys.platform == "win32" else os.path.join("..", "saves", "Models")

def save_model(model, name: str, save_path=save_path):
    """
    Function to save a model

    Args:
        model:      Model to save
        name:       Name of the model
        save_path:  Where to save the model
    """

    with open(os.path.join(save_path, name + '.pkl'), 'wb') as output:
        pickle.dump(model, output, -1)


def load_model(name, save_path=save_path, verbose: int = 1):
    """
    Function to load a model

    Args:
        name:       Name of the model
        save_path:  Path where to save the model
        verbose:    Verbose

    Returns:
        The downloaded model
    """

    # Build the full path to the model and check its existence
    full_path = os.path.join(save_path, name + '.pkl')
    assert os.path.exists(full_path), f"The file {full_path} doesn't exist."

    # If it exists, load it
    with open(full_path, 'rb') as input:
        model = pickle.load(input)

    # Print the success of the loading and return the model
    if verbose > 0:
        print('Model loaded successfully.')

    return model
