"""
File to run in terminal. Its main purpose is that tensorboard only works there not in jupyter (Loris)
"""

import sys
# To make the modules visible
sys.path.append("../")

from models.data_handling import NESTData
from models.NN_models import NNModel, NNExtractionModel, LSTMExtractionModel
from models.recurrent import LSTMModel

from parameters import data_kwargs, model_kwargs


def prepare_model(data_kwargs=data_kwargs, model_kwargs=model_kwargs):

    # Create the data in the right form
    nest_data = NESTData(data_kwargs)

    if model_kwargs["model_name"] == "LSTMModel":
        model = LSTMModel(nest_data=nest_data,
                          model_kwargs=model_kwargs)

    elif model_kwargs["model_name"] == "NNModel":
        model = NNModel(nest_data=nest_data,
                        model_kwargs=model_kwargs)

    elif model_kwargs["model_name"] == "NNExtractionModel":
        model = NNExtractionModel(nest_data=nest_data,
                                  model_kwargs=model_kwargs)

    elif model_kwargs["model_name"] == "LSTMExtractionModel":
        model = LSTMExtractionModel(nest_data=nest_data,
                                    model_kwargs=model_kwargs)

    else:
        raise ValueError(f"Unknown model type {model_kwargs['model_name']}")

    return model


if __name__ == "__main__":

    model = prepare_model(data_kwargs=data_kwargs,
                          model_kwargs=model_kwargs)

    model.fit(n_epochs=5, save_checkpoints=True)
    model.predictions_analysis()

    model.plot_predictions(model.validation_sequences[2])


