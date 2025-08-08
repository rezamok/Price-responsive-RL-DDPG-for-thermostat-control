"""
File containing the models for the battery as well as a complete pipeline function creating
and fitting a model to the wanted data
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.linear_model import LinearRegression

from models.helpers import name_factory, load_model, save_model
from util.plots import _plot_helpers, _save_or_show

from models.base_models import BaseModel
from models.data_handling import BatteryData

from models.parameters import model_kwargs


path = "saves" if sys.platform == "win32" else os.path.join("..", "saves")

class BatteryModel(BaseModel):
    """
    Base class for the battery models with some general functions common to all models
    """

    def __init__(self, battery_data: BatteryData, model_kwargs: dict = model_kwargs) -> None:
        """
        Initialization of the battery model, which is link to some data, provided by the BatteryData
        class.
        Note that the "data" attribute is a DataFrame that has been processed from the DataSet to be
        in the wanted form, relating power inputs to SoC changes (done during the initialization
        of the BatteryData)

        Args:
            battery_data: data to link to the model
        """

        # Initialize a general model
        super().__init__(dataset=battery_data.dataset,
                         data=battery_data.data,
                         differences=None,
                         model_kwargs=model_kwargs)

        # At the initialization stage, all attributes are undefined or False
        self.is_fitted = False
        self.residuals_mean = None
        self.residuals_std = None

    def compute_residuals(self) -> None:
        """
        Function to compute the residuals between the model predictions and the actual observed
        values
        """

        # Compute the residuals, i.e. the difference between the observed and predicted SoC differences
        self.data["SoC Residuals"] = self.data["Delta SoC"] - self.predict(x=self.data["Average power"],
                                                                           with_noise=False)

        # Save the mean and std of the residuals for later
        self.residuals_mean = self.data["SoC Residuals"].mean()
        self.residuals_std = self.data["SoC Residuals"].std()

    def clean_residuals(self) -> None:
        """
        Function to further clean the data: remove extreme residuals, i.e. datapoints that
        were very wrongly predicted by the model
        """

        # If the mean and std of residuals aren't computed yet, do it
        if (self.residuals_mean is None) | (self.residuals_std is None):
            self.compute_residuals()

        # Get the indices where the residuals are more than 3 std away from the mean
        to_drop = np.where((self.data["SoC Residuals"] > self.residuals_mean + 3 * self.residuals_std) |
                           (self.data["SoC Residuals"] < self.residuals_mean - 3 * self.residuals_std))[0]

        # Drop these indices
        self.data.drop(index=self.data.index[to_drop], inplace=True)

        # Compute the new residuals
        self.compute_residuals()

    def _battery_plots_helpers(self, **kwargs):
        """
        Plot helpers for the battery model
        """

        # Define the kwargs specific to the battery plot
        xlabel = "Power"
        ylabel = "Delta Soc"
        legend = True

        # Use the custom plot helpers to design the plot
        return _plot_helpers(xlabel=xlabel,
                             ylabel=ylabel,
                             legend=legend,
                             **kwargs)

    def battery_model_plot(self, show: bool = True, save: bool = False, **kwargs) -> None:
        """
        Function to plot both the measurements and the predictions of the model

        Args:
            Typical arguments for plotting, the kwargs can be seen in the util.plots file
            (xlabel, ylabel, title, ...)
        """

        # Design the plot
        _, _ = self._battery_plots_helpers(title="Battery model", **kwargs)

        # Scatter plot the actual measurements
        plt.scatter(self.data["Average power"], self.data["Delta SoC"], label="Measurements")

        # Create a range of values from the minimum to the maximum observed power
        x = np.arange(min(self.data["Average power"]), max(self.data["Average power"]), 0.01)

        # use the model to predict the SoC changes along the defined range
        y = self.predict(x, with_noise=False)

        # plot the model predictions
        plt.plot(x, y, color='red', label="Model")

        # Save and/or show the plot
        _save_or_show(save_name=f"scatter_plot__battery_model",
                      show=show,
                      save=save,
                      legend=True)

    def battery_predictions_plot(self, show: bool = True, save: bool = False, **kwargs) -> None:
        """
        Function to plot both the measurements and the predictions of the model

        Args:
            Typical arguments for plotting, the kwargs can be seen in the util.plots file
            (xlabel, ylabel, title, ...)
        """

        # Design the plot
        _, _ = self._battery_plots_helpers(title="Battery predictions",
                                           **kwargs)

        # Use the model to predict the SoC changes along the defined range
        predictions = self.predict(self.data["Average power"], with_noise=True)

        # Plot the model predictions
        plt.plot(self.data["Average power"], predictions, color='red', label="Predictions",
                 alpha=0.25, linestyle='', marker='o', markersize=1.5)

        # Scatter plot the actual measurements
        plt.plot(self.data["Average power"], self.data["Delta SoC"], label="Measurements",
                 alpha=0.25, linestyle='', marker='o', markersize=1.5)

        # Save and/or show the plot
        _save_or_show(save_name=f"scatter_plot__battery_predictions",
                      show=show,
                      save=save,
                      legend="Battery predictions")


class LinearBatteryModel(BatteryModel):
    """
    Linear battery models: fits a model f(x) = ax + b
    """

    def __init__(self, battery_data: BatteryData, model_kwargs: dict = model_kwargs) -> None:
        """
        Initialization of the linear model.
        Note that the model is immediately fitted to the data

        Args:
            data: BatteryData to link the model to
        """

        model_kwargs["model_name"] = "LinearBatteryModel"
        # Create a general BatteryModel
        super().__init__(battery_data=battery_data,
                         model_kwargs=model_kwargs)

        # Fit the model and recall the slope and intercept (a and b)
        # Additionally, recall the LinearRegression itself from sklearn, which provides nice features
        # such as a prediction function
        self.a, self.b, self.regression = self.fit()

        # Plot the created model
        self.battery_model_plot()

    def fit(self):
        """
        Function to fit the data, using sklearn's LinearRegression.
        Note that a first regression is done, after which values that yielded extreme residuals
        are cleaned, before the model is refitted to this data to have a better approximation
        and a better residuals structure
        """

        # Use sklearn to fit an affine function to the data
        regression = LinearRegression().fit(self.data["Average power"].values.reshape(-1, 1),
                                            self.data["Delta SoC"])

        # Print the "goodness of fit" of the model
        print(f"Linear fit with score "
              f"{regression.score(self.data['Average power'].values.reshape(-1, 1), self.data['Delta SoC']):.3f}")

        # Momentarily save the regression model to be able to plot the model performance
        self.regression = regression
        self.battery_model_plot()

        # Remove the extreme residuals (more than 3 std away from the mean)
        print("Removing extreme residuals and refitting.")
        n_datapoints = len(self.data)
        self.clean_residuals()

        # Print how many datapoints this removed
        print(f"Deleted {n_datapoints-len(self.data)} datapoints that yielded extreme residuals, "
              f"{(n_datapoints-len(self.data))/n_datapoints :.2f}% of the data.")

        # Fit the model again, this time saving the slope and intercept as well
        regression = LinearRegression().fit(self.data["Average power"].values.reshape(-1, 1),
                                            self.data["Delta SoC"])

        # Remove the extreme residuals (more than 3 std away from the mean)
        print("Removing extreme residuals and refitting.")
        n_datapoints = len(self.data)
        self.clean_residuals()

        # Print how many datapoints this removed
        print(f"Deleted {n_datapoints - len(self.data)} datapoints that yielded extreme residuals, "
              f"{(n_datapoints - len(self.data)) / n_datapoints :.2f}% of the data.")

        # Fit the model again, this time saving the slope and intercept as well
        regression = LinearRegression().fit(self.data["Average power"].values.reshape(-1, 1),
                                            self.data["Delta SoC"])

        a = regression.coef_[0]
        b = regression.intercept_

        # Print the new "Goodness of fit of the model"
        print(f"Linear fit with score "
              f"{regression.score(self.data['Average power'].values.reshape(-1, 1), self.data['Delta SoC']):.3f}")

        # The model is now fitted, we can save this info and return the quantities of interest
        self.is_fitted = True
        return a, b, regression

    def predict(self, x, with_noise: bool = True):
        """
        Function to make predictions using the model, using sklearn again
        Args:
            x:          The power input for which the change in SoC is wanted
            with_noise: Flag to return a noisy prediction or not
        """
        # sklearn demands a numpy array as input, so we need to check that
        # If we input a single value to be predicted, make sure it is in the numpy format
        if (type(x) == int) | (type(x) == float) | (type(x) == np.float64):
            x = np.array(np.float64(x)).reshape(-1, 1)
            size = 1
        # If we want to predict a Series, transform it into an array
        if type(x) == pd.Series:
            x = x.values
            size = len(x)

        # Compute the predictions using sklearn
        predictions = self.regression.predict(x.reshape(-1, 1))

        # If the pure model is wanted, return the predictions
        if not with_noise:
            return predictions

        # Add some noise fitted to the data
        else:
            # If the mean and std of residuals aren't computed yet, do it
            if (self.residuals_mean is None) | (self.residuals_std is None):
                self.compute_residuals()

            # Compute one noise realisation
            noise = np.random.normal(loc=self.residuals_mean,
                                     scale=self.residuals_std,
                                     size=size)

            # Return a noisy prediction
            return predictions + noise


def prepare_linear_battery_model(data_kwargs, model_name: str = "LinearBatteryModel",
                                 model_kwargs: dict = model_kwargs,
                                 save_path=os.path.join(path, "Models"), show_model: bool = True,
                                 show_predictions: bool = True) -> LinearBatteryModel:
    """
    Function to fully pipeline the battery model, returning a fitted model, either
    downloading it or fitting it on the spot if needed.

    Args:
        data_kwargs:        See model.helpers, several parameters defined
        model_name:         Name of the model
        save_path:          Path to save and loads models to and from
        show_model:         Flag to plot the (pure) model against the measurements
        show_predictions:   Flag to show one realisation of the noisy model agains the data

    Returns:
        The fitted model
    """

    # Create the data in the right form
    battery_data = BatteryData(data_kwargs)

    # Create the name associated to the model
    save_name = name_factory(data=battery_data,
                             model_kwargs=model_name)

    # Try to load the model
    try:
        model = load_model(name=save_name,
                           save_path=save_path)

        # Check that the model is fitted - this should not fail under the current implementations
        assert model.is_fitted, "The model is not fitted, you might need to do it"

    # Otherwise, create the model (which also fits it)
    except AssertionError:
        model = LinearBatteryModel(battery_data=battery_data,
                                   model_kwargs=model_kwargs)

        model.data.drop(columns=['Battery power input', 'Battery power measurement', 'SoC Residuals'], inplace=True)

        # Save the model for next time
        save_model(model=model,
                   name=save_name,
                   save_path=save_path)

    # Plot the model performance
    if show_model:
        model.battery_model_plot()

    # Plot some realisation of the model with noise
    if show_predictions:
        model.battery_predictions_plot()
        model.battery_predictions_plot(xlim=(-25, 25), ylim=(-10, 10))

    # Return the model
    return model
