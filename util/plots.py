"""
File containing the plots to visualize data and results
You can plot time series or scatter plots, with some helpers
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def _plot_helpers(**kwargs):
    """
    Helper function that can take various arguments and customize the plots. Also defines
    the figure and make it nice for jupyter notebooks

    Args:
        **kwargs:   title, xlabel, ylabel
                    xdates: if this argument is passed (with an integer value corresponding to
                            the length of the time series to plot), then the x axis is
                            customized to look good with dates

    Returns:
        fig, ax of the plot

    """

    # Define the figure, with custom size for the use in Jupyter Notebook
    fig, ax = plt.subplots(figsize=(16, 9))

    # Define the title if there is any
    if 'title' in kwargs.keys():
        if kwargs['title'] is not None:
            plt.title(kwargs['title'], size=25)

    # Custom x axis
    if 'xlabel' in kwargs.keys():
        if kwargs['xlabel'] is not None:
            plt.xlabel(kwargs['xlabel'], size=20)

    # Custom y axis
    if 'ylabel' in kwargs.keys():
        if kwargs['ylabel'] is not None:
            plt.ylabel(kwargs['ylabel'], size=20)

    # Custom formatter to have a nice x axis if it corresponds to the time
    if 'xdates' in kwargs.keys():
        fig.autofmt_xdate()
        if kwargs['xdates'] >= 7:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %h-%d'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%h-%d %H:%M'))
        plt.xlabel("Time", size=20)

    # Limits for the axes
    if 'xlim' in kwargs.keys():
        plt.xlim(kwargs['xlim'])
    if 'ylim' in kwargs.keys():
        plt.ylim(kwargs['ylim'])

    # To plot the grid
    if 'grid' in kwargs.keys():
        plt.grid(kwargs['grid'])

    # Make the ticks bigger
    plt.yticks(size=15)
    plt.xticks(size=15)

    # Return the fig to work with it
    return fig, ax


def _save_or_show(save_name: bool = None, show: bool = True, save: bool = False,
                  save_path: str = "../saves/Figures/", **kwargs) -> None:
    """
    Helper function called at the end of other function to show the plot, save it and close it

    Args:
        save_name:  Name to save the figure under
        show:       Show the plot or not
        save:       Save the plot or not
        save_path:  Where to save it
    """

    # To plot the legend
    if kwargs is not None:
        if 'legend' in kwargs.keys():

            # Special case
            if kwargs['legend'] == "Battery predictions":
                plt.legend(markerscale=7, prop={'size': 15})

            # General case
            else:
                plt.legend(prop={'size': 15})

    # If you want to save the figure
    if save:

        # Ensure a save name if defined, otherwise put a default one to still save the plot
        if save_name is None:
            save_name = "Default"

        if save_name == "Default":
            print('Warning: no saving name was given - defined as Default')

        # Define the path to the file and save the figure there
        plt.savefig(save_path+save_name)

    # If you want to show the plot
    if show:
        plt.show()

    # Close the plot
    plt.close()


def plot_time_series(data, sensors=None, show: bool = True, save: bool = False, **kwargs) -> None:
    """
    Function to plot a particular sensor time series

    Args:
        data: the DataFrame containing all the time series (or DataSet)
        sensors: either the column index or the column name to be plotted, to put in a list
                    if several are wanted
        show: flag to show the plot
        save: flag to save the plot
        kwargs: xlabel, ylabel, title

    Returns:

    """
    print('Reza')
    # Define the plot using the helpers
    if type(data) == pd.DataFrame:
        kwargs['xdates'] = (data.index[-1] - data.index[0]).days
    elif type(data) == list:
        kwargs['xdates'] = (data[0].index[-1] - data[0].index[0]).days
    _, _ = _plot_helpers(**kwargs)

    # Check if the name or the column of the sensor is given, and plot the values and the label
    # Case of the column number
    if type(sensors) == int:
        plt.plot(data.index, data.iloc[:, sensors])
        plt.ylabel(data.columns[sensors], size=20)

        # Save and/or show the plot
        _save_or_show(
            save_name=f"{data.index[0].strftime('%h-%d-%Hh%M')}__{data.index[-1].strftime('%h-%d-%Hh%M')}__{sensors}",
            show=show,
            save=save)

    # Case when the name is given
    elif type(sensors) == str:
        plt.plot(data.index, data.loc[:, sensors])
        plt.ylabel(sensors, size=20)

        # Save and/or show the plot
        _save_or_show(
            save_name=f"{data.index[0].strftime('%h-%d-%Hh%M')}__{data.index[-1].strftime('%h-%d-%Hh%M')}__{sensors}",
            show=show,
            save=save)

    # Otherwise, one can pass a list of sensors to plot
    elif type(sensors) == list:
        for sensor in sensors:
            if type(sensor) == int:
                plt.plot(data.index, data.iloc[:, sensor], label=data.columns[sensor])
            elif type(sensor) == str:
                plt.plot(data.index, data.loc[:, sensor], label=sensor)

        # Save and/or show the plot
        _save_or_show(
            save_name=f"{data.index[0].strftime('%h-%d-%Hh%M')}__{data.index[-1].strftime('%h-%d-%Hh%M')}__{sensors}",
            show=show,
            save=save,
            legend=True)

    # Otherwise, one can give a list of time series to plot
    elif sensors is None:
        if type(data) == list:
            if type(data[0]) == pd.Series:
                # Name to save the figure
                name = ""
                for series in data:
                    plt.plot(series.index, series.values, label=series.name)
                    name += series.name

                # Save and/or show the plot
                _save_or_show(
                    save_name=f"{data[0].index[0].strftime('%h-%d-%Hh%M')}__{data[0].index[-1].strftime('%h-%d-%Hh%M')}__{name}",
                    show=show,
                    save=save,
                    legend=True)

        elif type(data) == pd.DataFrame:
            # Name to save the figure
            name = ""
            for column in data.columns:
                plt.plot(data.index, data[column].values, label=column)
                name += column

            # Save and/or show the plot
            _save_or_show(
                save_name=f"{data.index[0].strftime('%h-%d-%Hh%M')}__{data.index[-1].strftime('%h-%d-%Hh%M')}__{name}",
                show=show,
                save=save,
                legend=True)

    # Not implemented cases
    else:
        raise ValueError(f"Type {type(sensors)} is not valid for the plot")



def scatter_plot(datax: np.ndarray, datay: np.ndarray, xlabel=None, ylabel=None, show: bool = True,
                 save: bool = False, **kwargs) -> None:
    """
    Function to create a scatter plot

    Args:
        datax: data of the x axis
        datay: data of the y axis
        xlabel: label of the x axis
        ylabel: label of the y axis
        show: flag to show the plot
        save: flag to save the plot
        kwargs: xlabel, ylabel, title
    """
    # Define the plot using the helpers
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    _, _ = _plot_helpers(**kwargs)

    # Scatter plot
    plt.scatter(datax, datay)

    # Save and/or show the plot
    _save_or_show(save_name=f"scatter_plot__{xlabel}__{ylabel}",
                  show=show,
                  save=save)


def battery_scatter_plot(data) -> None:
    """
    Function to plot the battery scatter plot of the power vs the SoC difference

    Args:
        data: a DataSet containing the battery information
    """

    # Find the columns corresponding to the wanted information
    try:
        power_col = np.where(data.data.columns == 'Battery power measurement')[0][0]
        soc_col = np.where(data.data.columns == 'Battery SoC')[0][0]
    except:
        raise ValueError("The data doesn't contain the battery measurements - or the columns\
                            are named unexpetedly")

    # Compute the average power observed between two time steps
    power = (data.data.iloc[1:, power_col].values + data.data.iloc[:-1, power_col].values) / 2
    # Compute the difference of SoC between two time steps
    SoC_diff = data.data.iloc[1:, soc_col].values - data.data.iloc[:-1, soc_col].values

    # Scatter plot
    scatter_plot(power, SoC_diff, xlabel='Power', ylabel='Delta Soc', title="Battery measurements")


def histogram(data, bins, save_name: str = 'Default', show: bool = True, save: bool = False, **kwargs) -> None:
    """
    Function to create a scatter plot

    Args:
        data: data to plot
        bins: bins specifications, either an integer or a sequence, see matplotlib
        save_name: name to save the histogram to if wanted
        show: flag to show the plot
        save: flag to save the plot
        kwargs: xlabel, ylabel, title, ...
    """
    # Define the plot using the helpers
    if "ylabel" in kwargs.keys():
        pass
    else:
        kwargs['ylabel'] = "Number of occurences"
    _, _ = _plot_helpers(**kwargs)

    # Scatter plot
    plt.hist(data, bins)

    # Save and/or show the plot
    _save_or_show(save_name=f"histogram_{save_name}",
                  show=show,
                  save=save)