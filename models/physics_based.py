"""
File containing physics-based models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import math

from datetime import timedelta
from torch.nn.utils.rnn import pad_sequence

from data.preprocessing import correct_interval
from data.dataset import prepare_dataset, create_time_data
from util.functions import compute_solar_irradiation_window
from util.util import save_data, load_data
from models.util import get_rc_model_control_from_sequence
from util.plots import _plot_helpers, _save_or_show

from parameters import DATA_SAVE_PATH



class Room3R1C():
    """
    inspired from Antoon's Master Thesis.

    Models one room with 1 capacitance and 4 disturbances: the outside temperature, irradiation, and
    neighbouring room temperature.

    Here we take real data from UMAR and typically model room 274 or 272 with the neighbouring room 273
    and actual weather data.

    The parameters are found using least squares and the Euler method.
    """

    def __init__(self, data_kwargs: dict, room: int, force_preprocess: bool = False, threshold: int = 48 * 60,
                 testing_proportion: float = 0.25, overlapping_distance: int = 60):
        """
        Function to initialize the model, in particular defining the data to be used

        Args:
            data_kwargs:            Args of the data from `parameters.py`
            room:                   Room to model, 272 or 274
            force_preprocess:       Whether to repreprocess the data
            threshold:              Threshold length of sequences to predict with the model
            testing_proportion:     Proportion of data to use for testing
            overlapping_distance:   Distance for overlapping sequences in the test sequences
        """

        assert room in [272, 274], f'Only rooms 272 and 274 can be modeled, not {room}'

        self.starts = []
        self.ends = []
        self.train_starts = []
        self.train_ends = []
        self.test_starts = []
        self.test_ends = []
        # To recall when the physics-based model doesn't work, so the black-box model doesn't
        # learn that
        self.breaks = []

        self.p = None
        self.data_predicted = None

        self.threshold_length = threshold
        self.interval = data_kwargs['interval']
        self.overlapping_distance = overlapping_distance

        self.dataset, self.data = self.prepare_data(data_kwargs=data_kwargs, room=room,
                                                    force_preprocess=force_preprocess)

        self.create_sequences(threshold=self.threshold_length)
        self.train_test_separation(testing_proportion=testing_proportion)

    def prepare_data(self, data_kwargs, room: int, force_preprocess: bool = False) -> pd.DataFrame:
        """
        Function to prepare the data: use the tools from `src/data` to download and preprocess data
        from NEST, then adapt it to the physics-based model

        Args:
            data_kwargs:        Args of the data from `parameters.py`
            room:               Room to model, 272 or 274
            force_preprocess:   Whether to repreprocess the data
        """

        # Download the data, force_preprocess might be needed. The `physics_based_data` arguments
        # tells the function to only load useful data (i.e. weather and room data)
        dataset = prepare_dataset(data_kwargs=data_kwargs,
                                  force_preprocess=force_preprocess,
                                  data_analysis=False,
                                  clean_data=True,
                                  physics_based_data=True)

        try:
            print('\nLooking for the data...')
            data = load_data(save_name=dataset.save_name + '_data',
                             save_path=DATA_SAVE_PATH)
            self.p = np.load(os.path.join(DATA_SAVE_PATH, dataset.save_name + '_p.npy'))
            print('Found!')

        except AssertionError:
            print('Nohting found, preprocessing it...')
            # Columns of the data to keep, renaming them for simplicity
            columns = ['Weather outside temperature', 'Weather solar irradiation', f'Power room {room}',
                       f'Thermal temperature measurement {room}', 'Thermal temperature measurement 273',
                       'Case', f'Thermal valve {room}']

            data = dataset.data[[x for x in dataset.data.columns if x in columns]].copy()

            data.rename(columns={'Weather outside temperature': 'T_out', 'Weather solar irradiation': 'Q_irr',
                                 f'Power room {room}': 'Q_heat',
                                 f'Thermal temperature measurement {room}': 'T',
                                 'Thermal temperature measurement 273': 'T_1', f'Thermal valve {room}': 'valve'}, inplace=True)

            assert ((data.index[1] - data.index[0]).seconds // 60) % 60 == 1, \
                f'The sampling interval is not 1 minute but {((data.index[1] - data.index[0]).seconds // 60) % 60}'

            # The heating/cooling power is given in kW, transform it in W
            data['Q_heat'] *= 1000

            # transform the solar irradiation into the actual irradiation on the windows
            data['Q_irr_original'] = data['Q_irr']
            data['Q_irr'] = compute_solar_irradiation_window(data=data['Q_irr'])

            # Add the missing first timestep (to start at midnight and not 00:00:01)
            data = data.append(pd.Series(data.iloc[0, :], name=data.index[0] - timedelta(minutes=1))).sort_index()

            # Save the data
            save_data(data=data,
                      save_name=dataset.save_name + '_data',
                      save_path=DATA_SAVE_PATH)

        return dataset, data

    def create_sequences(self, threshold: int = 48 * 60) -> None:
        """
        Function to create lists of starts and ends of sequences of data that can be used (i.e.
        there is no missing data).

        Args:
            threshold:          Threshold on the sequences length (we use Euler's method, it cannot
                                  be extrapolated to too long sequences)
        """

        # Find the places with missing values
        nans = np.where(pd.isnull(self.data).any(axis=1))[0]

        # Define the starts and ends of sequences of data without missing values
        starts_ = list(nans[np.where(np.diff(nans) != 1)[0]] + 1)
        ends_ = list(nans[np.where(np.diff(nans) != 1)[0] + 1])

        # Go through the sequences and separate the ones that are too long in several chunks
        starts = []
        ends = []
        for start, end in zip(starts_, ends_):
            # Cut the sequence in chunk of `threshold` length
            while end - start > threshold:
                starts.append(start)
                ends.append(start + threshold + 1)
                start += threshold
                # Recall the breaks, as the data will be inconsistent
                self.breaks.append(start + threshold + 1)
            starts.append(start)
            ends.append(end)

        # Small correction to handle the start and end of the data correctly
        if nans[0] > 0:
            starts = [0] + starts
            ends = [nans[0]] + ends
        if nans[-1] < len(self.data) - 1:
            starts = starts + [nans[-1] + 1]
            ends = ends + [len(self.data) - 1]

        self.starts = starts
        self.ends = ends

    def train_test_separation(self, testing_proportion: float = 0.25) -> None:
        """
        Function to separate the data in training and testing sets.

        Args:
            testing_proportion: Proportion of the data to put in testing
        """

        # Separation in training and testing
        self.train_starts = self.starts[:-int(len(self.starts) * testing_proportion)]
        self.test_starts = self.starts[-int(len(self.starts) * testing_proportion):]

        self.train_ends = self.ends[:-int(len(self.starts) * testing_proportion)]
        self.test_ends = self.ends[-int(len(self.starts) * testing_proportion):]

        # Adapt test sequences so they can overlap a bit
        starts = []
        ends = []
        running = False
        for x, y in zip(self.test_starts, self.test_ends):
            if y - x >= self.threshold_length + 1:
                if not running:
                    starts.append(x)
                    running = True
            else:
                if running:
                    ends.append(y)
                    running = False

        self.test_starts = []
        self.test_ends = []

        for start, end in zip(starts, ends):
            self.test_starts += [start + self.overlapping_distance * x for x in
                                 range(math.ceil((end - start - self.threshold_length) / self.overlapping_distance))]
            self.test_ends += [min(start + self.threshold_length + self.overlapping_distance * x + 1, end) for x in
                               range(math.ceil((end - start - self.threshold_length - 1) / self.overlapping_distance))]

    def fit(self) -> None:
        """
        Function to do the system identification, i.e. fit the parameters of the RC model.
        """

        print('\nFitting the model...')

        # Put the data together, mostly as differences
        X = self.data[['Q_heat', 'Q_irr']].iloc[:-1, :].copy()
        X['dTout'] = self.data['T_out'] - self.data['T']
        X['dT1'] = self.data['T_1'] - self.data['T']

        # 60 seconds for one time step
        X['dT'] = np.diff(self.data['T']) / 60

        # We only fit the data on the training part
        train_X = X.iloc[:self.train_ends[-1], :]

        # We cannot handle missing values with the Least Squares (matrix multiplication)
        train_X.dropna(how='any', inplace=True)

        ## Sys ID using least squares to find the parameters `p`
        # (X^T * X) ^-1 * X^T * Y
        self.p = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(train_X.iloc[:, :-1].values),
                                                             train_X.iloc[:, :-1].values)),
                                     np.transpose(train_X.iloc[:, :-1].values)), train_X['dT'].values)

        self.C = 1 / self.p[0]
        self.k = self.C * self.p[1]
        self.R_ext = 1 / (self.p[2] * self.C)
        self.R_1 = 1 / (self.p[3] * self.C)

        print(f'Parameters found:\nC = {self.C:.3f}\nk = {self.k:.3f}\nR_ext = {self.R_ext:.3f}\nR_1 = {self.R_1:.3f}')

    def predict(self, start: int, end: int, control: np.array = None, from_predicted: bool = False):
        """
        Function to predict the temperature over a given sequence of data (start, end)

        Args:
            start:          Start of the sequence
            end:            End of the sequence
            control:        Control sequence if another sequence than what is in the data is to be taken
            from_predicted: To take the predicted data as starting point
        """

        if control is None:
            control = self.data['Q_heat'][start: end].values

        # The temperature starts at the right value
        T = np.zeros(end - start)

        if from_predicted:
            T[0] = self.data_predicted['T'][start]
        else:
            T[0] = self.data['T'][start]

        # Euler method for integration with 60 seconds time steps
        for t in range(end - start - 1):
            time = self.data.index[start] + timedelta(minutes=t)
            T[t + 1] = T[t] + 60 * (control[t] * self.p[0] +
                                    self.data.loc[time, 'Q_irr'] * self.p[1] +
                                    (self.data.loc[time, 'T_out'] - T[t]) * self.p[2] +
                                    (self.data.loc[time, 'T_1'] - T[t]) * self.p[3])

        return T

    def error_analysis(self, return_: bool = False, number: int = 100, horizon: int = None):
        """
        Function to analyze the goodness of fit of the model parameters on the testing sequences.

        Args:
            return_:    Flag to put to `True` if the errors are wanted in numbers, otherwise plots it.
            number:     Number of sequences to analyze
            horizon:    Horizon of the predictions to analyze
        """

        if horizon is None:
            horizon = self.threshold_length

        seqs = []
        for j, (start, end) in enumerate(zip(self.test_starts, self.test_ends)):
            if end-start >= horizon + 1:
                seqs.append((start, end))

        np.random.shuffle(seqs)
        seqs = seqs[:number]

        print(f'\nAnalyzing the errors over {len(seqs)} sequences')

        errors = []
        for j, seq in enumerate(seqs):
            start, end = seq
            if j % 25 == 24:
                print(f'{j+1} done...')

            # Predict the temperature
            T = self.predict(start=start, end=end)

            # Store the error
            errors.append(np.abs(self.data['T'][start: end] - T))

        if return_:
            return errors

        else:
            # Compute the Absolute Error, as well as the number of sequences used to compute it
            aes = []
            numbers = []
            # Run through the prediction steps
            for j in range(horizon + 1):
                ae = []
                number = 0
                # Each time run through all the errors (when possible)
                for error in errors:
                    # Some sequences are shorter and have to be discarded
                    if len(error) > j:
                        ae.append(error[j])
                        number += 1
                aes.append(ae)
                numbers.append(number)

            # Compute the mean and std of the Absolute Error
            mean = np.array([np.mean(mae) for mae in aes])
            std = np.array([np.std(mae) for mae in aes])
            numbers = np.array(numbers)

            # Plot the evolution of the rrors over the prediction horizon
            fig = plt.figure(figsize=(16, 9))
            plt.fill_between(np.arange(horizon + 1) / 60, mean + std,
                             [max(x - y, 0) for x, y in zip(mean, std)], alpha=0.25, color='blue')
            plt.fill_between(np.arange(horizon + 1) / 60, mean + 2 * std,
                             [max(x - 2 * y, 0) for x, y in zip(mean, std)], alpha=0.25, color='blue')
            plt.plot(np.arange(horizon + 1) / 60, mean, color='blue', linewidth=3)
            plt.plot(np.arange(horizon + 1) / 60, numbers / numbers[0], color='black')
            plt.xticks(size=15)
            plt.xlabel('Hour ahead prediction', size=20)
            plt.yticks(size=15)
            plt.ylabel('Absolute Error', size=20)
            plt.title('Error propagation', size=25)
            plt.show()

            # Print the information
            print(f'Mean absolute error after:\n 1h: {mean[60]:.3f} K   ({numbers[60]} trajectories)\n'
                  f' 6h: {mean[360]:.3f} K   ({numbers[360]} trajectories)\n'
                  f'12h: {mean[720]:.3f} K   ({numbers[720]} trajectories)')
            if horizon >= 24*60:
                print(f'24h: {mean[1440]:.3f} K   ({numbers[1440]} trajectories)')
            if horizon >= 48*60:
                print(f'48h: {mean[48*60]:.3f} K   ({numbers[48*60]} trajectories)')

            errors_ = [z for y in aes for z in y]
            for q in [0.5, 0.75, 0.9, 0.95, 0.975]:
                print('\nQuantiles:', end="")
                print(f" | {q}: {np.quantile(errors_, q=q):.3f}", end="")

    def plot_prediction(self, start: int, end: int):
        """
        Function to plot and compare the temperature predictions to real data.

        Args:
            start:  Start of the sequence
            end:    End of the sequence
        """

        # Predict the temperature
        T = self.predict(start=start, end=end)

        fig, ax = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        ax[0].plot(self.data.index[start:end], T, color='blue')
        ax[0].plot(self.data['T'][start:end], color='black')
        ax[0].set_ylabel('Temperature', size=20)
        ax[0].set_title('Predictions', size=25)

        ax[1].plot(self.data.iloc[start:end, :]['Q_heat'], color='blue')
        ax[1].set_ylabel('Power', size=20)

        for i in range(2):
            ax[i].tick_params(axis='x', which='major', labelsize=15)
            ax[i].tick_params(axis='y', which='major', labelsize=15)
            ax[i].set_xlabel('Time', size=20)
        fig.autofmt_xdate()
        plt.show()

    def compare_control_sequences(self, start: int, end: int, controls: np.array = None):
        """
        Function to plot and compare the temperature predictions to real data.

        Args:
            start:      Start of the sequence
            end:        End of the sequence
            controls:   Control sequences to apply
        """

        colors = ['blue', 'red', 'orange', 'green']

        if controls is None:
            controls = np.zeros((end - start - 1, 4))

            X_ = self.data.iloc[start: end, :].copy()

            sep1 = np.where(X_['Q_heat'][:-1].cumsum() / X_['Q_heat'][:-1].sum() > 1 / 3)[0][0]
            sep2 = np.where(X_['Q_heat'][:-1].cumsum() / X_['Q_heat'][:-1].sum() > 2 / 3)[0][0]

            controls[:, 0] = X_['Q_heat'][:-1].copy()
            controls[:sep1, 1] = X_['Q_heat'][:-1][:sep1]
            controls[sep1: sep2, 2] = X_['Q_heat'][:-1][sep1: sep2]
            controls[sep2:, 3] = X_['Q_heat'][:-1][sep2:]

        T = np.zeros((end - start, controls.shape[1]))

        for i in range(controls.shape[1]):
            T[:, i] = self.predict(start=start, end=end, control=controls[:,i])

        fig, ax = plt.subplots(2, 1, figsize=(20, 16), sharex=True)
        ax[0].set_ylabel('Temperature', size=20)
        ax[1].set_ylabel('Power', size=20)
        ax[0].set_title('Control sequences comparison', size=25)

        ax[0].plot(self.data['T'][start:end], color='black')

        for i in range(controls.shape[1]):
            ax[0].plot(X_.index, T[:, i], color=colors[i], lw=2)
            ax[1].plot(X_.index[:-1], controls[:, i], color=colors[i], lw=2)

        for i in range(2):
            ax[i].tick_params(axis='x', which='major', labelsize=15)
            ax[i].tick_params(axis='y', which='major', labelsize=15)
            ax[i].set_xlabel('Time', size=20)
        fig.autofmt_xdate()
        plt.show()

    def predict_all_data(self, interval: int = 15, use_real_data: bool = False):
        """
        Function to predict all the room temperatures in the data - used to change real temperatures by
        the ones from the model. This function is subsequently used by black-box models to learn
        the physics-based model instead of the true values.

        After the prediction, we can save the new data, as well as a version with larger intervals
        to be used for the black-box models.

        Args:
            interval:   Interval to correct the data to before saving (to use afterwards with black-box models)
            use_real_data:  Flag whether to use the real data instead of th data generated by the model
        """

        print(f'\nPredicting {len(self.starts)} sequences...')

        np.save(file=os.path.join(DATA_SAVE_PATH, self.dataset.save_name + '_p'), arr=self.p)

        # Longer interval version: correct the data interval
        if use_real_data:
            df = correct_interval(self.data, interval=interval)
        else:
            try:
                self.data_predicted = load_data(save_name=self.dataset.save_name + '_predicted',
                                                save_path=DATA_SAVE_PATH)
            except AssertionError:
                self.data_predicted = self.data.copy()
                # Loop over all the sequences to predict
                for j, (start, end) in enumerate(zip(self.starts, self.ends)):
                    if j % 25 == 24:
                        print(f'{j + 1} done...')

                    T = self.predict(start=start, end=end)

                    self.data_predicted.loc[self.data_predicted.index[start]: self.data_predicted.index[end - 1], 'T'] = T

                # Save the data and the parameters
                save_data(data=self.data_predicted,
                          save_name=self.dataset.save_name + '_predicted',
                          save_path=DATA_SAVE_PATH)

                df = correct_interval(self.data_predicted, interval=interval)

                # Put the breaks to NaN, to ensure the data is consistent - otherwise the physics-based model
                # predictions will jump there
                to_break = [int(np.round(break_ / 15)) for break_ in self.breaks]
                df.iloc[to_break, :] = np.nan

        df['Q_irr'] = df['Q_irr_original']
        df.drop(columns=['Q_irr_original'], inplace=True)
        #compute_solar_irradiation_window(data=df['Q_irr'], reverse=True)
        df = create_time_data(data=df)

        # Add two columns for the black-box models (they look for those)
        #df['Case'] = (df['Q_heat'] > 0) * 1 * 2 - 1
        #df['valve'] = - ((np.abs(df['Q_heat']) < 1e-6) * 1 - 1)

        # Rename some columns for the black-box models, again so they find what the search
        df.rename(columns={'Q_heat': 'Power room', 'T': 'Room temperature'}, inplace=True)

        # Save the data
        save_data(data=df,
                  save_name=self.dataset.save_name[:-1] + str(interval),
                  save_path=DATA_SAVE_PATH)


def compute_both_errors(rc_model, pcnn, sequences=None, number: int = 250):
    """
    Function to compute the absolute error of both the RC model and the PCNN on some data

    Args:
        rc_model:   The RC model
        pcnn:       The PCNN
        sequences:  Sequences of data (for the PCNN model) to compare
        number:     Number of predictions

    Returns:
        Absolute Errors of the RC model and PCNN in array (sequence x time step)
    """

    # Define needed constants
    if sequences is None:
        sequences = pcnn.validation_sequences
    interval = pcnn.dataset.interval

    # Take long enough sequences to have interesting predictions, and keep the wanted number
    seqs = [sequence for sequence in sequences if
            sequence[1] - sequence[0] - pcnn.warm_start_length >= pcnn.maximum_sequence_length]
    seqs = seqs[:number]
    print(f"Analyzing {len(seqs)} predictions...")


    # RC model errors computation
    rc_model_errors = []
    for j, sequence in enumerate(seqs):
        # Find the start and end in the corresponding data
        start = (sequence[0] + pcnn.warm_start_length) * interval - rc_model.data.index[
            (sequence[0] + pcnn.warm_start_length) * interval].minute % 15
        end = (sequence[1]) * interval - rc_model.data.index[
            (sequence[0] + pcnn.warm_start_length) * interval].minute % 15 + 1

        if j % 250 == 249:
            print(f'{j + 1} done...')

        # Predict the temperature
        T = rc_model.predict(start=start, end=end)[1:]

        # Store the error
        rc_model_errors.append(np.abs(rc_model.data['T'][start + 1: end] - T))

    # PCNN errors
    predictions, true_data = pcnn.scale_back_predictions(sequences=seqs)
    pcnn_errors = np.abs(predictions - true_data)

    return np.array(rc_model_errors), pcnn_errors[:, pcnn.warm_start_length:, 0]


def plot_error_propagation(rc_model, pcnn, rc_model_errors, pcnn_errors, paper: bool = False, scale: int = 1,
                           ticks: list = None, save: bool = True):
    """
    Function to plot the error propagation (MAE at each time step) of both models along the horizon,
    based on errors computed by `compute_both_errors`.

    Args:
        rc_model:           The RC model
        pcnn:               The PCNN
        rc_model_errors:    MAEs of the RC model
        pcnn_errors:        MAEs of the PCNN
        paper:              Flag to custom the plot for the paper
        scale:              Scale of the size of labels, title, ...
        ticks:              If needed, custom ticks for the x-axis
        save:               Flag to save the plot
    """

    # Define the figure with helpers
    fig, ax = _plot_helpers(figsize=(20, 12))

    # Plot the error bars (1 std) of both models
    plt.fill_between(np.arange(rc_model.threshold_length) + 1,
                     [max(0, x) for x in (np.mean(rc_model_errors, axis=0) - np.std(rc_model_errors, axis=0))],
                     (np.mean(rc_model_errors, axis=0) + np.std(rc_model_errors, axis=0)),
                     color='blue', alpha=0.2)
    plt.fill_between((np.arange(pcnn.maximum_sequence_length) + 1) * 15,
                     [max(0, x) for x in (np.mean(pcnn_errors, axis=0) - np.std(pcnn_errors, axis=0)).squeeze()],
                     (np.mean(pcnn_errors, axis=0) + np.std(pcnn_errors, axis=0)).squeeze(),
                     color='green', alpha=0.2)

    # Plot mean errors
    plt.plot(np.arange(rc_model.threshold_length) + 1, np.mean(rc_model_errors, axis=0), color='blue', label='RC model',
             lw=3)
    plt.plot((np.arange(pcnn.maximum_sequence_length) + 1) * 15, np.mean(pcnn_errors, axis=0), color='green',
             label='PCNN', lw=3)

    # Customizations
    if not paper:
        plt.xticks(size=15)
        plt.xlabel('Time step ahead prediction', size=20)
        plt.yticks(size=15)
        plt.ylabel('Absolute Error', size=20)
        plt.title('Error propagation', size=25)
        plt.legend(prop={'size': 20})
        plt.show()
    else:
        if ticks is None:
            ticks = [1, 12, 24, 48, 72]
        plt.xticks(size=15 * scale, ticks=np.array(ticks) * 60, labels=[f'{x}h' for x in ticks])
        plt.xlabel('Hour ahead', size=20 * scale)
        plt.yticks(size=15 * scale)
        plt.ylabel('Absolute Error ($^\circ$C)', size=20 * scale)
        plt.legend(prop={'size': 15 * scale})
        _save_or_show(save=save, save_name='Error_propagation_paper')

    # Print the information along key points
    print('MAEs:\tRC model:\tPCNN:')
    print(f' 1h:\t{np.mean(rc_model_errors[:, 59]):.3f} K\t\t{np.mean(pcnn_errors[:, 3]):.3f} K\n'
          f' 6h:\t{np.mean(rc_model_errors[:, 359]):.3f} K\t\t{np.mean(pcnn_errors[:, 23]):.3f} K\n'
          f'12h:\t{np.mean(rc_model_errors[:, 719]):.3f} K\t\t{np.mean(pcnn_errors[:, 47]):.3f} K')
    if rc_model.threshold_length >= 24 * 60:
        print(f'24h:\t{np.mean(rc_model_errors[:, 24 * 60 - 1]):.3f} K\t\t{np.mean(pcnn_errors[:, 95]):.3f} K')
    if rc_model.threshold_length >= 48 * 60:
        print(f'48h:\t{np.mean(rc_model_errors[:, 48 * 60 - 1]):.3f} K\t\t{np.mean(pcnn_errors[:, 191]):.3f} K')
    if rc_model.threshold_length >= 72 * 60:
        print(f'72h:\t{np.mean(rc_model_errors[:, 72 * 60 - 1]):.3f} K\t\t{np.mean(pcnn_errors[:, 287]):.3f} K')


def scatter_histogram(rc_model_errors, pcnn_errors, paper: bool = False, scale: int = 1, binwidth: float = 0.025):
    """
    Old function to plot the histograms and scatter plot of MAEs of both models together,
    based on errors computed by `compute_both_errors`.

    Inspired from https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html

    Args:
        rc_model_errors:    MAEs of the RC model
        pcnn_errors:        MAEs of the PCNN
        paper:              Flag to custom the plot for the paper
        scale:              Scale of the size of labels, title, ...
        binwidth:           Width of the bins (in degrees) for the histograms
    """

    # start with a square Figure
    fig = plt.figure(figsize=(20, 20))

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(np.mean(rc_model_errors, axis=1), np.mean(pcnn_errors, axis=1), alpha=0.7, color='red', s=70 * scale)

    # now determine nice limits by hand:
    xmax = np.max(np.mean(rc_model_errors, axis=1))
    limx = (int(xmax / binwidth) + 1) * binwidth
    ymax = np.max(np.mean(pcnn_errors, axis=1))
    limy = (int(ymax / binwidth) + 1) * binwidth
    xymax = max(np.max(np.mean(rc_model_errors, axis=1)), np.max(np.mean(pcnn_errors, axis=1)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    # Plot the diagonal
    ax.plot(np.arange(int(lim) + 2), np.arange(int(lim) + 2), lw=2 * scale, color='black')

    # Both histograms
    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histx.hist(np.mean(rc_model_errors, axis=1), bins=bins, color='blue', label='RC model')
    ax_histy.hist(np.mean(pcnn_errors, axis=1), bins=bins, orientation='horizontal', color='green', label='PCNN')

    # Set nice limits and nice grid
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xticks(np.arange(int(lim) + 1))
    ax.set_yticks(np.arange(int(lim) + 1))
    ax.grid()

    # Cosmetics
    ax_histx.legend(prop={'size': 15 * scale})
    ax_histy.legend(prop={'size': 15 * scale})
    ax.set_xlabel('RC model Absolute Error ($^\circ$C)', size=20 * scale)
    ax.set_ylabel('PCNN Absolute Error ($^\circ$C)', size=20 * scale)
    ax_histx.set_ylabel('Occurences', size=20 * scale)
    ax_histy.set_xlabel('Occurences', size=20 * scale)
    ax.tick_params(axis='x', which='major', labelsize=15 * scale)
    ax_histx.tick_params(axis='y', which='major', labelsize=15 * scale)
    ax.tick_params(axis='y', which='major', labelsize=15 * scale)
    ax_histy.tick_params(axis='x', which='major', labelsize=15 * scale)

    # Print and plot the quantiles
    print("\nRC model:", end="")
    print("\nAbsolute errors:", end="")
    print(f"\nQuantiles", end="")

    for j, q in enumerate([0.5, 0.75, 0.9, 0.95, 0.975]):
        print(f" | {q}: {np.quantile(np.mean(rc_model_errors, axis=1).flatten(), q=q):.3f}", end="")
        ax_histx.axvline(np.quantile(np.mean(rc_model_errors, axis=1).flatten(), q=q), color="black",
                         linewidth=5 - j)

    print("\nPCNN:", end="")
    print("\nAbsolute errors:", end="")
    print(f"\nQuantiles", end="")

    for j, q in enumerate([0.5, 0.75, 0.9, 0.95, 0.975]):
        print(f" | {q}: {np.quantile(np.mean(pcnn_errors, axis=1).flatten(), q=q):.3f}", end="")
        ax_histy.axhline(np.quantile(np.mean(pcnn_errors, axis=1).flatten(), q=q), color="black",
                         linewidth=5 - j)

    if paper:
        _save_or_show(save=True, save_name='Scatter_and_histogram_paper')


def histograms(pcnn, rc_model_errors, pcnn_errors, horizons: list = 96 * 3, binwidth: int = 0.05, scale: int = 1,
               paper: bool = False, quantiles: list = [0.5, 0.75, 0.9, 0.95, 0.975]):
    """
    Function to plot the histograms of MAEs of both models together over sequences of data, as computed in
    `compute_both_errors`. It compares the MAE obtained by both models on each predictions, so one
    data point per test sequence.

    Args:
        pccn:               The PCNN
        rc_model_errors:    MAEs of the RC model
        pcnn_errors:        MAEs of the PCNN
        horizons:           Horizons to do histograms on
        binwidth:           Width of the bins (in degrees) for the histograms
        scale:              Scale of the size of labels, title, ...
        paper:              Flag to custom the plot for the paper
        quantiles:          Which quantiles to analyze
    """

    # Check we have a list of horizons to iterate over
    if not isinstance(horizons, list):
        horizons = [horizons]

    # Define the plot
    fig, ax = _plot_helpers(subplots=[2, len(horizons)], sharex=True, sharey=True, figsize=(20, 12))
    if len(horizons) == 1:
        ax = ax.reshape(-1, 1)

    # Plot helpers
    xymax = max(np.max(np.mean(rc_model_errors, axis=1)), np.max(np.mean(pcnn_errors, axis=1)))
    lim = (int(xymax / binwidth) + 1) * binwidth
    bins = np.arange(0, lim + binwidth, binwidth)

    # Plot results over each horizon in a different column
    for k, horizon in enumerate(horizons):
        print(f'\nHorizon: {int(horizon / 4)}h')

        for i, errors_ in enumerate([rc_model_errors, pcnn_errors]):

            # Compute the MAE over each sequence
            if i == 0:
                print("\nRC model:", end="")
                errors = np.mean(errors_[:, :horizon * pcnn.dataset.interval], axis=1)
            else:
                print("\nPCNN:", end="")
                errors = np.mean(errors_[:, :horizon], axis=1)

            # Print the quantiles of the errors and plot them
            print("\nAbsolute errors:", end="")
            print(f"\nQuantiles", end="")

            for j, q in enumerate(quantiles):
                print(f" | {q}: {np.quantile(errors.flatten(), q=q):.3f}", end="")
                if paper:
                    colors = ['red', 'black']
                    ax[i, k].axvline(np.quantile(errors.flatten(), q=q), color=colors[j],
                                     linewidth=5, label=f'{q}-quantile')
                else:
                    ax[i, k].axvline(np.quantile(errors.flatten(), q=q), color="black",
                                     linewidth=5 - j)

            # Histograms
            ax[i, k].hist(errors.flatten(), bins=bins, color='blue' if i == 0 else 'green')

        ax[1, k].set_xlabel("Mean Absolute Error ($^\circ$C)", size=20 * scale)
        print('')

    # Cosmetics
    ax[0, 0].set_ylabel('RC model\nFrequency', size=20 * scale)
    ax[1, 0].set_ylabel('PCNN\nFrequency', size=20 * scale)
    ax[0, 0].tick_params(axis='y', which='major', labelsize=15 * scale)
    ax[1, 0].tick_params(axis='y', which='major', labelsize=15 * scale)
    for k in range(len(horizons)):
        ax[1, k].tick_params(axis='x', which='major', labelsize=15 * scale)
        ax[1, k].set_xlim(0, 4)

    if paper:
        plt.legend(prop={"size": 15 * scale})
        _save_or_show(save=True, save_name=f"Histograms_paper_mean")


def scatter(rc_model_errors, pcnn_errors, paper: bool = False, scale: int = 1):
    """
    Function to scatter plot MAEs of both models over sequences of data, as computed in
    `compute_both_errors`. It compares the MAE obtained by both models on each predictions, so one
    data point per test sequence.

    Args:
        rc_model_errors:    MAEs of the RC model
        pcnn_errors:        MAEs of the PCNN
        scale:              Scale of the size of labels, title, ...
        paper:              Flag to custom the plot for the paper
    """

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # the scatter plot:
    ax.scatter(np.mean(rc_model_errors, axis=1), np.mean(pcnn_errors, axis=1), alpha=0.7, color='red', s=50 * scale)

    # Helpers
    xymax = max(np.max(np.mean(rc_model_errors, axis=1)), np.max(np.mean(pcnn_errors, axis=1)))
    lim = int(xymax + 1)

    # Plot the diagonal
    ax.plot(np.arange(int(lim) + 2), np.arange(int(lim) + 2), lw=5, color='black')

    # Cosmetics
    ax.set_xlabel('RC model MAE ($^\circ$C)', size=20 * scale)
    ax.set_ylabel('PCNN MAE ($^\circ$C)', size=20 * scale)
    ax.set_yticks(ticks=np.arange(3))
    ax.set_xticks(ticks=np.arange(4))
    ax.tick_params(axis='x', which='major', labelsize=15 * scale)
    ax.tick_params(axis='y', which='major', labelsize=15 * scale)

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.grid()

    if paper:
        _save_or_show(save=True, save_name='Scatter_paper_rot')

def comparison_rc_model_pcnn(rc_model, pcnn, data_sequence: tuple, control_sequence: tuple = None,
                             number_comparisons: int = 2, rc_model_controls: np.array = None,
                             pcnn_controls: np.array = None, labels: list = None, colors: list = None,
                             return_: bool = False, two_cols: bool = False, paper: bool = False, scale: float = 1.,
                             yticks: list = None):
    """
    Function to compare different control inputs for a physics-based and black-box model. The idea in general
    is to compare the same input for both models, but you can give several of them,
    e.g. compare no heating to some heating (the same for both models) to see if they do the same thing.

    Args:
        rc_model:          The physics-based model
        pcnn:              The black-box model
        data_sequence:          The sequence of data to use
        control_sequence:       The seuqence to use for the control
        number_comparisons:     Number of equal control input energy to compare
        rc_model_controls: The controls for the physics-based model. If `None` by default, it takes the
                                  true input and then separates it in 3 equal parts to compare
        pcnn_controls:     Same as `rc_model_controls` for the black-box model
        labels:                 Labels of the plot
        colors:                 Colors of the different control sequences
        return_:                Whether to return the results instead of plotting them
        two_cols:          Plot the two columns version
        paper:             Plot for the paper
        scale:             Scale of all the texts
        yticks:            Custom ticks for the y-axis

    Returns:
        Either plots the physics-based and black-box temperature predictions or returns them.
    """

    # Sanity check that we are dealing with the same dataset, otherwise comparisons will be flawed
    assert ((pcnn.dataset.data.index[-1] - pcnn.dataset.data.index[0])
            - (rc_model.data.index[-1] - rc_model.data.index[0])).seconds / 60 \
           < 2 * pcnn.dataset.interval, 'Are you comparing similar data?'

    # Sanity check: the contol sequence is the same length as the data to use, otherwise trim the longest one
    if control_sequence is None:
        control_sequence = data_sequence
    if control_sequence == data_sequence:
        plot_ground_truth = True
    else:
        # If it's a new control sequence there is no ground truth to plot as comparison
        plot_ground_truth = False

    if data_sequence[1] - data_sequence[0] < control_sequence[1] - control_sequence[0]:
        control_sequence = (control_sequence[0], control_sequence[0] + data_sequence[1] - data_sequence[0])

    elif data_sequence[1] - data_sequence[0] > control_sequence[1] - control_sequence[0]:
        data_sequence = (data_sequence[0], data_sequence[0] + control_sequence[1] - control_sequence[0])

    # Define constants
    interval = pcnn.dataset.interval
    if colors is None:
        colors = ['black', 'blue', 'red', 'orange', 'green']

    # Compute the control sequence from the data corresponding to the wanted sequence
    rc_model_control = get_rc_model_control_from_sequence(rc_model=rc_model,
                                                          pcnn=pcnn,
                                                          sequence=control_sequence)
    # Check we actually have energy inputs, else raise an IndexError, which would be raised further
    # in the code in that case
    if np.abs(rc_model_control.sum()) < 1e-6:
        raise IndexError(f'No energy is used in this sequence')

    # Compute the corresponding black-box control input
    pcnn_control = torch.FloatTensor(pcnn.dataset.X[control_sequence[0] + pcnn.warm_start_length:
                                                    control_sequence[1], pcnn.power_column].copy()).to \
        (pcnn.device)

    # Default controls: compare no energy, the true control sequence and 3 other controls: one with
    # only the first third of the control input, then the second one, and the third
    if rc_model_controls is None:
        # Corresponding labels
        labels = ['No Energy', 'Full control sequence', 'First third', 'Second third',
                  'Last third'] if number_comparisons == 3 else ['No power', 'Full power input', 'First half',
                                                                 'Second half']

        rc_model_controls = np.zeros(
            ((control_sequence[1] - control_sequence[0] - pcnn.warm_start_length) * interval,
             number_comparisons + 2))
        rc_model_controls[:, 1] = rc_model_control.copy()

        # Define constants to separate the data equally and an array to save the controls
        if number_comparisons > 0:
            seps = [0] + [int(np.round(
                np.where(rc_model_control.cumsum() / rc_model_control.sum() > sep / number_comparisons)[0][0] \
                / interval) * interval) for sep in range(1, number_comparisons)] + [-1]

            # Define the different control inputs (the first one is zero)
            for i, (sep1, sep2) in enumerate(zip(seps[:-1], seps[1:]), 2):
                rc_model_controls[sep1: sep2, i] = rc_model_control[sep1: sep2]

        # Same for the black-box model - here we might deal with normalized or standardized data, so we need
        # to correct the zero value using the zero power argument
        pcnn_controls = torch.zeros((control_sequence[1] - control_sequence[0] - pcnn.warm_start_length,
                                     number_comparisons + 2), dtype=torch.float).to(pcnn.device) + pcnn.zero_power
        pcnn_controls[:, 1] = pcnn_control.squeeze()

        # Same as above
        if number_comparisons > 0:
            seps = [0] + [int(sep / interval) for sep in seps[1: -1]] + [-1]
            for i, (sep1, sep2) in enumerate(zip(seps[:-1], seps[1:]), 2):
                pcnn_controls[sep1: sep2, i] = pcnn_control[sep1: sep2].squeeze()

    # Arrays to store the computed temperatures
    rc_model_T = np.zeros(((data_sequence[1] - data_sequence[0] - pcnn.warm_start_length) * interval + 1,
                           rc_model_controls.shape[1]))
    pcnn_T = np.zeros((data_sequence[1] - data_sequence[0] - pcnn.warm_start_length + 1,
                       pcnn_controls.shape[1]))

    # Physics-based model predictions for all the control inputs

    start = (data_sequence[0] + pcnn.warm_start_length) * interval \
            - rc_model.data.index[(data_sequence[0] + pcnn.warm_start_length) * interval].minute % 15
    end = data_sequence[1] * interval + 1 \
          - rc_model.data.index[(data_sequence[0] + pcnn.warm_start_length) * interval].minute % 15

    # Check we are indeed giving back the true control inputs
    assert pcnn.dataset.data.index[data_sequence[0] + pcnn.warm_start_length] == \
           rc_model.data.index[start], 'Something went wrong with the control definition, indexes not matching'
    assert pcnn.dataset.data.index[data_sequence[1]] == rc_model.data.index[end] - timedelta(minutes=1), \
        'Something went wrong with the control definition, indexes not matching'

    for i in range(rc_model_T.shape[1]):
        rc_model_T[:, i] = rc_model.predict(start=start,
                                            end=end,
                                            control=rc_model_controls[:, i],
                                            from_predicted=False)

    # Define the datas for the black-box model
    pcnn_datas = [pcnn.X[data_sequence[0]: data_sequence[1]].copy(),
                  pcnn.Y[data_sequence[0]: data_sequence[1]].copy()]
    pcnn_datas = [torch.FloatTensor(x).to(pcnn.device) for x in pcnn_datas]

    # Black-box model predictions for all the control inputs
    for i in range(pcnn_T.shape[1]):
        # Put the right control input in the data
        pcnn_datas[0][pcnn.warm_start_length:, pcnn.power_column] = pcnn_controls[:, i].type(torch.FloatTensor).reshape(-1,1)
        pcnn_datas[1][pcnn.warm_start_length:, -1] = pcnn_controls[:, i].type(torch.FloatTensor).squeeze()

        # Correct the valves
        pcnn_datas[0][pcnn.warm_start_length:, pcnn.valve_column] = (np.abs(pcnn_controls[:, i].type(torch.FloatTensor).reshape(-1,1) -
                                                                            pcnn.zero_power) > 1e-6) * 1 * 0.8 + 0.1

        # Correct the case
        pcnn_datas[0][pcnn.warm_start_length:, pcnn.case_column] = (pcnn_controls[:, i].type(torch.FloatTensor).squeeze() > torch.FloatTensor(pcnn.zero_power)) * 1 * 0.8 + 0.1

        # Predict and store
        T, _ = pcnn.scale_back_predictions(data=pcnn_datas)
        pcnn_T[:, i] = T[0, pcnn.warm_start_length - 1:, 0].squeeze()

    if return_:
        return rc_model_T, pcnn_T, rc_model_control, pcnn_control

    else:
        # Plot everything nicely
        if two_cols:
            fig, ax = plt.subplots(2, 2, figsize=(30, 12), sharex=True)
            ax[0, 0].set_ylabel('Temperature', size=20)
            ax[1, 0].set_ylabel('Power', size=20)
            ax[0, 0].set_title('Physics-based model', size=20)
            ax[0, 1].set_title('Black-box model', size=20)
            ax[1, 0].set_title('Control sequence', size=20)
            ax[1, 1].set_title('Control sequence', size=20)

            if plot_ground_truth:
                for i in range(2):
                    ax[0, i].plot(list(pcnn.dataset.data.index[data_sequence[0] + pcnn.warm_start_length:
                                                               data_sequence[1] + 1]),
                                  pcnn.dataset.inverse_normalize(pcnn.dataset.data['Room temperature']
                                                                 [data_sequence[0] + pcnn.warm_start_length:
                                                                  data_sequence[1] + 1]),
                                  color=colors[1], linestyle='dashed', alpha=0.7, label='Ground truth')

            # Shade the physics-based prediction in the black-box plot
            ax[0, 1].fill_between(rc_model.data.index[
                                  (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[
                                      1]) * interval + 1],
                                  rc_model_T[:, 0], rc_model_T[:, 1], color='grey', alpha=0.1)
            ax[0, 1].plot(rc_model.data.index[
                          (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[
                              1]) * interval + 1],
                          rc_model_T[:, 0], color='grey', linestyle='dashed', lw=1, alpha=0.2,
                          label='Physics-based predictions')
            ax[0, 1].plot(rc_model.data.index[
                          (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[
                              1]) * interval + 1],
                          rc_model_T[:, 1], color='grey', linestyle='dashed', lw=1, alpha=0.2)

            # Plot the physics-based prediction, the black-box one and the control input in each case
            for i in range(rc_model_controls.shape[1]):
                ax[0, 0].plot(rc_model.data.index[
                              (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[
                                  1]) * interval + 1],
                              rc_model_T[:, i], color=colors[i], lw=2, label=labels[i])
                ax[0, 1].plot(list(
                    pcnn.dataset.data.index[data_sequence[0] + pcnn.warm_start_length: data_sequence[1] + 1]),
                    pcnn_T[:, i], color=colors[i], lw=2, label=labels[i])
                for j in range(2):
                    ax[1, j].plot(rc_model.data.index[
                                  (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[
                                      1]) * interval],
                                  rc_model_controls[:, i], color=colors[i], lw=2)

            for i in range(2):
                ax[0, i].set_ylim(min(ax[0, 0].get_ylim()[0], ax[0, 1].get_ylim()[0]),
                                  max(ax[0, 0].get_ylim()[1], ax[0, 1].get_ylim()[1]))

            # Some cosmetics and plot it
            for i in range(2):
                for j in range(2):
                    ax[i, j].tick_params(axis='x', which='major', labelsize=15)
                    ax[i, j].tick_params(axis='y', which='major', labelsize=15)
                ax[1, i].set_xlabel('Time', size=20)

            fig.autofmt_xdate()
            ax[0, 1].legend(prop={"size": 15}, loc="upper left", ncol=1, bbox_to_anchor=(1, 1))
            plt.show()
        else:
            fig, ax = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
            ax[0].set_ylabel('Temperature\n($^\circ$C)', size=20 * scale)
            ax[1].set_ylabel('Temperature\n($^\circ$C)', size=20 * scale)
            ax[2].set_ylabel('Power\n(kW)', size=20 * scale)
            ax[0].set_title('RC model', size=20 * scale)
            ax[1].set_title('PCNN', size=20 * scale)
            ax[2].set_title('Power input', size=20 * scale)

            if plot_ground_truth:
                for i in range(2):
                    ax[i].plot(list(pcnn.dataset.data.index[data_sequence[0] + pcnn.warm_start_length:
                                                            data_sequence[1] + 1]),
                               pcnn.dataset.inverse_normalize(pcnn.dataset.data['Room temperature']
                                                              [data_sequence[0] + pcnn.warm_start_length:
                                                               data_sequence[1] + 1]),
                               color=colors[1], linestyle='dashed', alpha=0.7)  # , label='Ground truth')

            # Shade the physics-based prediction in the black-box plot
            ax[1].fill_between(rc_model.data.index[
                               (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[
                                   1]) * interval + 1],
                               rc_model_T[:, 0], rc_model_T[:, 1], color='grey', alpha=0.1)
            ax[1].plot(rc_model.data.index[
                       (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[1]) * interval + 1],
                       rc_model_T[:, 1], color='grey', linestyle='dashed', lw=1, alpha=0.2)

            # Plot the physics-based prediction, the black-box one and the control input in each case
            for i in range(rc_model_controls.shape[1]):
                ax[0].plot(rc_model.data.index[
                           (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[
                               1]) * interval + 1],
                           rc_model_T[:, i], color=colors[i], lw=3, label=labels[i])
                ax[1].plot(list(
                    pcnn.dataset.data.index[data_sequence[0] + pcnn.warm_start_length: data_sequence[1] + 1]),
                    pcnn_T[:, i], color=colors[i], lw=3, label=labels[i])
                ax[2].plot(rc_model.data.index[
                           (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[1]) * interval],
                           rc_model_controls[:, i] / 1000, color=colors[i], lw=3)

            for i in range(2):
                ax[i].set_ylim(min(ax[0].get_ylim()[0], ax[1].get_ylim()[0]),
                               max(ax[0].get_ylim()[1], ax[1].get_ylim()[1]))
                if yticks is not None:
                    ax[i].set_yticks(ticks=np.array(yticks))  # , labels=[x for x in yticks])

            # Some cosmetics and plot it
            for i in range(3):
                ax[i].tick_params(axis='x', which='major', labelsize=15 * scale)
                ax[i].tick_params(axis='y', which='major', labelsize=15 * scale)
                ax[i].set_xlabel('Time', size=20 * scale)
            fig.autofmt_xdate()
            ax[0].legend(prop={"size": 15 * scale}, loc="upper left")  # , ncol=1, bbox_to_anchor=(1, 1))
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%-d %b, %-Hh"))

        if paper:
            _save_or_show(save=True, save_name='Physical_consistency_paper')


def comparison_rc_model_pcnn_differences(rc_model, pcnn, data_sequence: tuple,
                                         control_sequence: tuple = None, number_comparisons: int = 2,
                                         rc_model_controls: np.array = None,
                                         pcnn_controls: np.array = None, labels: list = None,
                                         colors: list = None,
                                         normalized: bool = True, return_: bool = False, paper: bool = False,
                                         scale: float = 1.):
    """
    Function to compare the differences in prediction between the physics-based and the black-box model
    given different control sequences. It compute the difference between each control input and when no
    energy is used and plots it.

    Uses `comparison_rc_model_pcnn` to do the comparison. If no controls are given as input,
    the default one of `comparison_rc_model_pcnn` are used.

    Args:
        rc_model:          The physics-based model
        pcnn:              The black-box model
        data_sequence:          The sequence of data to use
        control_sequence:       The control sequence to use
        number_comparisons:     Number of equal control input energy to compare
        rc_model_controls: The controls for the physics-based model. If `None` by default, it takes the
                                  true input and then separates it in 3 equal parts to compare
        pcnn_controls:     Same as `rc_model_controls` for the black-box model
        labels:                 Labels of the plot
        colors:                 Colors of the different control sequences
        normalized:             Whether to normalize the comparison
        return_:                Flag to return the results instead of plotting them
    """

    # Define constants and helpers
    interval = pcnn.dataset.interval
    if colors is None:
        colors = ['black', 'blue', 'red', 'orange', 'green', 'magenta']
    if labels is None:
        labels = ['No Energy', 'Full control sequence', 'First part', 'Second part', 'Third part',
                  'Fourth'] if number_comparisons >= 3 else ['No Energy', 'Full power input', 'First half',
                                                             'Second half']

    # Predict the different controls for both models
    rc_model_T, pcnn_T, rc_model_control, pcnn_control = comparison_rc_model_pcnn(
        rc_model=rc_model,
        pcnn=pcnn,
        data_sequence=data_sequence,
        control_sequence=control_sequence,
        number_comparisons=number_comparisons,
        rc_model_controls=rc_model_controls,
        pcnn_controls=pcnn_controls,
        labels=labels,
        colors=colors,
        return_=True)

    rc_model_differences = np.zeros((rc_model_T.shape[0], rc_model_T.shape[1] - 1))
    pcnn_differences = np.zeros((pcnn_T.shape[0], pcnn_T.shape[1] - 1))

    # Normalized the data for both models in axis 0 and 1, and plot the difference in axis 2
    for i in range(1, rc_model_T.shape[1]):
        if normalized:
            rc_model_differences[:, i - 1] = (rc_model_T[:, i] - rc_model_T[:, 0]) \
                                             / np.max(np.abs(rc_model_T[:, 1] - rc_model_T[:, 0]))
            pcnn_differences[:, i - 1] = (pcnn_T[:, i] - pcnn_T[:, 0]) \
                                         / np.max(np.abs(pcnn_T[:, 1] - pcnn_T[:, 0]))

        # Normal version, without normalization
        else:
            rc_model_differences[:, i - 1] = rc_model_T[:, i] - rc_model_T[:, 0]
            pcnn_differences[:, i - 1] = pcnn_T[:, i] - pcnn_T[:, 0]

    if return_:
        return rc_model_differences, pcnn_differences

    else:
        fig, ax = plt.subplots(2, 1, figsize=(20, 12), sharex=True, sharey=True)
        ax[0].set_ylabel('Difference\n($^\circ$C)', size=17 * scale)
        ax[1].set_ylabel('Difference\n($^\circ$C)', size=17 * scale)
        ax[0].set_title('RC model', size=17 * scale)
        ax[1].set_title('PCNN', size=17 * scale)

        for i in range(1, rc_model_T.shape[1]):
            ax[0].plot(rc_model.data.index[
                       (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[1]) * interval + 1],
                       rc_model_differences[:, i - 1], color=colors[i], lw=4, label=labels[i])
            ax[1].plot(list(pcnn.dataset.data.index[data_sequence[0] + pcnn.warm_start_length:
                                                    data_sequence[1] + 1]),
                       pcnn_differences[:, i - 1], color=colors[i], lw=4, label=labels[i])

        # Cosmetics and plot
        for i in range(2):
            ax[i].tick_params(axis='x', which='major', labelsize=15 * scale)
            ax[i].tick_params(axis='y', which='major', labelsize=15 * scale)
            ax[i].set_yticks([0, 3, 6])
            ax[i].set_xlabel('Time', size=17 * scale)
            ax[i].grid()
        fig.autofmt_xdate()
        box = ax[0].get_position()
        ax[0].set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        box = ax[1].get_position()
        ax[1].set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax[1].legend(prop={"size": 13 * scale}, loc='upper center', bbox_to_anchor=(0.45, -1.1), ncol=3)
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%-d %b, %-Hh"))

        plt.tight_layout()

        if paper:
            _save_or_show(save=True, save_name='Physical_differences_paper')


def compute_differences(physics_based, black_box, data_sequences: list, control_sequences: list = None,
                        number_comparisons: int = 0, labels: list = None, colors: list = None, normalized: bool = True):
    """
    Function to compare the differences in prediction between the physics-based and the black-box model
    given different control sequences, each time using `comparison_physics_based_black_box_differences`
    with the default controls.

    Args:
       physics_based:          The physics-based model
       black_box:              The black-box model
       data_sequences:         The sequences of data to use
       control_sequences:      The control sequences to use
       number_comparisons:     Number of equal control input energy to compare
       labels:                 Labels of the plot
       colors:                 Colors of the different control sequences
       normalized:             Whether to normalize the comparison
    """

    print(f'\nComputing differences on {len(data_sequences)} sequences...')

    if control_sequences is None:
        control_sequences = data_sequences

    if colors is None:
        colors = ['blue', 'red', 'orange', 'green', 'magenta']
    if labels is None:
        labels = ['Full control sequence', 'First part', 'Second part', 'Third part', 'Fourth part']

    differences = np.empty((int(black_box.maximum_sequence_length + 1),
                            (number_comparisons + 1) * len(data_sequences) * len(control_sequences)))
    differences[:] = np.nan

    # Loop over the sequences and compute the output under different control inputs
    count = 0
    for j, data_sequence in enumerate(data_sequences):
        for control_sequence in control_sequences:
            # Little trick: the default control inputs don't work if no energy was used during the whole
            # sequence: we catch it
            try:
                physics_based_differences, black_box_differences = comparison_rc_model_pcnn_differences(
                    physics_based=physics_based,
                    black_box=black_box,
                    data_sequence=data_sequence,
                    control_sequence=control_sequence,
                    number_comparisons=number_comparisons,
                    normalized=normalized,
                    return_=True)
                count += 1
            except IndexError:
                pass

            # Store the 4 differences (full control sequence, first, second and third part against no power)
            differences[:black_box_differences.shape[0], (number_comparisons + 1) * (count - 1):
                                                         (number_comparisons + 1) * count] = \
                np.abs(black_box_differences - physics_based_differences[::black_box.dataset.interval])

        if j % 5 == 4:
            print(f'{j + 1} sequences treated...')

    print(f'Compared {count} sequences...')

    _plot_helpers(subplots=(1, 1), figsize=(20, 10), xlabel='Horizon', ylabel='Difference',
                                sharex=True, sharey=True)
    for i in range(number_comparisons + 1):
        plt.plot(np.nanmean(differences[:, i::4], axis=1), color=colors[i], label=labels[i])
    plt.legend(prop={"size": 15}, loc="upper left", ncol=1, bbox_to_anchor=(1, 1))
    plt.show()

    return differences[:, :(number_comparisons + 1) * count]


def comparison_pcnn_lstm(rc_model, pcnn, lstm, data_sequence: tuple, control_sequences, labels,
                         paper: bool = False, scale: float = 1., yticks: list = None,
                         save_name: str = 'LSTM_inconsistency_paper', **kwargs):
    """
    Function to compare different control inputs for a physics-based and black-box model. The idea in general
    is to compare the same input for both models, but you can give several of them,
    e.g. compare no heating to some heating (the same for both models) to see if they do the same thing.

    Args:
        rc_model:          The physics-based model
        pcnn:              The black-box model
        data_sequence:          The sequence of data to use
        control_sequences:       The sequences to use for the control
        paper:             Plot for the paper
        labels:            Labels of the plot
        scale:             Scale of all the texts
        yticks:            Custom ticks for the y-axis

    Returns:
        Either plots the physics-based and black-box temperature predictions or returns them.
    """

    # Sanity check that we are dealing with the same dataset, otherwise comparisons will be flawed
    assert ((pcnn.dataset.data.index[-1] - pcnn.dataset.data.index[0])
            - (rc_model.data.index[-1] - rc_model.data.index[0])).seconds / 60 \
           < 2 * pcnn.dataset.interval, 'Are you comparing similar data?'

    shortest = control_sequences[np.argmin([x[1] - x[0] for x in control_sequences])]
    if data_sequence[1] - data_sequence[0] < shortest[1] - shortest[0]:
        control_sequences = [(sequence[0], sequence[0] + data_sequence[1] - data_sequence[0]) for sequence in
                             control_sequences]
    elif data_sequence[1] - data_sequence[0] > shortest[1] - shortest[0]:
        data_sequence = (data_sequence[0], data_sequence[0] + shortest[1] - shortest[0])
        control_sequences = [(sequence[0], sequence[0] + shortest[1] - shortest[0]) for sequence in control_sequences]

    # Define constants
    interval = pcnn.dataset.interval
    colors = ['black', 'red', 'blue', 'orange']

    rc_model_controls = np.zeros(
        ((control_sequences[0][1] - control_sequences[0][0] - pcnn.warm_start_length) * interval,
         len(control_sequences) + 1))
    # Same for the black-box model - here we might deal with normalized or standardized data, so we need
    # to correct the zero value using the zero power argument
    pcnn_controls = torch.zeros((control_sequences[0][1] - control_sequences[0][0] - pcnn.warm_start_length,
                                 len(control_sequences) + 1)).to(pcnn.device) + pcnn.zero_power

    for i, control_sequence in enumerate(control_sequences):
        # Compute the control sequence from the data corresponding to the wanted sequence
        rc_model_controls[:, i + 1] = get_rc_model_control_from_sequence(rc_model=rc_model,
                                                                         pcnn=pcnn,
                                                                         sequence=control_sequence)
        # Check we actually have energy inputs, else raise an IndexError, which would be raised further
        # in the code in that case
        if np.abs(rc_model_controls[:, i + 1].sum()) < 1e-6:
            raise IndexError(f'No energy is used in sequence {i}')

        # Compute the corresponding black-box control input
        pcnn_controls[:, i + 1] = torch.FloatTensor(pcnn.dataset.X[control_sequence[0] + pcnn.warm_start_length:
                                                                   control_sequence[1], pcnn.power_column].copy()).to(
            pcnn.device).squeeze()

    # Arrays to store the computed temperatures
    rc_model_T = np.zeros(((data_sequence[1] - data_sequence[0] - pcnn.warm_start_length) * interval + 1,
                           rc_model_controls.shape[1]))
    pcnn_T = np.zeros((data_sequence[1] - data_sequence[0] - pcnn.warm_start_length + 1,
                       pcnn_controls.shape[1]))
    lstm_T = np.zeros((data_sequence[1] - data_sequence[0] - pcnn.warm_start_length + 1,
                       pcnn_controls.shape[1]))

    # Physics-based model predictions for all the control inputs

    start = (data_sequence[0] + pcnn.warm_start_length) * interval \
            - rc_model.data.index[(data_sequence[0] + pcnn.warm_start_length) * interval].minute % 15
    end = data_sequence[1] * interval + 1 \
          - rc_model.data.index[(data_sequence[0] + pcnn.warm_start_length) * interval].minute % 15

    # Check we are indeed giving back the true control inputs
    assert pcnn.dataset.data.index[data_sequence[0] + pcnn.warm_start_length] == \
           rc_model.data.index[start], 'Something went wrong with the control definition, indexes not matching'
    assert pcnn.dataset.data.index[data_sequence[1]] == rc_model.data.index[end] - timedelta(minutes=1), \
        'Something went wrong with the control definition, indexes not matching'

    for i in range(rc_model_T.shape[1]):
        rc_model_T[:, i] = rc_model.predict(start=start,
                                            end=end,
                                            control=rc_model_controls[:, i],
                                            from_predicted=False)

    # Define the datas for the black-box model
    pcnn_datas = [pcnn.X[data_sequence[0]: data_sequence[1]].copy(),
                  pcnn.Y[data_sequence[0]: data_sequence[1]].copy()]
    pcnn_datas = [torch.FloatTensor(x).to(pcnn.device) for x in pcnn_datas]

    # Black-box model predictions for all the control inputs
    for i in range(pcnn_T.shape[1]):
        # Put the right control input in the data
        pcnn_datas[0][pcnn.warm_start_length:, pcnn.power_column] = pcnn_controls[:, i].type(torch.FloatTensor).reshape(-1, 1)
        pcnn_datas[1][pcnn.warm_start_length:, -1] = pcnn_controls[:, i].type(torch.FloatTensor).squeeze()

        # Correct the valves
        pcnn_datas[0][pcnn.warm_start_length:, pcnn.valve_column] = (np.abs(pcnn_controls[:, i].type(torch.FloatTensor).reshape(-1, 1) -
                                                                            pcnn.zero_power) > 1e-6) * 1 * 0.8 + 0.1

        # Correct the case
        pcnn_datas[0][pcnn.warm_start_length:, pcnn.case_column] = (pcnn_controls[:, i].type(torch.FloatTensor).squeeze() > torch.FloatTensor(
            pcnn.zero_power)) * 1 * 0.8 + 0.1

        # Predict and store
        T, _ = pcnn.scale_back_predictions(data=pcnn_datas)
        pcnn_T[:, i] = T[0, pcnn.warm_start_length - 1:, 0].squeeze()

    # Define the datas for the black-box model
    lstm_datas = [lstm.X[data_sequence[0]: data_sequence[1]].copy(),
                  lstm.Y[data_sequence[0]: data_sequence[1]].copy()]
    lstm_datas = [torch.FloatTensor(x).to(lstm.device) for x in lstm_datas]

    # Black-box model predictions for all the control inputs
    for i in range(pcnn_T.shape[1]):
        # Put the right control input in the data
        lstm_datas[0][lstm.warm_start_length:, lstm.power_column] = pcnn_controls[:, i].type(torch.FloatTensor).reshape(-1, 1)
        lstm_datas[1][lstm.warm_start_length:, -1] = pcnn_controls[:, i].type(torch.FloatTensor).squeeze()

        # Correct the valves
        lstm_datas[0][lstm.warm_start_length:, lstm.valve_column] = (np.abs(pcnn_controls[:, i].type(torch.FloatTensor).reshape(-1, 1) -
                                                                            lstm.zero_power) > 1e-6) * 1 * 0.8 + 0.1

        # Correct the case
        lstm_datas[0][lstm.warm_start_length:, lstm.case_column] = (pcnn_controls[:, i].type(torch.FloatTensor).squeeze() > torch.FloatTensor(
            lstm.zero_power)) * 1 * 0.8 + 0.1

        T, _ = lstm.scale_back_predictions(data=lstm_datas)
        lstm_T[:, i] = T[0, lstm.warm_start_length - 1:, 0].squeeze()

    fig, ax = plt.subplots(4, 1, figsize=(20, 18), sharex=True)

    ax[0].set_ylabel('Temperature\n($^\circ$C)', size=20 * scale)
    ax[1].set_ylabel('Temperature\n($^\circ$C)', size=20 * scale)
    ax[2].set_ylabel('Temperature\n($^\circ$C)', size=20 * scale)
    ax[3].set_ylabel('Power\n(kW)', size=20 * scale)
    ax[0].set_title('Classical RC model', size=20 * scale)
    ax[1].set_title('Classical LSTM', size=20 * scale)
    ax[2].set_title('PCNN (ours)', size=20 * scale)
    ax[3].set_title('Power input', size=20 * scale)

    # Shade the physics-based prediction in the black-box plot
    ax[1].fill_between(rc_model.data.index[
                       (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[
                           1]) * interval + 1],
                       rc_model_T[:, 2], rc_model_T[:, 1], color='grey', alpha=0.1)
    ax[2].fill_between(rc_model.data.index[
                       (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[
                           1]) * interval + 1],
                       rc_model_T[:, 2], rc_model_T[:, 1], color='grey', alpha=0.1)

    # Plot the physics-based prediction, the black-box one and the control input in each case
    for i in range(rc_model_controls.shape[1]):
        ax[0].plot(rc_model.data.index[
                   (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[
                       1]) * interval + 1],
                   rc_model_T[:, i], color=colors[i], lw=3, label=labels[i])
        ax[1].plot(list(
            lstm.dataset.data.index[data_sequence[0] + lstm.warm_start_length: data_sequence[1] + 1]),
            lstm_T[:, i], color=colors[i], lw=3, label=labels[i])
        ax[2].plot(list(
            pcnn.dataset.data.index[data_sequence[0] + pcnn.warm_start_length: data_sequence[1] + 1]),
            pcnn_T[:, i], color=colors[i], lw=3, label=labels[i])
        ax[3].plot(rc_model.data.index[
                   (data_sequence[0] + pcnn.warm_start_length) * interval: (data_sequence[1]) * interval],
                   rc_model_controls[:, i] / 1000, color=colors[i], lw=3)

    for i in range(3):
        ax[i].set_ylim(min(ax[0].get_ylim()[0], ax[1].get_ylim()[0], ax[2].get_ylim()[0]),
                       max(ax[0].get_ylim()[1], ax[1].get_ylim()[1], ax[2].get_ylim()[1]))
        if yticks is not None:
            ax[i].set_yticks(ticks=np.array(yticks))

    # Some cosmetics and plot it
    for i in range(4):
        ax[i].tick_params(axis='x', which='major', labelsize=15 * scale)
        ax[i].tick_params(axis='y', which='major', labelsize=15 * scale)
        ax[i].set_xlabel('Time', size=20 * scale)
    fig.autofmt_xdate()
    ax[0].legend(prop={"size": 15 * scale}, loc="upper left")  # , ncol=1, bbox_to_anchor=(1, 1))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%-d %b, %-Hh"))

    if paper:
        _save_or_show(save=True, save_name=save_name)


class OneRoomModel:

    def __init__(self, data, dt, room, neigh, threshold, interval, overlap=None):

        self.data = data
        self.dt = dt
        self.room = room
        self.neigh = neigh

        self.threshold_length = threshold
        self.overlap = overlap
        self.interval = interval
        self.create_sequences(threshold=self.threshold_length, overlap=self.overlap)

        self.p = np.array([1e-5, 1e-5, 1e-5] + [1e-5] * len(self.neigh)).astype(float)

    def create_sequences(self, threshold: int = None, overlap=None, return_: bool = False, data=None) -> None:
        """
        Function to create lists of starts and ends of sequences of data that can be used (i.e.
        there is no missing data).

        Args:
            threshold:          Threshold on the sequences length (we use Euler's method, it cannot
                                  be extrapolated to too long sequences)
        """

        if overlap is None:
            overlap = self.overlap
        if threshold is None:
            threshold = self.threshold
        if data is None:
            data = self.data
        # Find the places with missing values
        nans = np.where(pd.isnull(data).any(axis=1))[0]

        # Define the starts and ends of sequences of data without missing values
        starts_ = list(nans[np.where(np.diff(nans) != 1)[0]] + 1)
        ends_ = list(nans[np.where(np.diff(nans) != 1)[0] + 1])

        # Go through the sequences and separate the ones that are too long in several chunks
        starts = []
        ends = []
        for start, end in zip(starts_, ends_):
            # Cut the sequence in chunk of `threshold` length
            while end - start > threshold:
                starts.append(start)
                ends.append(start + threshold + 1)
                start += overlap
            starts.append(start)
            ends.append(end)

        # Small correction to handle the start and end of the data correctly
        if nans[0] > 0:
            starts = [0] + starts
            ends = [nans[0]] + ends
        if nans[-1] < len(data) - 1:
            starts = starts + [nans[-1] + 1]
            ends = ends + [len(data) - 1]

        if return_:
            return starts, ends
        else:
            self.starts = starts
            self.ends = ends

    def predict(self, starts: int, ends: int, control: np.array = None, data:pd.DataFrame = None):
        """
        Function to predict the temperature over a given sequence of data (start, end)

        Args:
            starts:     Start of the sequences
            ends:       End of the sequences
            control:    Control sequence if another sequence than what is in the data is to be taken
            data:       Data to predict from
        """

        if isinstance(starts, int):
            starts = [starts]
        if isinstance(ends, int):
            ends = [ends]
        if data is None:
            data = self.data

        assert len(starts) == len(ends), 'Provide same number of starts and ends'

        truth = pad_sequence(
            [torch.FloatTensor(data.iloc[start+1:end, :].astype(float).values) for start, end in zip(starts, ends)],
            padding_value=0, batch_first=True).detach().numpy()

        if control is None:
            control = pad_sequence(
                [torch.FloatTensor(data[[f'Power room {self.room}']][start:end-1].values)
                 for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()

        irr = pad_sequence([torch.FloatTensor(data['Weather solar irradiation'][start:end-1].values)
                            for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()
        out = pad_sequence([torch.FloatTensor(data['Weather outside temperature'][start:end-1].values)
                            for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()

        neighs = []
        for x in self.neigh:
            neighs.append(
                pad_sequence([torch.FloatTensor(data[f'Thermal temperature measurement {x}'][start:end-1].values)
                              for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy())

        # The temperature starts at the right value
        max_ = np.max([end - start for start, end in zip(starts, ends)])
        T = np.zeros((len(starts), max_))
        T[:, 0] = data[f'Thermal temperature measurement {self.room}'][starts]

        # Euler method for integration with 60 seconds time steps
        for t in range(max_ - 1):
            T[:, t + 1] = T[:, t] + self.dt * self.interval * (control[:, t] * self.p[0] +
                                                          irr[:, t] * self.p[1] +
                                                          (out[:, t] - T[:, t]) * self.p[2]).squeeze()
            for i in range(len(self.neigh)):
                T[:, t + 1] += self.dt * self.interval * ((neighs[i][:, t] - T[:, t]) * self.p[i + 3]).squeeze()

        T[np.where(np.abs(truth.mean(axis=2)) < 1e-6)] = 0

        return T[:,1:], truth


class OneRoomSepModel(OneRoomModel):

    def __init__(self, data, dt, room, neigh, threshold, interval, overlap=None):

        super().__init__(data, dt, room, neigh, threshold, interval, overlap=overlap)

        self.p = np.array([1e-5, 1e-5, 1e-5, 1e-5] + [1e-5] * len(self.neigh)).astype(float)

        self._create_sequences(threshold=threshold, overlap=overlap)

    def _create_sequences(self, threshold: int = None, overlap=None) -> None:
        """
        Function to create lists of starts and ends of sequences of data that can be used (i.e.
        there is no missing data).

        Args:
            threshold:          Threshold on the sequences length (we use Euler's method, it cannot
                                  be extrapolated to too long sequences)
        """

        data = self.data.copy()
        data.iloc[np.where(self.data['Case'] < 0)[0],:] = np.nan
        starts, ends = self.create_sequences(data=data, threshold=threshold, overlap=overlap, return_=True)

        data = self.data.copy()
        data.iloc[np.where(self.data['Case'] > 0)[0], :] = np.nan
        starts_, ends_ = self.create_sequences(data=data, threshold=threshold, overlap=overlap, return_=True)

        self.starts = starts + starts_
        self.ends = ends + ends_

    def predict(self, starts: int, ends: int, control: np.array = None, data:pd.DataFrame = None):
        """
        Function to predict the temperature over a given sequence of data (start, end)

        Args:
            starts:     Start of the sequences
            ends:       End of the sequences
            control:    Control sequence if another sequence than what is in the data is to be taken
            data:       Data to predict from
        """

        if isinstance(starts, int):
            starts = [starts]
        if isinstance(ends, int):
            ends = [ends]
        if data is None:
            data = self.data

        assert len(starts) == len(ends), 'Provide same number of starts and ends'

        truth = pad_sequence(
            [torch.FloatTensor(data.iloc[start+1:end, :].astype(float).values) for start, end in zip(starts, ends)],
            padding_value=0, batch_first=True).detach().numpy()

        if control is None:
            control = pad_sequence(
                [torch.FloatTensor(data[[f'Power room {self.room}']][start:end-1].values)
                 for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy().squeeze()

        irr = pad_sequence([torch.FloatTensor(data['Weather solar irradiation'][start:end-1].values)
                            for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()
        out = pad_sequence([torch.FloatTensor(data['Weather outside temperature'][start:end-1].values)
                            for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()

        neighs = []
        for x in self.neigh:
            neighs.append(
                pad_sequence([torch.FloatTensor(data[f'Thermal temperature measurement {x}'][start:end-1].values)
                              for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy())

            # The temperature starts at the right value
        max_ = np.max([end - start for start, end in zip(starts, ends)])
        T = np.zeros((len(starts), max_))
        T[:, 0] = data[f'Thermal temperature measurement {self.room}'][starts]

        heating = [data.iloc[start:end]['Case'].mean() >= 0 for start, end in zip(starts, ends)]
        cooling = [data.iloc[start:end]['Case'].mean() < 0 for start, end in zip(starts, ends)]

        # Euler method for integration with 60 seconds time steps
        for t in range(max_ - 1):
            if len(heating) > 0:
                T[heating, t + 1] = T[heating, t] + self.dt * self.interval * (control[heating, t] * self.p[0] +
                                                                               irr[heating, t] * self.p[2] +
                                                                               (out[heating, t] - T[heating, t]) *
                                                                               self.p[3])
            if len(cooling) > 0:
                T[cooling, t + 1] = T[cooling, t] + self.dt * self.interval * (control[cooling, t] * self.p[1] +
                                                                               irr[cooling, t] * self.p[2] +
                                                                               (out[cooling, t] - T[cooling, t]) *
                                                                               self.p[3])
            for i in range(len(self.neigh)):
                T[:, t + 1] += self.dt * self.interval * ((neighs[i][:, t] - T[:, t]) * self.p[i + 4]).squeeze()

        T[np.where(np.abs(truth.mean(axis=2)) < 1e-6)] = 0

        return T[:,1:], truth


class UMARRCModel:

    def __init__(self, data, dt, threshold, interval, overlap=None):

        self.data = data
        self.dt = dt

        self.threshold_length = threshold
        self.overlap = overlap
        self.interval = interval
        self.create_sequences(threshold=self.threshold_length)

        self.p = np.array([1e-6] * 11).astype(float)

    def create_sequences(self, threshold: int = None, overlap: int = None, data=None, return_: bool = False) -> None:
        """
        Function to create lists of starts and ends of sequences of data that can be used (i.e.
        there is no missing data).

        Args:
            threshold:          Threshold on the sequences length (we use Euler's method, it cannot
                                  be extrapolated to too long sequences)
        """

        if overlap is None:
            overlap = self.overlap
        if threshold is None:
            threshold = self.threshold
        if data is None:
            data = self.data

        # Find the places with missing values
        nans = np.where(pd.isnull(self.data).any(axis=1))[0]

        # Define the starts and ends of sequences of data without missing values
        starts_ = list(nans[np.where(np.diff(nans) != 1)[0]] + 1)
        ends_ = list(nans[np.where(np.diff(nans) != 1)[0] + 1])

        # Go through the sequences and separate the ones that are too long in several chunks
        starts = []
        ends = []
        for start, end in zip(starts_, ends_):
            # Cut the sequence in chunk of `threshold` length
            while end - start > threshold:
                starts.append(start)
                ends.append(start + threshold + 1)
                start += overlap
            starts.append(start)
            ends.append(end)

        # Small correction to handle the start and end of the data correctly
        if nans[0] > 0:
            starts = [0] + starts
            ends = [nans[0]] + ends
        if nans[-1] < len(self.data) - 1:
            starts = starts + [nans[-1] + 1]
            ends = ends + [len(self.data) - 1]

        if return_:
            return starts, ends
        else:
            self.starts = starts
            self.ends = ends

    def predict(self, starts: int, ends: int, control: np.array = None, data: pd.DataFrame = None):
        """
        Function to predict the temperature over a given sequence of data (start, end)

        Args:
            starts:     Start of the sequences
            ends:       End of the sequences
            control:    Control sequence if another sequence than what is in the data is to be taken
            data:       Data to predict from
        """

        if isinstance(starts, int):
            starts = [starts]
        if isinstance(ends, int):
            ends = [ends]
        if data is None:
            data = self.data

        assert len(starts) == len(ends), 'Provide same number of starts and ends'

        truth = pad_sequence([torch.FloatTensor(data.iloc[start+1:end,:].astype(float).values) for start, end in zip(starts, ends)],
                             padding_value=0, batch_first=True).detach().numpy()

        if control is None:
            control = pad_sequence(
                [torch.FloatTensor(data[['Power room 272', 'Power room 273', 'Power room 274']][start:end-1].values)
                 for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()

        irr = pad_sequence([torch.FloatTensor(data['Weather solar irradiation'][start:end-1].values)
                            for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()
        out = pad_sequence([torch.FloatTensor(data['Weather outside temperature'][start:end-1].values)
                            for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()

        # The temperature starts at the right value
        max_ = np.max([end - start for start, end in zip(starts, ends)])
        T = np.zeros((len(starts), max_, 3))
        T[:, 0, :] = data[['Thermal temperature measurement 272', 'Thermal temperature measurement 273',
                           'Thermal temperature measurement 274']].iloc[starts, :]

        # Euler method for integration with 60 seconds time steps
        for t in range(max_ - 1):
            T[:, t + 1, 0] = T[:, t, 0] + self.dt * self.interval * (control[:, t, 0] * self.p[0] +
                                                                irr[:, t] * self.p[1] +
                                                                (out[:, t] - T[:, t, 0]) * self.p[2] +
                                                                (T[:, t, 1] - T[:, t, 0]) * self.p[3]).squeeze()

            T[:, t + 1, 1] = T[:, t, 1] + self.dt * self.interval * (control[:, t, 1] * self.p[4] +
                                                                irr[:, t] * self.p[5] +
                                                                (out[:, t] - T[:, t, 1]) * self.p[6] +
                                                                (T[:, t, 0] - T[:, t, 1]) * self.p[3] +
                                                                (T[:, t, 2] - T[:, t, 1]) * self.p[10]).squeeze()

            T[:, t + 1, 2] = T[:, t, 2] + self.dt * self.interval * (control[:, t, 2] * self.p[7] +
                                                                irr[:, t] * self.p[8] +
                                                                (out[:, t] - T[:, t, 2]) * self.p[9] +
                                                                (T[:, t, 1] - T[:, t, 2]) * self.p[10]).squeeze()

        T[np.where(np.abs(truth.mean(axis=2)) < 1e-6)] = 0

        return T[:,1:,:], truth

class UMARSepRCModel(UMARRCModel):

    def __init__(self, data, dt, threshold, interval, overlap=None):

        super().__init__(data, dt, threshold, interval, overlap=overlap)

        self.p = np.array([1e-6] * 14).astype(float)

        self._create_sequences(threshold=threshold, overlap=overlap)

    def _create_sequences(self, threshold: int = None, overlap=None) -> None:
        """
        Function to create lists of starts and ends of sequences of data that can be used (i.e.
        there is no missing data).

        Args:
            threshold:          Threshold on the sequences length (we use Euler's method, it cannot
                                  be extrapolated to too long sequences)
        """

        data = self.data.copy()
        data.iloc[np.where(self.data['Case'] < 0)[0], :] = np.nan
        starts, ends = self.create_sequences(data=data, threshold=threshold, overlap=overlap, return_=True)

        data = self.data.copy()
        data.iloc[np.where(self.data['Case'] > 0)[0], :] = np.nan
        starts_, ends_ = self.create_sequences(data=data, threshold=threshold, overlap=overlap, return_=True)

        self.starts = starts + starts_
        self.ends = ends + ends_

    def predict(self, starts: int, ends: int, control: np.array = None, data: pd.DataFrame = None):
        """
        Function to predict the temperature over a given sequence of data (start, end)

        Args:
            starts:     Start of the sequences
            ends:       End of the sequences
            control:    Control sequence if another sequence than what is in the data is to be taken
            data:       Data to predict from
        """

        if isinstance(starts, int):
            starts = [starts]
        if isinstance(ends, int):
            ends = [ends]
        if data is None:
            data = self.data

        assert len(starts) == len(ends), 'Provide same number of starts and ends'

        truth = pad_sequence(
            [torch.FloatTensor(data.iloc[start+1:end, :].astype(float).values) for start, end in zip(starts, ends)],
            padding_value=0, batch_first=True).detach().numpy()

        if control is None:
            control = pad_sequence(
                [torch.FloatTensor(data[['Power room 272', 'Power room 273', 'Power room 274']][start:end-1].values)
                 for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()

        irr = pad_sequence([torch.FloatTensor(data['Weather solar irradiation'][start:end-1].values)
                            for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()
        out = pad_sequence([torch.FloatTensor(data['Weather outside temperature'][start:end-1].values)
                            for start, end in zip(starts, ends)], padding_value=0, batch_first=True).detach().numpy()

        # The temperature starts at the right value
        max_ = np.max([end - start for start, end in zip(starts, ends)])
        T = np.zeros((len(starts), max_, 3))
        T[:, 0, :] = data[['Thermal temperature measurement 272', 'Thermal temperature measurement 273',
                           'Thermal temperature measurement 274']].iloc[starts, :]

        heating = [data.iloc[start:end]['Case'].mean() >= 0 for start, end in zip(starts, ends)]
        cooling = [data.iloc[start:end]['Case'].mean() < 0 for start, end in zip(starts, ends)]

        # Euler method for integration with 60 seconds time steps
        for t in range(max_-1):
            T[heating, t + 1, 0] = T[heating, t, 0] + self.dt * self.interval * (control[heating, t, 0] * self.p[0] +
                                                                irr[heating, t] * self.p[2] +
                                                                (out[heating, t] - T[heating, t, 0]) * self.p[3] +
                                                                (T[heating, t, 1] - T[heating, t, 0]) * self.p[4]).squeeze()

            T[heating, t + 1, 1] = T[heating, t, 1] + self.dt * self.interval * (control[heating, t, 1] * self.p[5] +
                                                                irr[heating, t] * self.p[7] +
                                                                (out[heating, t] - T[heating, t, 1]) * self.p[8] +
                                                                (T[heating, t, 0] - T[heating, t, 1]) * self.p[4] +
                                                                (T[heating, t, 2] - T[heating, t, 1]) * self.p[13]).squeeze()

            T[heating, t + 1, 2] = T[heating, t, 2] + self.dt * self.interval * (control[heating, t, 2] * self.p[9] +
                                                                irr[heating, t] * self.p[11] +
                                                                (out[heating, t] - T[heating, t, 2]) * self.p[12] +
                                                                (T[heating, t, 1] - T[heating, t, 2]) * self.p[13]).squeeze()

            T[cooling, t + 1, 0] = T[cooling, t, 0] + self.dt * self.interval * (control[cooling, t, 0] * self.p[1] +
                                                             irr[cooling, t] * self.p[2] +
                                                             (out[cooling, t] - T[cooling, t, 0]) * self.p[3] +
                                                             (T[cooling, t, 1] - T[cooling, t, 0]) * self.p[4]).squeeze()

            T[cooling, t + 1, 1] = T[cooling, t, 1] + self.dt * self.interval * (control[cooling, t, 1] * self.p[6] +
                                                             irr[cooling, t] * self.p[7] +
                                                             (out[cooling, t] - T[cooling, t, 1]) * self.p[8] +
                                                             (T[cooling, t, 0] - T[cooling, t, 1]) * self.p[4] +
                                                             (T[cooling, t, 2] - T[cooling, t, 1]) * self.p[13]).squeeze()

            T[cooling, t + 1, 2] = T[cooling, t, 2] + self.dt * self.interval * (control[cooling, t, 2] * self.p[10] +
                                                             irr[cooling, t] * self.p[11] +
                                                             (out[cooling, t] - T[cooling, t, 2]) * self.p[12] +
                                                             (T[cooling, t, 1] - T[cooling, t, 2]) * self.p[13]).squeeze()

        T[np.where(np.abs(truth.mean(axis=2)) < 1e-6)] = 0

        return T[:,1:,:], truth