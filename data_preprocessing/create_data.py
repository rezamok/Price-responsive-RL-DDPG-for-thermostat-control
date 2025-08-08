"""
File to run to download, preprocess and clean the data from NEST --> everything is put into form
to be used later easily and fast (through local saves)

You can provide a starting and an ending date from which you would like to download the data.

The sensors from which the data is downloaded can be modified in the "data_from_NEST.py" file
"""

# To make the modules visible
import sys
sys.path.append("../")

from data_preprocessing.dataset import prepare_dfab_data

start_date = '2020-01-01'
end_date = '2020-01-15'
verbose = 2

if __name__ == "__main__":

    _ = prepare_dfab_data(start_date, end_date, verbose=2)