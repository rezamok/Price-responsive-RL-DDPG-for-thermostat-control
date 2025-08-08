## Data pre-processing

Tested with mac only.

To create your data:
* Modify the starting and ending date of the data collection in *create_data.py*
* If needed: modify the list of sensors from which to download the data in *data_from_NEST.py*
  * Warning: This is NOT robust at the moment, all the names and lists provided in *data_from_NEST.py* should be defined and the preprocessing has some dependencies to them that could break (see `prepare_data()` in *dataset.py*)
* Run *create_data.py* from the terminal