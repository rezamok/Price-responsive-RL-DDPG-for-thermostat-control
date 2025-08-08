"""
File gathering how to download the data from the wanted sensors in UMAR.
Separated by type, ie. domestic hot water, electricity, valves, thermal and rooms data, because all of them
will be loaded and treated differently before being put together.
"""

## Domestic hot water heat pump information:

dhw_hp_name = "DHW_HP"

dhw_hp_cop = 4

dhw_hp_names = {"42150262": "HP thermal consumption", "42150242": "HP Boiler temperature"}

dhw_hp_ids = list(dhw_hp_names.keys())

## Electricity information

electricity_name = "Electricity"

pv_scale_factor = 2

electricity_names = {"42150423": "Electricity total measurement", "42190053": "Electricity PV production"}

electricity_ids = list(electricity_names.keys())

## All the valves

valves_name = "Valves"

valves_names = {
    "42150270": "Thermal valve 272",
    "42150271": "Thermal valve 273",  # There are 3 but always the same
    "42150274": "Thermal valve 274",
    "42150275": "Thermal valve 275",
    "42150276": "Thermal valve 276",
}

valves_ids = list(valves_names.keys())

## Thermal information of the rooms

thermal_name = "Thermal"

thermal_names = {
    "42150477": "Thermal heating power",
    "42150459": "Thermal cooling power",
    "42150221": "Thermal inlet temperature",
    "42150288": "Thermal temperature measurement 272",
    "42150300": "Thermal temperature measurement 273",
    "42150312": "Thermal temperature measurement 274",
    "42150324": "Thermal temperature measurement 275",
    "42150336": "Thermal temperature measurement 276",
}

thermal_ids = list(thermal_names.keys())

## Additional information of the rooms

rooms_name = "Rooms"

rooms_names = {
    "42150284": "Room 272 window",
    "42150289": "Room 272 humidity",
    "42150291": "Room 272 brightness",
    "42150285": "Room 273 window a",
    "42150286": "Room 273 window b",
    "42150301": "Room 273 humidity",
    "42150303": "Room 273 brightness",
    "42150287": "Room 274 window",
    "42150313": "Room 274 humidity",
    "42150315": "Room 274 brightness",
}

rooms_ids = list(rooms_names.keys())

## Add occupancy data if wanted, see DFAB
