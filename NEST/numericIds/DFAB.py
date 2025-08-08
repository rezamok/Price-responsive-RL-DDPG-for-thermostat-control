"""
File gathering how to download the data from the wanted sensors in DFAB, NEST
There is also the data on the occupancy of the rooms
"""

import pandas as pd

## Domestic hot water heat pump information:

dhw_hp_name = "DHW_HP"

dhw_hp_names = {
    "42190031": "HP electricity consumption",
    "421110178": "HP Boiler temperature",
    "421100067": "HP thermal power consumption",
    "421110179": "HP thermal COP"
}

dhw_hp_ids = list(dhw_hp_names.keys())

## Electricity information

electricity_name = "Electricity"

electricity_names = {
    "42190139": "Electricity total measurement",            # Doesn't exactly add up to the sum of the rest but that's the total measured (Sascha has the same error)
    "42190018": "HVAC consumption",                         # Careful: This includes heat pumps and stuff ('cheating' by telling when heating is on)
    "42190064": "Kitchen consumption",                      # To tell when occupied
    "42190108": "Lights and powerplugs consumption",        # To tell when occupied
    #"42190178": "Circulating air cooling",                 # Automatic (?) ventilation - no care
    #"42190119": "Reserve electricity",                     # Don't care
    #"42190000": "Emergency mains electricity consumption", # All the reserves here we don't care
    #"42190004": "Alarm electricity consumption",
    #"42190008": "Fire alarm electricity consumption",
    "42190053": "PV electricity production",                # This is different from the pattern observed in the total measurement but we can't do anything about it
                                                            # Maybe take the inverter measurement instead?
}

electricity_ids = list(electricity_names.keys())

## All the valves

valves_name = "Valves"

valves_names = {
    "421110008": "Valve 371 a",
    "421110009": "Valve 371 b",
    "421110010": "Valve 371 c",
    "421110011": "Valve 371 d",
    "421110012": "Valve 371 e",
    "421110013": "Valve 371 f",
    "421110014": "Valve 371 g",
    "421110023": "Valve 472 a",
    "421110024": "Valve 472 b",
    "421110025": "Thermal valves 474",
    "421110026": "Valve 476 a",
    "421110027": "Valve 476 b",
    "421110028": "Valve 476 c",
    "421110029": "Valve 472 c",
    "421110038": "Valve 571 a",
    "421110039": "Valve 574 a",
    "421110040": "Valve 574 b",
    "421110041": "Valve 574 c",
    "421110042": "Thermal valves 573",
    "421110043": "Valve 571 b",
    "421110044": "Valve 571 c",
    "421100174": "Thermal volume flow",  # m^3/h
    "421100172": "Thermal total power",
    "421100067": "HP thermal power consumption",
}

valves_ids = list(valves_names.keys())

## Thermal information

# plt.plot(thermal_df['Thermal total power'][:100] / (4.1796 * 10**6 / 1000 * thermal_df['Thermal volume flow'] / 3600
#          * (thermal_df['Thermal temperature high'] - thermal_df['Thermal temperature low']))[:100])
# with heat capacity of water 4.1796 J/cm3/K, flow in m^3/h and power in kW
# roughly equals 1 (almost always between 0.99 and 1) so that's how the power is computed

thermal_name = "Thermal"

thermal_names = {
    "421100172": "Thermal total power",
    "421100168": "Thermal inlet temperature",               # Not at the place where the power is measured
    "421100170": "Thermal outlet temperature",              # Before the HP, not at the place where the power is measured
    "421100174": "Thermal volume flow",                     # m^3/h
    #"421100175": "Thermal temperature high",                # At the power meter - used to compute power
    "401180625": "Thermal temperature high",                # At the pump
    #"421100176": "Thermal temperature low",                 # At the power meter - used to computed power
    "401180849": "Thermal temperature low",                 # At the pump
    "421110001": "Thermal inlet temperature 371",
    "421110048": "Thermal temperature measurement 371",
    "421110016": "Thermal inlet temperature 472 474 476",
    "421110054": "Thermal temperature measurement 472",
    "421110060": "Thermal temperature measurement 474",
    "401160010": "Thermal temperature measurement 476",
    "421110031": "Thermal inlet temperature 571 573 574",
    "421110072": "Thermal temperature measurement 571",
    "421110078": "Thermal temperature measurement 573",
    "421110084": "Thermal temperature measurement 574",
    "421100067": "HP thermal power consumption",
    #"401180625": "Thermal hot water forward",
    #"421100067": "Thermal hot water low",
    #"421100067": "Thermal cold water high",
    #"421100067": "Thermal cold water low",
}

thermal_ids = list(thermal_names.keys())

## Occupancy data
# TODO more precise dates ?

occupancy_start_472 = pd.Timestamp(2019, 4, 30)
occupancy_start_571 = pd.Timestamp(2019, 6, 4)
occupancy_start_574 = pd.Timestamp(2019, 4, 30)

occupancy_end_472 = pd.Timestamp(2020, 1, 21)
occupancy_end_571 = pd.Timestamp(2019, 12, 23)
occupancy_end_574 = pd.Timestamp(2020, 2, 25)
