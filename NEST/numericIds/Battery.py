"""
File gathering the sensors corresponding to the static battery from NEST
"""

## Battery information:

battery_name = "Battery"

battery_names = {"40200005": "Battery power input", "40200017": "Battery power measurement", "40200019": "Battery SoC"}

battery_ids = list(battery_names.keys())
