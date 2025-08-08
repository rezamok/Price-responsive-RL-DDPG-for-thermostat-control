"""
File for the weather sensors
"""

weather_name = "Weather"

weather_ids = [
    3200000,  # Temperature
    3200002,  # Rel. humidity
    3200008,  # Global solar irradiation
    3200004,  # Wind speed
    3200006,  # Wind direction
    3200017,  # Relative air pressure
]

weather_names = {
    "3200000": "Weather outside temperature",
    "3200002": "Weather relative humidity",
    "3200008": "Weather solar irradiation",
    "3200004": "Weather wind speed",
    "3200006": "Weather wind direction",
    "3200017": "Weather relative air pressure",
}
