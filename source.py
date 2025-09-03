#!/usr/bin/env python3

import pandas as pd
import requests
import types

data_url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/seattleWeather_1948-2017.csv"

py_url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0b_optimizer.py"

# import the weather data
data = pd.read_csv(data_url, parse_dates=["date"])

# filter out all months except January
data = data[[d.month == 1 for d in data.date]].copy()

# test:
print(data.head(5))
