#!/usr/bin/env python3

import pandas as pd
import requests
import types

py_url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0b_optimizer.py"
code = requests.get(py_url)

# error handling
if not code.ok:
    error_msg = f"Fetching url failed. Status code: {code.status_code}"
    raise RuntimeError(error_msg)

# create module
m0b_optimizer = types.ModuleType("m0b_optimizer")
exec(code.text, m0b_optimizer.__dict__)

data_url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/seattleWeather_1948-2017.csv"

# import the weather data
data = pd.read_csv(data_url, parse_dates=["date"])

# filter out all months except January
data = data[[d.month == 1 for d in data.date]].copy()

# test:
print(data.head(5))
