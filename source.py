#!/usr/bin/env python3

import pandas as pd
import requests
import types
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import numpy as np

class RegModel:
    def __init__(self):
        """ Initiates an untrained regression model."""
        self.slope = 0
        self.intercept = 0

    def predict(self, date):
        """Predict the temperature based on the date"""
        return self.slope * date + self.intercept

def cost_function(actual, estimated):
    """Computes the cost and difference between predicted and actual values"""
    # compute the difference
    difference = actual - estimated

    # compute the cost, i.e., difference squared
    cost = sum(difference ** 2)

    return difference, cost

# Create a model to be trained
model = RegModel()

py_url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0b_optimizer.py"
code = requests.get(py_url)

# error handling
if not code.ok:
    error_msg = f"Fetching url failed. Status code: {code.status_code}"
    raise RuntimeError(error_msg)

# create module
m0b_optimizer = types.ModuleType("m0b_optimizer")
exec(code.text, m0b_optimizer.__dict__)

# create an optimiser
optimiser = m0b_optimizer.MyOptimizer()

data_url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/seattleWeather_1948-2017.csv"

# import the weather data
data = pd.read_csv(data_url, parse_dates=["date"])

# filter out all months except January
data = data[[d.month == 1 for d in data.date]].copy()

# normalise the data
# number of years since 1982
data["years_since_1982"] = [(d.year + d.timetuple().tm_yday / 365.25) - 1982 for d in data.date]

# normalise temperatures
data["normalised_temperature"] = (data["min_temperature"] - np.mean(data["min_temperature"])) / np.std(data["min_temperature"])


# test:
#print(data.head(5))

# test plot:
plt.scatter(data["years_since_1982"], data["normalised_temperature"], marker=".")
plt.plot(data["years_since_1982"], model.predict(data["normalised_temperature"]), c="r")

# labels and legend
plt.xlabel("years_since_1982")
plt.ylabel("normalised_temperature")
plt.title("January Temperatures (Normalised)")

plt.savefig("plots/scatter_plot.pdf")
