#!/usr/bin/env python3

import pandas as pd
import requests
import types
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import numpy as np

# import module from Microsoft
py_url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0b_optimizer.py"
code = requests.get(py_url)
if not code.ok: # handle errors
    error_msg = f"Fetching url failed. Status code: {code.status_code}"
    raise RuntimeError(error_msg) 
m0b_optimizer = types.ModuleType("m0b_optimizer")
exec(code.text, m0b_optimizer.__dict__)

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

# create an optimiser
optimiser = m0b_optimizer.MyOptimizer()

def train_one_iteration(inputs, true_temperatures, last_cost:float):
    """
    Runs one iteration of training.

    Args:
        inputs: dates for the model to predict based upon.

        true_temperatures: actual temperatures.

        last_cost (float): cost of prediction on previous training.

    Out:
        bool: whether training should continue or not.

        float: cost after this iteration of training.
    """

    # get a prediction with current model
    estimated_temperatures = model.predict(inputs)
    
    # calculate the performance of the model
    difference, cost = cost_function(true_temperatures, estimated_temperatures)

    # decide whether training should continue
    if cost >= last_cost:
        # end training
        return False, cost
    else:
        intercept_update, slope_update = optimiser.get_parameter_updates(inputs, cost, difference)

        # update parameters
        model.intercept -= intercept_update
        model.slope -= slope_update

        return True, cost

def train(inputs, true_temperatures):
    """Trains the model until optimal.

    Args:
        inputs: dates for the model to predict based upon.

        true_temperatures: actual temperatures.

    Out: 
        None
    """
    last_cost = np.inf
    i = 0
    continue_loop = True

    while continue_loop:
        continue_loop, last_cost = train_one_iteration(inputs=data["years_since_1982"],
                                                       true_temperatures=data["normalised_temperature"],
                                                       last_cost=last_cost
                                                    )
        if i % 277 == 0:
            print(f"iteration: {i}")

        i += 1

if __name__ == "__main__":
    # import the weather data
    data_url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/seattleWeather_1948-2017.csv"
    data = pd.read_csv(data_url, parse_dates=["date"])

    # filter out all months except January
    data = data[[d.month == 1 for d in data.date]].copy()

    # optimise data for ML:
    # number of years since 1982
    data["years_since_1982"] = [(d.year + d.timetuple().tm_yday / 365.25) - 1982 for d in data.date]
    # normalise temperatures
    data["normalised_temperature"] = (data["min_temperature"] - np.mean(data["min_temperature"])) / np.std(data["min_temperature"])

    # train the model
    train(data["years_since_1982"], data["normalised_temperature"])

    # plot the data and model predictions
    plt.scatter(data["years_since_1982"], data["normalised_temperature"], marker=".")
    plt.plot(data["years_since_1982"], model.predict(data["years_since_1982"]), c="r")

    # labels and legend
    plt.xlabel("years_since_1982")
    plt.ylabel("normalised_temperature")
    plt.title("January Temperatures (Normalised)")

    # save plot as pdf
    plt.savefig("plots/scatter_plot.pdf")
