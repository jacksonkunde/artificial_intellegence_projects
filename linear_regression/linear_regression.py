# Jackson Kunde

import sys
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
    
def make_plot(csv_name):
    df = pd.read_csv(csv_name)
    columns = list(df.columns)
    plt.plot(df[columns[0]], df[columns[1]])
    plt.xticks(df[columns[0]])
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=10)

    plt.savefig("plot.jpg")
    return df

# computing the equation of the linear regression.
def train(X, Y):
    Z = np.dot(np.transpose(X), X)
    I = np.linalg.inv(Z)
    PI = np.dot(I, np.transpose(X))
    hat_beta = np.dot(PI, Y)
    return hat_beta
    
def predict(hat_beta, x_test):
    return np.dot(np.transpose(hat_beta), x_test)