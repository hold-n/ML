import pandas as pd
import numpy as np


def create_design_matrix(*xs):
    ones = pd.Series(1, index=np.arange(len(xs[0])))
    result = pd.concat([ones, *xs], axis="columns")
    result.columns = np.arange(len(result.columns))
    return result


def iterate_theta(gradient, theta, alpha):
    while True:
        yield theta
        theta = theta - alpha * gradient(theta)


def logistic_loss_gradient(theta, x, y):
    # NOTE: x and theta should be of the same size: assuming a row of 1's in x
    len_x = len(x)
    if len_x != len(y):
        raise ValueError("Series must be of the same length")
    sample_items = x.transpose() * (x.dot(theta) - y).values
    return 1 / len_x * sample_items.apply(np.sum, axis="columns")


def normalize(x):
    def _normalize(row):
        mean = row.mean()
        width = row.max() - row.min()
        return row.map(lambda item: (item - mean) / width)

    return x.apply(_normalize, axis="rows")
