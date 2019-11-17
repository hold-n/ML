import math

import pandas as pd
import numpy as np


def create_design_matrix(x, normalize=True):
    ones = pd.Series(1, index=np.arange(len(x)))
    new_x, means, widths = _normalize(x) if normalize else (x, 0, 1)
    means = np.insert(means, 0, 0)
    widths = np.insert(widths, 0, 1)
    result = pd.concat(
        [ones, new_x], axis="columns", names=(["dummy"].extend(x.columns))
    )
    return result, means, widths


def denormalize_known(x, means, widths, index=None):
    if index is None:
        return widths * x + means
    return widths[index] * x + means[index]


def gradient_descent(theta, x, y, alpha):
    gradient = lambda _theta: logistic_loss_gradient(_theta, x, y)
    for current_theta in iterate_theta(gradient, theta, alpha):
        loss = logistic_loss(current_theta, x, y)
        yield current_theta, loss


def iterate_theta(gradient, theta, alpha):
    while True:
        yield theta
        theta = theta - alpha * gradient(theta)


def logistic_cost(theta, x, y):
    prediction = logistic_hypothesis(x, theta)
    return -np.log(y * prediction + (1 - y) * (1 - prediction))


def logistic_hypothesis(x, theta):
    return sigmoid(np.dot(x, theta))


def logistic_loss(theta, x, y):
    return 1 / len(x) * logistic_cost(theta, x, y).sum()


def logistic_loss_gradient(theta, x, y):
    return (1 / len(x)) * np.dot(x.transpose(), logistic_hypothesis(x, theta) - y)


def normalize_known(x, means, widths, index=None):
    if index is None:
        return (x - means) / widths
    return (x - means[index]) / widths[index]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _normalize(x):
    means = [x[c].mean() for c in x.columns]
    # TODO: consider width = 0
    widths = [np.ptp(x[c]) for c in x.columns]
    result = (x - means) / widths
    return result, means, widths
