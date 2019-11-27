import numpy as np
import pandas as pd


def choose_threshold(x, mean, variance, y, epsilon):
    return max(epsilon, key=lambda e: f1_score(predict(x, mean, variance, e), y))


def f1_score(prediction, actual):
    p = precision(prediction, actual)
    r = recall(prediction, actual)
    return 2 * p * r / (p + r)


def gaussian(x, mean, variance):
    factor = 1 / np.sqrt(2 * np.pi * variance)
    return factor * np.exp(-(x - mean) ** 2 / 2 / variance)


def gaussian_params(x):
    mean = 1 / (len(x) - 1) * x.sum(axis=0)
    variance = 1 / len(x) * ((x - mean) ** 2).sum(axis=0)
    return np.array(mean), np.array(variance)


def precision(prediction, actual):
    positive = actual[actual == 1]
    positive_prediction = prediction[positive.index]
    return positive_prediction.sum() / prediction.sum()


def predict(x, mean, variance, epsilon):
    p = probability(x, mean, variance)
    return (p < epsilon).astype(int)


def probability(x, mean, variance):
    return np.product(gaussian(x, mean, variance), axis=1)


def recall(prediction, actual):
    positive = actual[actual == 1]
    positive_prediction = prediction[positive.index]
    return positive_prediction.sum() / actual.sum()
