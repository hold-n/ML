import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics.pairwise import euclidean_distances


def gaussian(sigma_squared):
    def kernel(x, y):
        distances = euclidean_distances(x, y)
        return np.exp(-distances ** 2 / 2 / sigma_squared)

    return kernel
