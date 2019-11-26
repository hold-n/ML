import numpy as np
import pandas as pd


def cov_matrix(x, is_scaled):
    scaled = x if is_scaled else x - x.mean()
    return 1 / len(x) * (scaled.T @ scaled)


def pca(x, k, is_scaled=True):
    c = cov_matrix(x, is_scaled)
    # TODO: implement selection of k
    u, _, _ = np.linalg.svd(c)
    uk = u[:, :k]
    return (x @ uk), uk.T


def pca_inverse(x, uk):
    return x @ uk
