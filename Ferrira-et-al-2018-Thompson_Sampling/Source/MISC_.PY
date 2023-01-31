import numpy as np
from typing import Union


def normpdf(x: Union[float, np.ndarray], mean: Union[float, np.ndarray], Sigma: Union[float, np.ndarray]) -> float:
    """Given the point, mean and the covariance matrix, calculate the probability

    Args:
        x (Union[float, np.ndarray]): the point
        mean (Union[float, np.ndarray]): mean value of normal distribution
        Sigma (Union[float, np.ndarray]): covariance matrix of normal distribution

    Returns:
        float: the probability of the normal distribution
    """
    assert type(x) == float or type(x) == np.float64 or len(x.shape) == 1, "The point of normal prob should be a scalr or a vector"
    assert type(mean) == float or type(mean) == np.float64 or len(mean.shape) == 1, "The mean of normal prob should be a scalr or a vector"
    assert type(Sigma) == float or type(Sigma) == np.float64 or len(Sigma.shape) == 2, "The covariance of normal prob should be a scalr or a vector"

    if type(x) == float or type(x) == np.float64:
        assert type(mean) == float or type(x) == np.float64, "The mean of normal prob is not a scalar"
        assert type(Sigma) == float or type(mean) == np.float64, "The variance of normal prob is not a scalar"
        assert Sigma > 0, "The standard variance should be greater than 0"
        prob = 1 / np.sqrt(2 * np.pi) / Sigma * np.exp(-((x - mean) ** 2) / 2 / Sigma**2)
        return prob
    else:
        assert len(mean.shape) == 1, "The mean of normal prob is not a vector"
        assert len(Sigma.shape) == 2, "The covariance matrix of normal prob is not a matrix"
        assert Sigma.shape[0] == Sigma.shape[1], "The covariance matrix of normal prob is not a square matrix"
        d = len(x)
        inverse_Sigma__x_minus_mean = np.linalg.solve(Sigma, x - mean)
        prob = 1 / np.sqrt((2 * np.pi) ** d * np.abs(np.linalg.det(Sigma))) * np.exp(-(x - mean).dot(inverse_Sigma__x_minus_mean) / 2)
