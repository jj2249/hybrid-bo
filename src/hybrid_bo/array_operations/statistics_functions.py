from typing import overload

import casadi as cas
import numpy as np

from hybrid_bo.type_aliases import CasadiType


@overload
def sample_mean(samples: cas.SX) -> cas.SX: ...
@overload
def sample_mean(samples: cas.MX) -> cas.MX: ...
@overload
def sample_mean(samples: cas.DM) -> cas.DM: ...
@overload
def sample_mean(samples: np.ndarray) -> np.ndarray: ...


def sample_mean(samples: CasadiType | np.ndarray) -> CasadiType | np.ndarray:
    # Every row of parameter samples represents a sample

    if len(samples.shape) != 2:
        msg: str = "len(samples.shape) != 2"
        raise Exception(msg)

    if isinstance(samples, CasadiType):
        return cas.sum1(samples) / samples.shape[0]
    return np.mean(samples, 0)[np.newaxis, :]


@overload
def sample_covariance(samples: cas.SX, ddof: int = 0) -> cas.SX: ...
@overload
def sample_covariance(samples: cas.MX, ddof: int = 0) -> cas.MX: ...
@overload
def sample_covariance(samples: cas.DM, ddof: int = 0) -> cas.DM: ...
@overload
def sample_covariance(samples: np.ndarray, ddof: int = 0) -> np.ndarray: ...


def sample_covariance(
    samples: CasadiType | np.ndarray,
    ddof: int = 0,
) -> CasadiType | np.ndarray:
    # Every row of parameter samples represents a sample
    # Parameter ddof: see numpy.var()

    if len(samples.shape) != 2:
        msg: str = "len(samples.shape) != 2"
        raise Exception(msg)

    mean: CasadiType | np.ndarray = sample_mean(samples)

    covariance: CasadiType | np.ndarray
    if isinstance(samples, CasadiType):
        covariance = type(samples).zeros(samples.shape[1], samples.shape[1])
    else:
        covariance = np.zeros((samples.shape[1], samples.shape[1]))

    for i_sample in range(samples.shape[0]):
        covariance += (samples[i_sample, :] - mean).T @ (samples[i_sample, :] - mean)
    covariance /= samples.shape[0] - ddof

    return covariance


@overload
def sample_variance(samples: cas.SX, ddof: int = 0) -> cas.SX: ...
@overload
def sample_variance(samples: cas.MX, ddof: int = 0) -> cas.MX: ...
@overload
def sample_variance(samples: cas.DM, ddof: int = 0) -> cas.DM: ...
@overload
def sample_variance(samples: np.ndarray, ddof: int = 0) -> np.ndarray: ...


def sample_variance(
    samples: CasadiType | np.ndarray,
    ddof: int = 0,
) -> CasadiType | np.ndarray:
    # Every row of parameter samples represents a sample
    # Parameter ddof: see numpy.var()

    if len(samples.shape) != 2:
        msg: str = "len(samples.shape) != 2"
        raise Exception(msg)

    variance: CasadiType | np.ndarray
    if isinstance(samples, CasadiType):
        mean: CasadiType = sample_mean(samples)
        mean_tiled: CasadiType = cas.repmat(mean, samples.shape[0], 1)
        samples_minus_mean_squared: CasadiType = (samples - mean_tiled) ** 2
        variance = cas.sum1(samples_minus_mean_squared) / (samples.shape[0] - ddof)

    else:
        variance = np.var(samples, 0, ddof=ddof)[np.newaxis, :]

    return variance
