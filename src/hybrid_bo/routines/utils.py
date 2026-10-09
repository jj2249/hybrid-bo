from typing import overload

import casadi as cas
import numpy as np
import scipy

from hybrid_bo.config import Config
from hybrid_bo.type_aliases import CasadiType


def get_sampling_based_data(
    config: Config,
    n_points_complete: list[int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:

    measurement_noise_complete: list[np.ndarray] = [
        np.atleast_2d(
            scipy.stats.norm.rvs(
                0,
                config.std_measurement_noise,
                n_points_complete[i_run_bo],
                random_state=config.rng,
            ),
        ).T
        for i_run_bo in range(config.n_runs_bo)
    ]

    gp_gaussian_standard_samples_complete: list[np.ndarray] = [
        np.atleast_2d(
            scipy.stats.norm.rvs(
                0,
                1,
                config.n_samples_gp,
                random_state=config.rng,
            ),
        ).T
        for i_run_bo in range(config.n_runs_bo)
    ]

    return measurement_noise_complete, gp_gaussian_standard_samples_complete


@overload
def pdf_normal(x: cas.SX) -> cas.SX: ...
@overload
def pdf_normal(x: cas.MX) -> cas.MX: ...
@overload
def pdf_normal(x: cas.DM) -> cas.DM: ...
@overload
def pdf_normal(x: np.ndarray) -> np.ndarray: ...


def pdf_normal(x: CasadiType | np.ndarray) -> CasadiType | np.ndarray:
    pdf: CasadiType | np.ndarray
    if isinstance(x, CasadiType):
        pdf = cas.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    else:
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    return pdf


@overload
def cdf_normal(x: cas.SX) -> cas.SX: ...
@overload
def cdf_normal(x: cas.MX) -> cas.MX: ...
@overload
def cdf_normal(x: cas.DM) -> cas.DM: ...
@overload
def cdf_normal(x: np.ndarray) -> np.ndarray: ...


def cdf_normal(x: CasadiType | np.ndarray) -> CasadiType | np.ndarray:
    cdf: CasadiType | np.ndarray
    if isinstance(x, CasadiType):
        cdf = 0.5 * (1 + cas.erf(x / np.sqrt(2)))
    else:
        cdf = 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))
    return cdf


@overload
def ei_gp(
    mean: cas.SX,
    std: cas.SX,
    incumbent: float,
    maximize: bool = True,
    epsilon: float = 1.0e-10,
) -> cas.SX: ...
@overload
def ei_gp(
    mean: cas.MX,
    std: cas.MX,
    incumbent: float,
    maximize: bool = True,
    epsilon: float = 1.0e-10,
) -> cas.MX: ...
@overload
def ei_gp(
    mean: cas.DM,
    std: cas.DM,
    incumbent: float,
    maximize: bool = True,
    epsilon: float = 1.0e-10,
) -> cas.DM: ...
@overload
def ei_gp(
    mean: np.ndarray,
    std: np.ndarray,
    incumbent: float,
    maximize: bool = True,
    epsilon: float = 1.0e-10,
) -> np.ndarray: ...


def ei_gp(
    mean: CasadiType | np.ndarray,
    std: CasadiType | np.ndarray,
    incumbent: float,
    maximize: bool = True,
    epsilon: float = 1.0e-10,
) -> CasadiType | np.ndarray:
    """Returns expected improvement for BO with Gaussian processes as surrogate model.

    Remember: max(a, b) = - min(-a, -b)
    ei_gp(maximize=True) = E[max(0, f - incumbent)] = - E[min(0, incumbent - f)]
    ei_gp(maximize=False) = E[max(0, incumbent - f)] = - E[min(0, f - incumbent)]

    If we want to maximize a function f,
    we use the greatest observed f as incumbent
    and the acquisition problem reads
    either max ei_gp(maximize=True)
    or min -ei_gp(maximize=True).

    If we want to maximize a function f,
    we use the least observed f as incumbent
    and the acquisition problem reads
    either max ei_gp(maximize=False)
    or min -ei_gp(maximize=False)

    Parameters
    ----------
    mean : CasadiType | np.ndarray
        Mean of the Gaussian process
    std : CasadiType | np.ndarray, m x n
        Standard deviation of the Gaussian process
    incumbent : float
        Best observation of the function to minimize so far
    maximize : bool, optional
        Determines if dealing with a maximization (True) or minimization (False) problem.
        By default True
    epsilon : float, optional
        Sometimes added to the standard deviation to prevent devision by zero.
        By default 1.0e-10

    Returns
    -------
    CasadiType | np.ndarray
        Expected improvement

    See Also
    --------
    https://botorch.org/docs/acquisition#analytic-acquisition-functions for maximization problem.
    https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html?utm_source=chatgpt.com#ego for minimization problem.
    """

    if mean.shape != std.shape:
        msg: str = "mean.shape != std.shape"
        raise Exception(msg)

    sign: float = 1.0
    if not maximize:
        sign = -1.0

    gamma: CasadiType | np.ndarray = sign * (mean - incumbent) / (std + epsilon)
    ei: CasadiType | np.ndarray = sign * (mean - incumbent) * cdf_normal(
        gamma,
    ) + std * pdf_normal(gamma)

    return ei
