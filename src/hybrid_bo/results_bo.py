from dataclasses import dataclass

import numpy as np


@dataclass
class ResultsBO:
    u_initial: np.ndarray
    x_initial: np.ndarray
    f_initial: np.ndarray
    incumbents_initial: np.ndarray
    x_no_noise_initial: np.ndarray
    f_no_noise_initial: np.ndarray
    incumbents_no_noise_initial: np.ndarray

    u_bo: np.ndarray
    x_bo: np.ndarray
    f_bo: np.ndarray
    incumbents_bo: np.ndarray
    x_no_noise_bo: np.ndarray
    f_no_noise_bo: np.ndarray
    incumbents_no_noise_bo: np.ndarray
