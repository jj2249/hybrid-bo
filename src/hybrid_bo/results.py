from dataclasses import dataclass, field

import numpy as np

from hybrid_bo.affine_transformers import AffineTransformer
from hybrid_bo.gp import GP


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


@dataclass
class ResultsGP:
    input_transformer_initial: AffineTransformer
    output_transformer_initial: AffineTransformer
    gp_initial: GP

    input_transformers_bo: list[AffineTransformer] = field(default_factory=list)
    output_transformers_bo: list[AffineTransformer] = field(default_factory=list)
    gps_bo: list[GP] = field(default_factory=list)
