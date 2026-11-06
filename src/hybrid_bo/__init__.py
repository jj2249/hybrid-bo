"""Bayesian optimization of partially known systems with hybrid models.

The library solves one iteration; you write the loop, and it holds what a loop uses.
"""

from hybrid_bo import gp, plots
from hybrid_bo.acquisition import (
    SAA,
    SoftSAA,
    hybrid_acquisition,
    hybrid_trial_point,
    replay,
    saa_scenarios,
    standard_acquisition,
    standard_trial_point,
)
from hybrid_bo.dataset import Dataset, extend, incumbents, load_dataset, save_dataset
from hybrid_bo.errors import NoSolutionError, RootFindingError
from hybrid_bo.evaluation import Measure, reconstruct_states
from hybrid_bo.gp import fit_gp
from hybrid_bo.hybrid_model import fit_surrogate
from hybrid_bo.problem import Problem, System, Var, lhs_points, uniform_points
