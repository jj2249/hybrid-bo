"""Bayesian optimization of partially known systems with hybrid models.

The library solves one iteration; you write the loop, and it holds what a loop uses.
"""

from hybrid_bo import gp, plots
from hybrid_bo.acquisition import (
    SAA,
    SoftSAA,
    expected_improvement,
    hybrid_acquisition,
    standard_acquisition,
)
from hybrid_bo.dataset import Dataset, extend, incumbents, load_dataset, save_dataset
from hybrid_bo.errors import NoSolutionError, RootFindingError
from hybrid_bo.gp import fit_surrogate
from hybrid_bo.problem import Problem, System, Var, lhs_points, uniform_points
from hybrid_bo.reconstruction import Measure, reconstruct_states
