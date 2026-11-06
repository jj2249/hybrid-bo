"""Bayesian optimization of partially known systems with hybrid models.

The library solves one iteration; you write the loop, and it holds what a loop uses.
"""

from hybrid_bo import gp, plots
from hybrid_bo.acquisition import (
    SAA,
    SoftSAA,
    hybrid_trial_point,
    saa_scenarios,
    standard_trial_point,
)
from hybrid_bo.dataset import Dataset, extend, incumbents, load_dataset, save_dataset
from hybrid_bo.errors import HybridBOError, NoSolutionError, RootFindingError
from hybrid_bo.evaluation import Measure, reconstruct_states
from hybrid_bo.gp import fit_gp
from hybrid_bo.hybrid_model import fit_surrogate
from hybrid_bo.problem import Problem, System, Var
from hybrid_bo.sampling import lhs_points, uniform_points
