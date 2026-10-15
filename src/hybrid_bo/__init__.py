"""Bayesian optimization of partially known systems with hybrid models.

The top level holds what a typical script uses: define a problem, run a method, look
at the results. Everything else is reachable through its module (`hb.gp.fit_gp`,
`hb.solvers.multistart`).
"""

# Every submodule, so type checkers see `hb.gp.fit_gp` after `import hybrid_bo as hb`
from hybrid_bo import (
    acquisition,
    baselines,
    bo,
    errors,
    evaluation,
    gp,
    history,
    kernels,
    plots,
    problem,
    run,
    sampling,
    saving,
    solvers,
    studies,
    views,
)
from hybrid_bo.acquisition import EI, LCB
from hybrid_bo.baselines import latin_hypercube_sampling, uniform_sampling
from hybrid_bo.bo import HybridBOSettings, StandardBOSettings, hybrid_bo, standard_bo
from hybrid_bo.errors import (
    HybridBOError,
    NoSolutionError,
    ReconstructionWarning,
    RootFindingError,
)
from hybrid_bo.evaluation import Experiment, oracle_experiment, oracle_optimum
from hybrid_bo.history import History, incumbents, oracle_incumbents
from hybrid_bo.problem import Problem, System, Var
from hybrid_bo.run import Run
from hybrid_bo.sampling import initial_points
from hybrid_bo.saving import load, save
from hybrid_bo.studies import Study, StudySettings, repeat, study
from hybrid_bo.views import evaluations, hyperparameters, summary
