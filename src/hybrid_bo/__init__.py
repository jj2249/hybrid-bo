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
    plots,
    problem,
    run,
    sampling,
    saving,
    settings,
    solvers,
    studies,
    views,
)
from hybrid_bo.acquisition import EI, LCB, SAA, Linearization, SoftSAA
from hybrid_bo.baselines import latin_hypercube_sampling, uniform_sampling
from hybrid_bo.bo import hybrid_bo, standard_bo
from hybrid_bo.errors import (
    HybridBOError,
    NoSolutionError,
    OracleOptimumWarning,
    ReconstructionWarning,
    RootFindingError,
)
from hybrid_bo.evaluation import Measure, oracle_measure, oracle_optimum
from hybrid_bo.history import History, incumbents, oracle_incumbents
from hybrid_bo.problem import Problem, System, Var
from hybrid_bo.run import Run
from hybrid_bo.sampling import initial_points
from hybrid_bo.saving import load, load_run, save, save_run
from hybrid_bo.settings import HybridBOSettings, StandardBOSettings
from hybrid_bo.studies import (
    Configuration,
    Study,
    StudySettings,
    Task,
    assemble,
    repeat,
    run_task,
    study,
    study_tasks,
)
from hybrid_bo.views import evaluations, hyperparameters, summary
