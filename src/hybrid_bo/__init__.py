"""Bayesian optimization of partially known systems with hybrid models.

The top level holds what a typical script uses: define a problem, run a method, look
at the results. Everything else is reachable through its module (`hb.gp.fit_gp`,
`hb.solvers.solve_with_multistart`).
"""

# Every submodule, so type checkers see `hb.gp.fit_gp` after `import hybrid_bo as hb`
from hybrid_bo import (
    acquisition,
    baselines,
    bo,
    dataset,
    errors,
    evaluation,
    gp,
    hybrid_model,
    oracle,
    plots,
    problem,
    run,
    sampling,
    saving,
    settings,
    solvers,
)
from hybrid_bo.acquisition import EI, LCB, SAA, Linearization, SoftSAA
from hybrid_bo.baselines import latin_hypercube_sampling, uniform_sampling
from hybrid_bo.bo import hybrid_bo, standard_bo
from hybrid_bo.dataset import (
    Dataset,
    check_oracle_optimum,
    incumbents,
    oracle_incumbents,
)
from hybrid_bo.errors import (
    HybridBOError,
    NoSolutionError,
    OracleOptimumWarning,
    OutOfBoundsWarning,
    ReconstructionWarning,
    ReplayWarning,
    RootFindingError,
)
from hybrid_bo.evaluation import Measure, reconstruct
from hybrid_bo.hybrid_model import fit_surrogate, optimize_hybrid_model
from hybrid_bo.oracle import oracle_measure, oracle_optimum
from hybrid_bo.problem import Problem, System, Var
from hybrid_bo.run import Run
from hybrid_bo.sampling import initial_points
from hybrid_bo.saving import load_run, load_runs, save_run, save_runs
