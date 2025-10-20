from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    # %% Problem

    # Number of elements in u
    n_u: int

    # Number of elements in x
    n_x: int

    # Indices of the elements of x that are measured
    indices_x_measured: list[int]

    # Indices of the elements of u that are inputs of the GP
    # We use lists because indexing with tuples sometimes is different from indexing with lists in NumPy as well as CasADi
    indices_u_input_gp: list[int]

    # Indices of the elements of x that are inputs of the GP
    indices_x_input_gp: list[int]

    # Indices of the elements of x that are outputs of the GP
    indices_x_output_gp: list[int]

    # Number of starting points the training of the GP is executed with
    n_starts_training_gp: int = 8

    # Standard deviation of the measurement noise
    std_measurement_noise: float = 0.0

    # Configure if scipy.optimize.fsolve() uses a Jacobian as a parameter when evaluating the problem
    use_jacobian_evaluate_problem: bool = True

    # Configure if evaluating the problem checks if the solution is within the variable bounds
    check_bounds_evaluate_problem: bool = True

    # Maximum number of executing scipy.optimize.fsolve() when evaluating the problem
    n_starts_max_evaluate_problem: int = 100

    # %% BO

    ## General

    # Configure if plots are supposed to be created
    create_plots: bool = False

    # Configure if results are supposed to be saved, bool
    save_results: bool = False

    # Number of Bayesion optimization runs, int
    n_runs_bo: int = 0

    # Number of Bayesian optimization iterations
    n_iterations_bo: int = 8

    # Configure which acquisition problem formulation to use
    # Options: "ei", "lcb"
    formulation_acq: str = "ei"

    # Number of starting points the optimization of the acquisition problem is executed with
    n_starts_acq_optimization: int = 10

    # Number of GP samples used in the acquisition problem
    n_samples_gp: int = 50

    # LCB = mean - factor_lcb_std * std
    factor_lcb_std: float = 1.0

    # EI Standard (no deterministic equivalent)
    epsilon_ei_standard: float = 1.0e-10

    # %% Plotting

    # Number of evaluation points used for every component of u
    n_eval_points: list[int] = field(default_factory=list)

    # Level of confidence in percent for plotting the confidence intervals, optional
    confidence_level: int = 95

    # %% General

    seed: int | None = None

    # %% Dependent fields
    # They are dependent on other fields and thus are initialized in __post_init__(self)
    # Code from here on should not be changed

    # Indices of the elements of x that are neither inputs nor outputs of the GP
    indices_x_no_gp: list[int] = field(init=False)

    # Random number generator
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.indices_x_no_gp: list[int] = list(
            set(range(self.n_x))
            - set(self.indices_x_input_gp)
            - set(self.indices_x_output_gp),
        )

        self.rng: np.random.Generator = np.random.default_rng(self.seed)
