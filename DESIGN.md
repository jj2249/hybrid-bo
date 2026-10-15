# Design: target API

This is where the restructure is heading. The vocabulary is in [CONTEXT.md](CONTEXT.md)
and the decisions that are hard to reverse are in [docs/adr/](docs/adr/).
[ROADMAP.md](ROADMAP.md) lists the PRs. Each PR may refine names or signatures; when one
does, it updates this file in the same PR.

All spelling is American, in code and in prose.

## Why

The code is hard to change for three reasons:

1. **Duplication.** The four routines in `routines/` are near-copies of one loop.
2. **Ungrouped, index-based data.** Related values travel as separate arrays (`u`, `x`,
   `x_no_noise`, `f`, `incumbents`, ... in `_initial`, `_bo` and `_complete` versions),
   and the structure of the problem lives in index lists in `config.yaml`
   (`indices_x_output_gp: [1]` is the only thing that says `x[1]` is really `y`).
3. **Stateful objects.** You configure an object, call methods in the right order and
   read results off its state (`set_XY_train` → `create_training_problem` →
   `solve_training_problem`; `set_problem` → `solve()`), and `Config` is passed
   everywhere.

## Principles

1. **Data in, data out.** Functions return new values and never mutate their inputs.
2. **Frozen dataclasses for data and settings; behavior in plain functions.** No strategy
   classes and no kernel classes.
3. **Functions take what they use.** There is no `Config`.
4. **One way to do each thing.** One CasADi code path for symbolic and numeric use.
5. **Explicit randomness.** Anything random takes an `rng: np.random.Generator`.
6. **Errors raise** specific exceptions. Expected failures (one multistart start not
   converging) are logged without tracebacks.
7. **Rows are points.** A point is `(n,)`, a set is `(n_points, n)`, `f` is `(n_points,)`.
8. **Examples are short to write.** Defaults may be derived from other arguments when
   the derivation is obvious and recorded in the run.

## Decisions

- **Notation follows the paper** (arXiv 2603.11199, Problem 4): decision variables `u`,
  mechanistic states `x`, unknown states `y`, known equations `0 = g(x, y, u)`, unknown
  equations `y = h(x, u)`, scenario draws `z` (the paper's ξ).
- **Problem, method and experiment are separate** ([ADR 0001](docs/adr/0001-problem-method-experiment-split.md)).
  The problem is a system plus an objective; the method owns the surrogate; the oracle
  exists only in test cases.
- **Equations are callables traced by CasADi** ([ADR 0002](docs/adr/0002-equations-are-casadi-callables.md)).
- **Runs are pure data** ([ADR 0003](docs/adr/0003-runs-are-pure-data.md)).
- **SX and MX.** Library math uses only CasADi operations, so it works for both. The
  symbol type is set in one place (default `SX`). `DM` appears only where a compiled
  `cas.Function` is called, and is converted to numpy there.
- **Known issues are not rebuilt.** Where the new design makes a Known issue impossible
  and the regression baselines do not move, the new code simply does not reproduce it;
  the PR says which Known issue it removes.
- **Saving.** The library never writes files on its own; saving is an explicit call.
  `hb.save`/`hb.load` write and read a study as JSON
  ([ADR 0004](docs/adr/0004-studies-are-saved-as-json.md)). Examples call them; the
  library never does.
- **Iteration plots replay the acquisition problem**
  ([ADR 0005](docs/adr/0005-iteration-plots-replay-the-acquisition-problem.md)).
- **A slim top level** (#52). `hb.*` holds what a typical script uses: `Var`, `System`,
  `Problem`, `Experiment`, `oracle_experiment`, `initial_points`, the methods and their
  settings, `EI`, `LCB`, `StudySettings`, `Study`, `repeat`, `study`,
  `oracle_optimum`, `save`, `load`, `Run`, `History`, `incumbents`,
  `oracle_incumbents`, `evaluations`, `summary`, `hyperparameters`, the errors and
  `plots`. Everything else is reached through its module (`hb.gp.fit_gp`,
  `hb.solvers.multistart`). A new name joins the top level only if the target example
  or a typical analysis script would use it.

## Target example

```python
import functools

import casadi as cas
import numpy as np

import hybrid_bo as hb

# Problem: what we know about the physics, and what we want
u = hb.Var("u", lower=-2.0, upper=2.0)  # start bounds default to the finite bounds
x = hb.Var("x", start=(-2.5, 1.0))  # unbounded, so start bounds are required
y = hb.Var("y", start=(-2.0, 2.0))

system = hb.System(
    u=[u],
    x=[x],
    y=[y],
    g=lambda v: [v.x + cas.exp(v.x) - v.y],  # 0 = g(x, y, u)
    h_inputs=[u],  # y = h(u)
    measured=[y],
)
problem = hb.Problem(system, f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4))


# Test case: an oracle plays the real system
def oracle(v):
    """The unknown equation y = h(u), known only to the test case."""
    return [cas.sin(v.u)]


experiment = hb.oracle_experiment(system, oracle, noise_std=0.0)

# Method: how we search
settings = hb.HybridBOSettings(acquisition=hb.EI(), n_scenarios=25)  # default GP

methods = {"hybrid_bo": functools.partial(hb.hybrid_bo, settings=settings)}

# Study: 10 runs, run i of every method from the same initial points; the same seed
# gives the same study
study_settings = hb.StudySettings(n_runs=10, n_initial=2, n_iterations=5, seed=0)
study = hb.study(methods, problem, experiment, study_settings, optimum=-50.4608)
hb.save(study, "results/illustrative")
```

## Target API

All snippets are illustrative.

### Problem (PR 4, done; `problem.py`, `sampling.py`)

```python
Symbolic = cas.SX  # the one place the symbol type is chosen


@dataclass(frozen=True, init=False)
class Var:
    name: str  # a Python identifier, used as v.name
    lower: float
    upper: float
    start: tuple[float, float]  # finite, inside the bounds

    # start defaults to (lower, upper) when both are finite, else it is required
    def __init__(self, name, lower=-np.inf, upper=np.inf, start=None): ...


@dataclass(frozen=True, init=False)
class System:
    u: tuple[Var, ...]  # N decision variables
    x: tuple[Var, ...]  # M mechanistic states
    y: tuple[Var, ...]  # P unknown states (P > 1 raises NotImplementedError)
    g: cas.Function  # g(x, y, u) -> (M, 1), column-vector inputs
    h_inputs: tuple[Var, ...]  # the arguments of h, from u and x
    measured: tuple[Var, ...]  # the states the sensors read, from x and y; P of them

    def __init__(self, u, x, y, g: Callable, h_inputs, measured): ...

    # n_u, n_x, n_y properties


@dataclass(frozen=True, init=False)
class Problem:
    system: System
    f: cas.Function  # f(x, y, u) -> scalar

    def __init__(self, system: System, f: Callable): ...


def initial_points(variables, n, rng) -> np.ndarray: ...  # lhs over the start bounds
```

Construction calls each callable once with a read-only namespace `v` of `Symbolic`
scalars (an unknown name raises `AttributeError` listing the available ones; `g` may
return one expression or a list) and builds the `cas.Function`s in declaration order,
with inputs and outputs named `x`, `y`, `u` → `g`/`f`. It checks: names are identifiers,
don't start with `_` (they would shadow the namespace's internals) and are unique;
callables return CasADi expressions (`TypeError` otherwise); `lower <= upper`; start bounds finite, ordered and inside the bounds;
`u`, `x`, `y` non-empty; `len(g) == M`; `f` scalar; `h_inputs` from `u` and `x`,
`measured` from `x` and `y`, each without repeats; `len(measured) == P`. `Var` is a plain
frozen dataclass, so membership checks use its equality; no CasADi `==` is involved.

`lhs(lower, upper, n, rng)` and `uniform(...)` stay the array primitives the library
uses for starts; `initial_points` is the `Var` front end for examples.

### Evaluation (#25, done; `evaluation.py`, `history.py`, `errors.py`)

```python
@dataclass(frozen=True)
class OracleMeasurement:  # what the oracle simulation returns
    measured: np.ndarray  # (P,)
    x: np.ndarray  # (M,) oracle values
    y: np.ndarray  # (P,)


Experiment = Callable[
    [np.ndarray, np.random.Generator], np.ndarray | OracleMeasurement
]  # u (N,) -> measured (P,)


def oracle_experiment(
    system,
    oracle: Callable,  # like g: lambda v: [cas.sin(v.u)], y = oracle(x, u)
    noise_std: float = 0.0,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> Experiment: ...


def evaluate(
    problem, experiment, u: np.ndarray, rng, n_starts=100, tol=1e-10
) -> History: ...  # u (n, N); the points are the History's initial points
```

- An **experiment** returns the measured variables for one trial point. A real
  application supplies only this, as a plain array.
- The **oracle simulation** solves `g` with `y = oracle(x, u)` for every state
  (root-finding with CasADi Jacobians), reads the measured variables and adds
  `rng.normal(0, noise_std)`. Its full states are the **oracle values**, returned in an
  `OracleMeasurement`.
- **Oracle vs real system.** Being simulated doesn't make a system an oracle. An in
  silico simulation of a plant (a flowsheet simulator, say) is a real system to the
  library: a custom experiment that returns only the measured variables, with no
  oracle values and no disagreement check. An oracle is the test case stating the
  unknown equations in closed form, next to `g`, so the oracle values are known and a
  method can be scored against them. The method never sees the oracle either way. A
  custom experiment that knows its oracle values can return an `OracleMeasurement`.
- The library then always solves the **root-finding problem**: `g` for the unmeasured
  states given `u` and the measured variables, from Latin hypercube starts over the
  unmeasured variables' start bounds, then computes `f` (and `f` at the oracle values).
- A root is accepted when its residual norm is below `tol` and it lies within the
  bounds; if no start gives one, `RootFindingError`.
- **Disagreement with the oracle.** Comparing the reconstruction directly with the
  oracle values would warn at every noisy point. Instead `evaluate` solves the same
  root-finding problem (same noisy measured variables) once more, started at the
  oracle values. That extra solve draws nothing, so the draw order is unchanged. If it
  gives a different root (`rtol = atol = 1e-6`) or none, the measured variables don't
  determine the states uniquely and the method may learn from states the system didn't
  settle at. `evaluate` then warns with `ReconstructionWarning` and keeps the
  reconstruction, because a real system would give nothing else. The fix is in the
  problem: measure other variables, or narrow the unmeasured start bounds to the right
  branch.
- `fsolve` always uses the CasADi Jacobian (`use_jacobian` is removed). A failed start is an
  expected failure (Principle 6), so fsolve's convergence warnings are caught and
  nothing is printed; only `RootFindingError` reports that every start failed.

### History and Run (#26: `Run` with `history`, `record` and `surrogates`; `run.py`)

```python
@dataclass(frozen=True)
class History:  # every evaluated point of one run, in order
    u: np.ndarray  # (n, N)
    x: np.ndarray  # (n, M)  measured and reconstructed
    y: np.ndarray  # (n, P)
    f: np.ndarray  # (n,)
    x_oracle: np.ndarray | None  # benchmark only; None without an oracle
    y_oracle: np.ndarray | None
    f_oracle: np.ndarray | None
    n_initial: int


def incumbents(history) -> np.ndarray: ...  # np.minimum.accumulate(history.f)
def oracle_incumbents(
    history,
) -> np.ndarray: ...  # f_oracle at each incumbent's point (#32)


@dataclass(frozen=True)
class AcquisitionSolution:  # acquisition.py (#53)
    value: float  # the acquisition problem's objective at the trial point
    x: np.ndarray | None = None  # (S, M) scenario states; hybrid BO only
    y: np.ndarray | None = None  # (S, P)


@dataclass(frozen=True)
class Run:
    history: History
    record: dict[str, str]  # how the run was produced, as text
    method: str  # its label: the key in the study's methods (#53)
    index: int = 0  # its position among that method's runs (#53)
    surrogates: list[FittedGP]  # one per iteration; empty for baselines (default)
    acquisitions: list[AcquisitionSolution]  # one per iteration (#53)
    scenarios: Scenarios | None = None  # hybrid BO only (#29)
```

The **record** is built by the library (`run_record`, before the method draws
anything), not the example: the variables with their bounds, `h_inputs`, `measured`,
`g` and `f` as expression text, the experiment, the method and its settings, the GP
spec and kernel expression (from #28), the seed, spawn key and generator state, and the
versions of `hybrid_bo` (with `git describe`), Python, CasADi, NumPy and SciPy. The
experiment is recorded by its docstring, else its name; the oracle simulation's
docstring is the oracle as text with its noise and solver settings, so a custom
experiment adds a description by writing a docstring.

A column is found by name with `system.column("x1")`.

### Solvers (#27, done; `solvers.py`)

```python
@dataclass(frozen=True)
class NLP:
    f: cas.Function  # w -> scalar
    lower: np.ndarray  # (n,), may be infinite
    upper: np.ndarray
    g: cas.Function | None = None  # g(w) = 0


@dataclass(frozen=True)
class Ipopt:
    plugin_options: dict = field(default_factory=lambda: {"print_time": 0})
    solver_options: dict = field(default_factory=lambda: {"print_level": 0})


@dataclass(frozen=True)
class Solution:
    w: np.ndarray  # (n,)
    f: float


def multistart(nlp: NLP, starts: np.ndarray, solver: Ipopt) -> Solution:
    """Best of the starts; raises NoSolutionError if every start fails."""
```

There is no `Solver` protocol until a second solver exists. The caller draws the starts
with `lhs(...)`. `multistart` builds the problem with `cas.Opti`, bounds as
constraints, as the legacy optimizer did, so IPOPT takes the same steps. A failed start
is logged at debug level without a traceback.

### GP (#27, done; `gp.py`, `kernels.py`)

```python
@dataclass(frozen=True)
class Hyper:
    name: str
    init: float
    bounds: tuple[float, float]
    size: int = 1  # e.g. one lengthscale per input (ARD)
    log: bool = True  # optimize in log space


@dataclass(frozen=True)
class GP:
    hyperparameters: tuple[Hyper, ...]  # declaration order is the optimization order
    kernel: Callable  # (h, X1, X2) -> (n1, n2)
    mean: Callable = zero_mean  # (h, X) -> (n, 1)
    jitter: float = 1e-10


def default_gp(n_inputs: int) -> GP: ...  # noise-free outputscale * Matern-5/2, ARD


@dataclass(frozen=True)
class Scaler:  # scaled = slope * value + intercept, per column
    slope: np.ndarray
    intercept: np.ndarray


@dataclass(frozen=True)
class FittedGP:  # data only (ADR 0003)
    hyperparameters: dict[str, np.ndarray]  # each (size,)
    X: np.ndarray  # training inputs, scaled
    Y: np.ndarray  # training outputs, scaled
    x_scaler: Scaler
    y_scaler: Scaler


def fit_gp(gp, X, Y, rng, n_starts=10, solver=Ipopt()) -> FittedGP: ...
def predict(gp, fitted, X) -> tuple: ...  # (mean, std) in original units
def log_likelihood(gp, fitted) -> float: ...  # on the scaled training data
```

Kernels are plain functions (`matern12/32/52`, `squared_exponential`, `linear`,
`constant`) combined with arithmetic. The kernel and mean read the hyperparameters as
`h.<name>`, a `(size, 1)` CasADi column:

```python
gp = hb.gp.GP(
    hyperparameters=(
        hb.gp.Hyper("noise_std", 1e-2, bounds=(1e-6, 1.0)),
        hb.gp.Hyper("outputscale", 1.0, bounds=(0.1, 10.0)),
        hb.gp.Hyper("lengthscale", 1.0, bounds=(1e-3, 1e2), size=2),
    ),
    kernel=lambda h, X1, X2: h.outputscale * hb.kernels.matern52(X1, X2, h.lengthscale),
)
```

- **Noise.** The hyperparameter named `noise_std`, if declared, is the observation
  noise: `K = k(X, X) + (noise_std**2 + jitter) I`. Without it the GP interpolates.
  `default_gp` is noise-free, because the illustrative example (legacy
  `noise_std.fix(0.0)`) is; add `noise_std` for noisy data.
- **Output scale** multiplies the kernel unsquared, as legacy `OutputScaleKernel` did,
  so `default_gp` fits the same hyperparameters as the legacy GP.
- `fit_gp` fits the scalers (`minmax` onto [0, 1] for X, `standard` for Y), calls
  `kernel` and `mean` once with CasADi symbols for the hyperparameters, builds the
  negative log marginal likelihood, draws starts with `lhs` over the (log-transformed)
  hyperparameter bounds and calls `multistart`. If every start fails it keeps the
  initial values and logs a warning. With the same data and generator its
  hyperparameters matched the legacy GP bit for bit, and its predictions to rounding
  (checked until #30; the regression baselines still pin the fit).
- `predict` traces the posterior once as a `cas.Function` of X; a numpy X gives arrays
  `(n,)`, a symbolic X (SX or MX) gives `(n, 1)` columns. The std is the square root of
  the posterior variance, so callers need no covariance.
- This removes `Parameterized`, `JoinedParameters`, `k_variable`/`m_variable` and the
  name prefixing, and with them the `LinearKernel` ARD Known issue. `minmax` uses the
  correct intercept (`lower - min * slope`), evaluated so that [0, 1] reproduces the
  legacy values bit for bit.

### Acquisition (#28: `EI`, `LCB`, `standard_acquisition`; #29: `Scenarios`, `hybrid_acquisition`; `acquisition.py`)

```python
@dataclass(frozen=True)
class EI: ...


@dataclass(frozen=True)
class LCB:
    std_factor: float = 1.0  # finite and >= 0, else ValueError


@dataclass(frozen=True)
class Scenarios:
    z: np.ndarray  # (S, P)
    w: np.ndarray  # (S,), >= 0 and sums to 1, else ValueError


def saa_scenarios(n: int, n_y: int, rng) -> Scenarios: ...  # z ~ N(0, I), w = 1/S


def hybrid_acquisition(
    problem,
    gp,
    fitted,
    scenarios,
    incumbent,
    acquisition,
    unknown_states_as_variables=True,
) -> tuple[NLP, tuple[np.ndarray, np.ndarray]]: ...
def standard_acquisition(
    problem, gp, fitted, incumbent, acquisition
) -> tuple[NLP, tuple[np.ndarray, np.ndarray]]: ...
```

Each returns the NLP and its start bounds `(lower, upper)`, for `lhs`.

- **Validation.** Every method checks its acquisition function the same way (one
  helper, called by the settings and by the builders): anything but an `EI` or `LCB`
  instance raises `TypeError`. This removes the `formulation_acq` Known issue.
- **Standard BO** computes EI and LCB analytically from the GP mean and std (EI adds
  1e-10 to the std in `gamma`, as legacy `ei_gp` did) and minimizes -EI or LCB over `u`
  within its bounds.
- **Hybrid BO** computes them as weighted sample averages over the scenarios (SAA-EI,
  SAA-LCB; paper, Problems 8 and 9). Each scenario has its own `x_s` and `y_s`, with `g`
  as constraints. SAA-LCB uses the weighted variance; `w = 1/S` reproduces today's
  `sample_variance` (`ddof=0`).
- `unknown_states_as_variables`: if True, each scenario's unknown states are variables
  of the acquisition problem tied to the GP by an equality constraint (paper, Problem 8),
  which helps IPOPT; if False, the GP prediction is substituted wherever they appear.
- The hybrid acquisition problem's variables are `u`, then the `x_s`, then the `y_s` (if
  variables), each state with one entry per scenario; the trial point is the first N
  entries. Every variable is boxed by its own bounds and started within its own start
  bounds, and the GP's inputs are the columns of `h_inputs`. This removes the two
  `setup_acq_problem` Known issues (`u` start bounds, number of GP inputs). With one
  mechanistic state, as in the illustrative example, the order matches legacy, so the
  Latin hypercube starts and the baseline are the same.
- Hybrid BO only needs "mean and std of `y` at given inputs" from the surrogate, through
  `predict`, so other surrogates or hyperparameter samples can be paired with scenarios
  later.

### Methods and the optimization loop (#28: the loop and standard BO; #29: hybrid BO; `bo.py`)

```python
@dataclass(frozen=True)
class HybridBOSettings:
    gp: GP | None = None  # None: default_gp(len(h_inputs))
    acquisition: EI | LCB = EI()
    n_scenarios: int = 25
    n_starts: int = 10  # acquisition multistart
    n_starts_gp: int = 10
    solver: Ipopt = Ipopt()
    unknown_states_as_variables: bool = True


@dataclass(frozen=True)
class StandardBOSettings:
    gp: GP | None = None  # None: default_gp(N)
    acquisition: EI | LCB = EI()
    n_starts: int = 10
    n_starts_gp: int = 10
    solver: Ipopt = Ipopt()


TrialPoint = Callable[
    [Problem, History, np.random.Generator],
    tuple[np.ndarray, FittedGP, AcquisitionSolution],
]


def optimization_loop(
    problem,
    experiment,
    initial_points,
    n_iterations,
    trial_point: TrialPoint,
    rng,
    record,  # from run_record, before the method draws
) -> Run: ...


def hybrid_bo(
    problem, experiment, initial_points, n_iterations, settings, rng
) -> Run: ...
def standard_bo(
    problem, experiment, initial_points, n_iterations, settings, rng
) -> Run: ...
def latin_hypercube_sampling(
    problem, experiment, initial_points, n_points, rng
) -> Run: ...  # done in #26, baselines.py
def uniform_sampling(problem, experiment, initial_points, n_points, rng) -> Run: ...
```

The baselines are named `*_sampling` because `hb.sampling.uniform` and
`hb.sampling.lhs` are the sampling primitives.

- `hybrid_bo` draws the scenarios from `rng`, then runs `optimization_loop` with a
  trial-point function that fits the GP on `(u, x)[h_inputs] -> y`, builds
  `hybrid_acquisition`, and solves it with `multistart`. The user never draws `z`.
  With `unknown_states_as_variables=False` (the legacy illustrative setting) it
  reproduces the legacy illustrative baseline.
- `standard_bo` fits on `u -> f` (measured), with the incumbent the best measured `f`,
  and solves `standard_acquisition` by `multistart` from `lhs` starts over the `u` start
  bounds. With the defaults it reproduces the legacy illustrative baseline.
- `optimization_loop` takes the record and returns the `Run`, so a `HybridBOError` in
  the loop (an evaluation, or the trial-point function) carries the partial run as
  `error.run` for every method: the points evaluated so far (also `error.history`) and
  one surrogate per evaluated trial point (a failed evaluation drops its surrogate). A method that adds fields to its `Run` (hybrid BO's scenarios)
  adds them to the returned run and to `error.run`.
- The record holds the settings without the GP, plus `gp` (hyperparameters, jitter),
  `kernel` and `mean` as expression text of the GP the run uses (the resolved
  default), since a GP's repr only names its kernel function.
- The baselines are not sequential: they draw their whole design and `evaluate` it after
  the initial points.
- `optimization_loop` is public, for new methods.

### Errors

`HybridBOError` and subclasses (`RootFindingError` from #25, `NoSolutionError`, ...). An
error raised in `evaluate` carries the points evaluated before it (`error.history`); an
error raised during a run carries the partial `Run` (`error.run`, from #26), so the
caller can save it. The loop does not catch errors. Warnings are `UserWarning`
subclasses (`ReconstructionWarning`) so they can be filtered.

### Plots

`hb.plots` is described under Analysis below (#55). Figures use the repo's style file,
`hb.plots.STYLE` (`src/hybrid_bo/hybrid_bo.mplstyle`); put rcParams there, not in
plotting code.

### Analysis (#53, #54, #55)

The ArviZ-like layer: a `Study` is the data object; Polars views and static plots read
it. Polars and matplotlib are optional (imported on first use; the
`hybrid-bo[analysis]` extra once packaged).

- **Data** (#53, done; `studies.py`). `Run` gains `method` (its label in the study),
  `index` and `acquisitions`, one `AcquisitionSolution` per iteration (the
  acquisition value at the trial point and, for hybrid BO, the scenario states
  there), and keeps its full `record`.
  `StudySettings(n_runs, n_initial, n_iterations, seed=None)`; every draw of a study
  comes from the seed, so the same settings give the same study.
  `Study(runs: tuple[Run, ...], settings, name, record)` is a flat collection; its
  settings hold the seed actually used (a fresh one for seed None) and its record
  holds the problem, experiment, methods, oracle optimum (`oracle_optimum`, when
  stated) and versions once.
- **Running** (#53, done). `hb.repeat(method, problem, experiment, settings)` runs
  one method `n_runs` times, run i from `SeedSequence(seed).spawn(n_runs)[i]`, logging
  failed runs and keeping them as far as they got. A `Method` is called as
  `method(problem, experiment, initial_points, n, rng=rng)`.
  `hb.study(methods, problem, experiment, settings, name=None, optimum=None)`
  resolves the seed once and gives run i of every method the same
  generator, so the same initial points; each method's later draws come from its own
  copy. Studies take a seed, not an `rng`, so that the settings alone reproduce a
  study and the caller's generator is never advanced as a side effect. The oracle
  optimum is stated (`optimum`), never computed (#62): a multistart search can miss
  it, and a wrong optimum silently corrupts every regret. `hb.oracle_optimum(problem,
  oracle, rng, n_starts)` helps find it once. The study records it and warns with
  `OracleOptimumWarning` if an evaluated point's oracle objective is below it.
- **Saving** (#53, done; `saving.py`). `hb.save(study, path)` writes `study.json`
  (settings, name, record, the run files in order) and `runs/<method>/run-000.json`;
  `hb.load(path)` reads them back exactly. `save` refuses an existing folder.
- **Views** (#54, done; `views.py`). `hb.evaluations(study, optimum=None)`: one row
  per evaluation (method, run, evaluation from 1, iteration from 1 or null for initial
  points, `u_<name>`/`x_<name>`/`y_<name>`, f, `x_oracle_<name>`/`y_oracle_<name>`/
  `f_oracle`, incumbent, oracle incumbent, regret, acquisition). Variables are named
  from the study's record. Regret is the oracle incumbent minus `optimum`, else the
  recorded oracle optimum, else null. `hb.summary(study)`: one row per method (runs,
  failed runs against `n_runs`, median and IQR of the final incumbent, oracle
  incumbent and regret). `hb.hyperparameters(study)`: one row per run, iteration and
  hyperparameter entry. Polars is imported on first use; without it a view raises an
  `ImportError` naming the extra.
- **Plots** (#55, done; `plots.py`), static, over **evaluations** (initial points are
  not free): `plots.convergence(study)` (median oracle incumbent, else incumbent, with
  an IQR band and a dotted line after the initial points), `plots.regret(study,
  optimum=None)` (the same on a log scale, as in the paper) and
  `plots.iteration(run, problem, i, *, experiment=None, gp=True, scenarios=True,
  acquisition=True, settings=None)`, one iteration per call, no widgets and no `rng`.
  The GP panel comes from the stored surrogate; the acquisition panels replay the
  method's acquisition problem with u fixed at each grid point, warm-started outwards
  from the stored solution (ADR 0005), so the curve passes through the stored
  acquisition value. The settings are read from the run's record; `settings` is
  needed only for a GP other than `default_gp`, since the record holds the GP as text,
  not its kernel function. The fields the replay uses (acquisition function, and for
  hybrid BO the solver and `unknown_states_as_variables`) must match the record.
- **Entry point** (#51, done; revised in #65). An example script ends with an
  explicit call that passes everything it defines: `cli.main(name, problem,
  experiment, methods, study_settings, optimum=..., true_curves=...)`.
  `examples/cli.py` parses the flags (run by default, or `plot [STUDY]`), overrides
  the settings and calls `hb.study` and `hb.save`, or `hb.load` and `hb.plots`.
  `pixi run example <name> [flags]` is `python -m <name> [flags]` run from
  `examples/` (a task with no declared args, so flags pass through), and `pixi run
  plot <name> [study]` a templated task over `python -m <name> plot`. #51 had
  `examples/main.py` discover the example and read its names with `getattr`; #65
  dropped that because the example then didn't say what its names were for, and a
  typo silently changed the study. The cost is that no command lists every example.
  A relative `--results-root` is relative to where pixi was started (`INIT_CWD`).
  Results go to `results/<name>/<study id>/{study.json, runs/, summary.csv,
  figures/}` at the repo root. Running prints `hb.summary` and saves it as
  `summary.csv`; `plot` takes the study saved last by default, writes under
  `figures/`, and saves `summary.csv` only if the study has none (older studies). The `hybrid-bo-examples` skill has the details. The
  CLI stays in `examples/` for people who clone the repo; it is not part of the
  package.

## Order of random draws

Every draw in a run comes from that run's generator (the example spawns one per run),
in this order:

1. initial points: `lhs` over the `u` start bounds (drawn by the caller);
2. method setup: hybrid BO draws the scenarios (`rng.standard_normal((S, P))`); the
   baselines draw their whole design; standard BO draws nothing;
3. each evaluation, initial points first: the experiment (oracle simulation: full-solve
   starts, then noise), then the root-finding starts;
4. each iteration: GP-fit starts, then acquisition starts, then the evaluation of the
   trial point as in 3.

Root-finding starts are drawn even without noise, because the reconstruction always
runs. Each consumer draws only what it uses, when it uses it.

**A generator has two independent sequences.** scipy's
`qmc.LatinHypercube(d, rng=rng)` does not consume `rng`'s stream: it spawns the next
child of `rng`'s `SeedSequence` (as `rng.spawn` does). So every `lhs` call (initial
points, fsolve and root-finding starts, multistart starts, LHS designs) is ordered only
against other `lhs` calls and `spawn`s, and the stream draws (`normal`, `uniform`,
`standard_normal`: noise, scenarios, the uniform design) only against each other. The
order above holds within each sequence; the order *between* the two sequences (e.g.
noise before or after the oracle's fsolve starts) has no effect on any value. New code must keep using
`LatinHypercube(..., rng=rng)` for `lhs`, and must not add a `spawn` or an `lhs` call
in the middle of a run, or the baselines move. `tests/test_sampling.py` pins this
scipy behavior.

## Removed

| What | Why | Where to find it |
|---|---|---|
| `array_operations/` numpy branches and their `@overload` stubs | one CasADi path (SX or MX); numeric use goes through `cas.Function` | tag `pre-refactor` |
| `SciPyLocalOptimizer` | never used; only offered as a commented-out alternative | tag `pre-refactor` |
| `use_jacobian` (fsolve without the CasADi Jacobian) | every equation is CasADi-traceable ([ADR 0002](docs/adr/0002-equations-are-casadi-callables.md)) | tag `pre-refactor` |
| `Problem.solve` (true optimum with `h` known) | called nowhere; would be a function on `multistart` | tag `pre-refactor` |
| `Config` and `config.yaml` | functions take what they use; problems are Python | tag `pre-refactor` |
| `hybrid_bo.legacy`: the `do_*` routines, `ResultsBO` pickles, the old `Problem`, GP and optimizers | replaced by the new API (#30); studies are saved as JSON (#53) | tag `pre-refactor` |
| Kernel and mean classes, `Parameterized`, `JoinedParameters` | hyperparameters are named specs; kernels are functions | tag `pre-refactor` |
| Per-iteration plot callbacks | plots are functions of a finished run ([ADR 0003](docs/adr/0003-runs-are-pure-data.md)) | tag `pre-refactor` |
| Strategy classes (planned, never built) | methods are settings plus functions | earlier DESIGN.md in git history |

## Future directions

Not built now; the design leaves room for them.

- **Bayesian hyperparameters.** MAP: an optional `prior` on `Hyper`, added to the
  negative log likelihood. Sampling: a `log_posterior(gp, X, Y)` seam returning a
  `cas.Function` with its gradient over the log-space hyperparameters, usable by slice
  sampling or NUTS (pymc through a custom op, or a standalone sampler). Each scenario
  then pairs a hyperparameter sample with its `z`. Suggested order: MAP, slice sampling,
  NUTS if the dimension grows.
- **Alternatives to SAA.** Sigma points or cubature produce a different `Scenarios`
  (deterministic `z`, specific `w`); the acquisition builders already take weights.
- **More than one unknown state (P > 1).** Shapes already allow it; the open modeling
  question is whether each unknown equation has its own arguments.

## Restructure PRs

Every PR keeps `tests/data/regression/` unchanged unless it is marked *logic*. Each PR
updates `hybrid-bo-src`, `hybrid-bo-examples`, the README and this file where its change
affects them.

The old code was moved to `hybrid_bo/legacy/` and the new modules built at their
final paths. New code never depended on old code (old code could call new helpers such
as `lhs`), so no shims were needed and deleting `legacy/` (#30) touched nothing new.

PRs 0-4 were planned layer by layer. After PR 4, the work is GitHub issues sliced
by method: each method ticket ports its own regression test to the new API and
reproduces its baseline from the draw-order PR exactly, so the new code is proven
end to end one method at a time. Those tickets add API without changing existing
behavior, so they are "new code" PRs, not *logic*.

0. **Design** (docs): CONTEXT.md, ADRs, this file, ROADMAP.md, CLAUDE.md conventions.
1. **R1 trimmed** (#17). *Done.* flash_unit excluded from ty (it targets the
   `pre-refactor` tag); `lhs(lower, upper, n, rng)`/`uniform(...)` in `sampling.py`
   (they stay array-based; PR 4 adds `initial_points` for `Var`s); `Problem.solve` and `use_jacobian` deleted;
   `examples/main.py` dispatches with `match`. Two hand-written LHS copies remain until
   their tickets: the fixed-x root-finding guesses (#25) and
   `MultiStartOptimizer.create_lhs_samples` (#27). The new evaluation uses `lhs`; the
  legacy copy stays until `legacy/` is deleted.
2. **Draw order** (*logic*). *Done.* The old routines draw in the order above:
   noise per evaluation (`noise_std` instead of a pre-drawn array), root-finding
   always runs, `measurement_noise.pkl` dropped (it is `x - x_no_noise` on the
   measured states). Baselines regenerated.
3. **Legacy**. *Done.* The old code moved to `hybrid_bo/legacy/` (deleted in #30),
   imported as `hybrid_bo.legacy`; the top-level `hybrid_bo` exported only the new
   API. Legacy could import new code, never the reverse. Result pickles saved before this PR
   (`hybrid_bo.results.ResultsBO`) need the old path to unpickle.
4. **Problem** (#23): `Var`, `System`, `Problem`, `initial_points`. *Done.*

The tickets, in a workable order:

- **#24 Workflow** (docs): PR template and CLAUDE.md fit the engineering skills;
  this list. Blocked by nothing.
- **#25 Evaluation**: experiments, `oracle_experiment`, root-finding, `evaluate`,
  `History`, the `HybridBOError` family. *Done*; `tests/test_evaluation.py` checks it
  against the legacy evaluation bit for bit.
- **#26 Baselines**: Latin hypercube and uniform as methods returning a `Run` and its
  record; their regression tests on the new API. *Done*; the regression test maps the
  `Run` to the `ResultsBO` keys the baselines were recorded with.
- **#27 Solvers and GP**: `NLP`, `Ipopt`, `multistart`; `Hyper`, kernel functions, `GP`,
  `fit_gp`, `predict`, scalers, `default_gp`; port the kernel, GP and scaler tests.
  *Done*; `tests/test_gp.py` checked the fit against the legacy GP until #30.
- **#28 Standard BO**: `optimization_loop`, `EI`/`LCB`, `standard_bo`; its regression
  test on the new API. *Done*; it reproduces the legacy baseline.
- **#29 Hybrid BO**: `Scenarios`, the hybrid acquisition problem, `hybrid_bo`; its
  regression test on the new API. *Done*; it reproduces the legacy baseline, and every
  regression test now runs on the new API.
- **#30 Switch**: the illustrative example as one script on the new API; delete
  `legacy/`, `Config` and YAML; README and both skills describe the new API.
  *Done*; `examples/illustrative_example.py` defines the problem, experiment and
  `methods`, runs and saves every method when run directly, and is what the
  regression test runs. Tests that compared with the legacy code check closed forms
  instead. The plot callbacks go with it; plots return in #31.
- **#31 Plots and CLI**: `hb.plots` (see Plots); `examples/main.py` is the CLI (`run`
  an example's methods, `plot` a results folder), and example modules only define
  things (a `settings` dict of the BO methods joins `methods`). *Done*.
- **#32 Benchmark metric** (*logic*): the oracle objective at the believed best
  point. *Done*: `oracle_incumbents(history)`. Without noise it is the running
  minimum of `f_oracle`, as in the paper's (noise-free) regret figures, so the
  baselines don't change.
- **#52 Namespace**: `hb.*` holds only the user-facing names (see Decisions); a
  Getting started docs page. *Done*.
- **#53 Study**: see Analysis. *Done*; `examples/main.py` runs and saves a study, and
  the examples define their study settings.
- **#54 Polars views**, then **#55 Plots**, then **#51 Entry point**: see Analysis.
  *Done*. With #51 the separate dict of method settings in an example is gone (the
  plots read them from the record).
