# Design

How the library is built and why. The vocabulary is in [GLOSSARY.md](GLOSSARY.md) and
the decisions that are hard to reverse are in [docs/adr/](docs/adr/). A PR that
changes the design updates this file in the same PR. The design came out of a
rewrite of the original code (tag `pre-refactor`, called *legacy* below), which the
new code reproduced bit for bit where the regression baselines cover it. Issue
numbers refer to the PRs that introduced or changed a part.

All spelling is American, in code and in prose.

## Why

The original code was hard to change for three reasons:

1. **Duplication.** The four routines in `routines/` were near-copies of one loop.
2. **Ungrouped, index-based data.** Related values traveled as separate arrays (`u`, `x`,
   `x_no_noise`, `f`, `incumbents`, ... in `_initial`, `_bo` and `_complete` versions),
   and the structure of the problem lived in index lists in `config.yaml`
   (`indices_x_output_gp: [1]` was the only thing that said `x[1]` is really `y`).
3. **Stateful objects.** You configured an object, called methods in the right order
   and read results off its state (`set_XY_train` → `create_training_problem` →
   `solve_training_problem`; `set_problem` → `solve()`), and `Config` was passed
   everywhere.

## Principles

1. **Data in, data out.** Functions return new values and never mutate their inputs.
2. **Frozen dataclasses for data and settings; behavior in plain functions.** No strategy
   classes. Kernels and means are dataclasses from a closed set, with their behavior in
   functions that match on the type (ADR 0008); their one operator,
   `outputscale * kernel`, only builds data.
3. **Functions take what they use.** There is no `Config`.
4. **One way to do each thing.** One CasADi code path for symbolic and numeric use.
5. **Explicit randomness.** Anything random takes an `rng: np.random.Generator`.
6. **Errors raise** specific exceptions. Expected failures (one multistart start not
   converging) are logged without tracebacks. Progress (each run and iteration, with
   times) is logged too; the library never configures logging.
7. **Rows are points.** A point is `(n,)`, a set is `(n_points, n)`, `f` is `(n_points,)`.
8. **Experiments are short to write.** Defaults may be derived from other arguments when
   the derivation is obvious and recorded in the run. A default may set how hard or how
   precisely we search, never what we search for ([ADR 0009](docs/adr/0009-defaults-never-choose-the-method.md)):
   the acquisition function, the approximation, the number of scenarios, LCB's
   `std_factor`, SoftSAA's temperature and the noise are required, and the settings
   are keyword-only.

## Decisions

- **Notation follows the paper** (arXiv 2603.11199, Problem 4): decision variables `u`,
  mechanistic states `x`, unknown states `y`, known equations `0 = g(x, y, u)`, unknown
  equations `y = h(x, u)`, scenario draws `z` (the paper's ξ).
- **Problem, method and measurement are separate** ([ADR 0001](docs/adr/0001-problem-method-experiment-split.md),
  names from [ADR 0006](docs/adr/0006-an-experiment-is-what-a-user-writes.md)).
  The problem is a system plus an objective; the method owns the surrogate; the oracle
  exists only in benchmark experiments.
- **Equations are callables traced by CasADi** ([ADR 0002](docs/adr/0002-equations-are-casadi-callables.md)).
- **Runs are pure data** ([ADR 0003](docs/adr/0003-runs-are-pure-data.md)).
- **SX only.** Library math is built from `SX` symbols, set by the `Symbolic` alias.
  MX does not work, for three reasons:
  - `cas.chol` has no MX overload, so with `Symbolic` set to MX, GP fitting,
    prediction and the log marginal likelihood fail.
  - A run's artifact prints the known equations and the objective as text. Under MX
    that text becomes an opaque `g(x, y, u){0}`.
  - The AMPL route to a global solver (#87) accepts only SX expressions.

  MX for the GP (or precomputing its terms) is #89, and MX for large systems is #90.
  If a second symbol type is ever needed, it would be chosen per problem rather than
  by a global toggle. `DM` appears only where a compiled `cas.Function` is called, and
  is converted to numpy there.
- **Legacy bugs were not rebuilt.** Where the new design made a known legacy bug
  impossible and the regression baselines did not move, the new code simply does not
  reproduce it.
- **Saving.** The library never writes files on its own; saving is an explicit call.
  `hb.save_run`/`hb.load_run` write and read a run as JSON, and `hb.save_runs`/
  `hb.load_runs` a list of runs per label.
- **A script runs the methods** ([ADR 0013](docs/adr/0013-scripts-run-the-methods.md)).
  The library is the optimization. Repeating a method over seeds is a loop in the script.
- **Iteration plots replay the acquisition problem**
  ([ADR 0005](docs/adr/0005-iteration-plots-replay-the-acquisition-problem.md)).
- **A slim top level** (#52). `hb.*` holds what a typical script uses: `Var`, `System`,
  `Problem`, `Measure`, `oracle_measure`, `initial_points`, the methods,
  `reconstruct_states`, `fit_surrogate`, `optimize_hybrid_model`,
  `EI`, `LCB`, `oracle_optimum`, `save_run`, `load_run`, `save_runs`, `load_runs`, `Run`, `Dataset`,
  `extend`, `incumbents`, `oracle_incumbents`, `check_oracle_optimum`, the errors and `plots`. Everything else is reached through its module (`hb.gp.fit_gp`,
  `hb.solvers.solve_with_multistart`). A new name joins the top level only if the
  example or a typical analysis script would use it.

## Example

```python
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


# Measurement: an oracle plays the real system
def oracle(v):
    """The unknown equation y = h(u), hidden from the methods."""
    return [cas.sin(v.u)]


measure = hb.oracle_measure(system, oracle, noise_std=0.0)

# Method: how we search. The GP, the acquisition function and its approximation have no
# defaults (ADR 0009)
gp = hb.gp.GP(mean=hb.gp.Zero(), kernel=hb.gp.Matern52(), noise_std=0.0)
run = hb.hybrid_bo(
    problem,
    measure,
    n_initial=2,
    n_iterations=5,
    rng=np.random.default_rng(0),
    gp=gp,
    acquisition=hb.EI(),
    approximation=hb.SAA(n_scenarios=25),
)
hb.save_run(run, "hybrid_bo.json")
```

## API

All snippets are illustrative.

### Problem (`problem.py`, `sampling.py`)

```python
Symbolic = cas.SX  # the symbol type of all library math; SX only (see Decisions)


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
    x: tuple[Var, ...]  # M >= 0 mechanistic states; optional with g (#115)
    y: tuple[Var, ...]  # P >= 1 unknown states, one GP each in hybrid BO (#114)
    g: cas.Function  # g(x, y, u) -> (M, 1), column-vector inputs; (0, 1) if M = 0
    h_inputs: tuple[Var, ...]  # the arguments of h, from u and x
    measured: tuple[Var, ...]  # the states the sensors read, from x and y; P of them

    # keyword-only; x and g go together, both omitted for no known equations
    def __init__(
        self, *, u, x=(), y, g: Callable | None = None, h_inputs, measured
    ): ...

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
`u` and `y` non-empty; `x` and `g` both given or both omitted (#115: without them
M = 0, and `g` is traced from no equations, so `System.g` is always a `cas.Function`
and the code downstream has no special case; with M = 0 the root-finding problem has
nothing to solve and `solve_with_multistart` adds no empty constraint); `len(g) == M`;
`f` scalar; `h_inputs` from `u` and `x`,
`measured` from `x` and `y`, each without repeats; `len(measured) == P`. `Var` is a plain
frozen dataclass, so membership checks use its equality; no CasADi `==` is involved.

`lhs(lower, upper, n, rng)` and `uniform(...)` stay the array primitives the library
uses for starts; `initial_points` is the `Var` front end for experiments.

### Evaluation (`evaluation.py`, `oracle.py`, `dataset.py`, `errors.py`)

```python
@dataclass(frozen=True)
class OracleMeasurement:  # what the oracle simulation returns
    measured: np.ndarray  # (P,)
    x: np.ndarray  # (M,) oracle values
    y: np.ndarray  # (P,)


Measure = Callable[
    [np.ndarray, np.random.Generator], np.ndarray | OracleMeasurement
]  # u (N,) -> measured (P,)


def oracle_measure(
    system,
    oracle: Callable,  # like g: lambda v: [cas.sin(v.u)], y = oracle(x, y, u)
    noise_std: float = 0.0,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> OracleSimulation: ...  # a frozen, callable dataclass, described by its repr


def evaluate(
    problem, measure, u: np.ndarray, rng, n_starts=100, tol=1e-10
) -> Dataset: ...  # u (n, N); the points are the Dataset's initial points
```

- A **measurement** returns the measured variables for one trial point. A real
  system supplies only this, as a plain array.
- **The oracle has a module of its own**, `oracle.py`, since only benchmark experiments
  need it. `evaluation.py` imports `OracleMeasurement` from it, for `Measure` and to
  tell an oracle's measurement from a real system's, so `oracle.py` never imports
  `evaluation.py`. What both need sits below them: `_find_root` in `solvers.py` and
  `_states` in `problem.py`.
- The **oracle simulation** solves `g` with `y = oracle(x, y, u)` for every state
  (root-finding with CasADi Jacobians), reads the measured variables and adds
  `rng.normal(0, noise_std)`. Its full states are the **oracle values**, returned in an
  `OracleMeasurement`.
- **Oracle vs real system.** Being simulated doesn't make a system an oracle. An in
  silico simulation of a plant (a flowsheet simulator, say) is a real system to the
  library: a custom measurement that returns only the measured variables, with no
  oracle values and no disagreement check. An oracle is the benchmark experiment
  stating the unknown equations in closed form, next to `g`, so the oracle values are
  known and a method can be scored against them. The method never sees the oracle
  either way. A custom measurement that knows its oracle values can return an
  `OracleMeasurement`.
- The library then always solves the **root-finding problem**: `g` for the unmeasured
  states given `u` and the measured variables, from Latin hypercube starts over the
  unmeasured variables' start bounds, then computes `f` (and `f` at the oracle values).
- A root is accepted when its residual norm is below `tol` and it lies within the
  bounds; if no start gives one, `RootFindingError`. A measured variable that isn't
  finite raises it too, before the solve: with no unmeasured states nothing is solved,
  and a measured variable outside `g` leaves a root, so the solve wouldn't notice.
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
- **Measurements from elsewhere.** `reconstruct_states(problem, *, u, measured, rng)` is
  `evaluate` without the measurement: no oracle values, and every row is tried, since
  plant data can have several bad rows, a missing value among them. `RootFindingError`
  names them and keeps the others as `error.dataset`, so the caller decides whether to
  carry on. A `u` or a
  measured variable outside its bounds is still a measurement, so it is kept, with
  `OutOfBoundsWarning`. An unmeasured root outside its bounds is a failed row, because
  the bounds may be what picks the right root.
- `fsolve` always uses the CasADi Jacobian (`use_jacobian` is removed). A failed start is an
  expected failure (Principle 6), so fsolve's convergence warnings are caught and
  nothing is printed; only `RootFindingError` reports that every start failed.

### Dataset and Run (`dataset.py`, `run.py`)

```python
@dataclass(frozen=True)
class Dataset:  # one row per point, in the order the points were taken
    u: np.ndarray  # (n, N)
    x: np.ndarray  # (n, M)  measured and reconstructed
    y: np.ndarray  # (n, P)
    f: np.ndarray  # (n,)
    x_oracle: np.ndarray | None  # benchmark only; None without an oracle
    y_oracle: np.ndarray | None
    f_oracle: np.ndarray | None
    n_initial: int


def extend(dataset, new) -> Dataset: ...  # a new dataset; a point is a one-row dataset
def incumbents(dataset) -> np.ndarray: ...  # np.minimum.accumulate(dataset.f)
def oracle_incumbents(
    dataset,
) -> np.ndarray: ...  # f_oracle at each incumbent's point (#32)


@dataclass(frozen=True)
class AcquisitionSolution:  # acquisition.py (#53)
    value: float  # the acquisition problem's objective at the trial point
    x: np.ndarray | None = None  # (S, M) scenario states; hybrid BO only
    y: np.ndarray | None = None  # (S, P)


@dataclass(frozen=True)
class Run:
    dataset: Dataset
    artifact: dict[str, str]  # how the run was produced, as text (#123)
    method: str  # the method's name (#53)
    surrogates: list[tuple[FittedGP, ...]]  # one per iteration; P GPs (hybrid BO) or
    # 1 (standard BO) each (#114); empty for baselines (default)
    acquisitions: list[AcquisitionSolution]  # one per iteration (#53)
    scenarios: Scenarios | None = None  # hybrid BO only (#29)
    settings: MethodSettings | None = None  # BO only (#120)
```

A BO run holds its method's **settings** in full (#120), the data the iteration plots
replay; the artifact holds them as text, for people, and nothing reads that text back.
The settings dataclasses live in `settings.py`, not with their methods in `bo.py`,
because `run.py` needs them and `bo.py` imports `run.py`. `MethodSettings` is their
union, the one place a new method's settings join.

The run's **artifact** is built by the library (`run_artifact`, before the method draws
anything), not the example: the variables with their bounds, `h_inputs`, `measured`,
`g` and `f` as expression text, the measurement, the method and its settings, the
prior GP's repr (#119), the seed, spawn key and generator state, and the
versions of `hybrid_bo`, Python, CasADi, NumPy and SciPy. The
measurement is recorded by its docstring, else its name, so a custom measurement adds
a description by writing a docstring. A measurement that is a dataclass is recorded
by its repr instead: the oracle simulation (`OracleSimulation`) gives the oracle as
text with its noise and solver settings.

The oracle simulation is a frozen, callable dataclass, not a closure (#141): it is
data (the system, the oracle traced like `g`, a `cas.Function`, and the settings),
and its repr is the measurement's text in the artifact, so it mustn't depend on the
process. A closure would need its `__doc__` set to describe it.

A run's artifact is text for people; the settings beside it are data for the plots. A
partial run in `error.run` carries one too, so it still says how it was made.

A column is found by name with `system.column("x1")`.

### Solvers (`solvers.py`)

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


def solve_with_multistart(nlp: NLP, starts: np.ndarray, solver: Ipopt) -> Solution:
    """Best of the starts; raises NoSolutionError if every start fails."""
```

There is no `Solver` protocol until a second solver exists. The caller draws the starts
with `lhs(...)`. `solve_with_multistart` builds the problem with `cas.Opti`, bounds as
constraints, as the legacy optimizer did, so IPOPT takes the same steps. A failed start
is logged at debug level without a traceback. Starts that stopped at IPOPT's iteration
limit also get one warning per call, under the objective's name: they fail like any
other start, but they are what makes a fit slow (#108).

### GP (`gp.py`)

GPs are data ([ADR 0008](docs/adr/0008-gps-are-data.md)): the prior GP, its mean, its
kernel and its hyperparameters are frozen dataclasses from a closed set.

```python
@dataclass(frozen=True)
class Param:  # a hyperparameter to fit
    init: float
    bounds: tuple[float, float]  # hard limits; the fit's starts are drawn within them
    log: bool = True  # fit in log space


@dataclass(frozen=True)
class Matern52:  # also Matern12, Matern32, SquaredExponential
    lengthscale: float | tuple[float, ...] | Param = Param(1.0, (1e-3, 1e2))
    ard: bool = True  # a fitted lengthscale per input


@dataclass(frozen=True)
class Scale:
    kernel: Kernel
    outputscale: float | Param = Param(1.0, (0.1, 10.0))  # unsquared


@dataclass(frozen=True, kw_only=True)
class GP:  # the prior GP, before it sees any data
    mean: Mean  # Zero()
    kernel: Kernel
    noise_std: float  # known, in measured units, no default (ADR 0009)
    jitter: float = 1e-6


@dataclass(frozen=True)
class FittedGP:  # the prior GP with every Param fitted, and its data
    gp: GP  # no Param left
    X: np.ndarray  # training inputs, scaled
    Y: np.ndarray  # training outputs, scaled
    x_scaler: Scaler
    y_scaler: Scaler


def covariance(kernel, X1, X2): ...  # (n1, n2), a match over the kernel types
def prior_mean(mean, X): ...  # (n, 1)
def fit_gp(gp, X, Y, rng, n_starts=10, solver=Ipopt()) -> FittedGP: ...
def predict(fitted, X) -> tuple: ...  # noise-free (mean, std) in measured units
def log_marginal_likelihood(fitted) -> float: ...  # on the scaled training data
```

An experiment builds each method's GP; there is no default:

```python
gp = hb.gp.GP(
    mean=hb.gp.Zero(),
    kernel=hb.gp.Param(1.0, bounds=(0.1, 10.0)) * hb.gp.Matern52(),
    noise_std=0.01,
)
```

`outputscale * kernel` (a number or a `Param`, on either side) is shorthand that
builds `Scale(kernel, outputscale)`: the experiment shows the output scale where the
math has it, and the stored GP, its repr and its JSON hold the `Scale`. It is the
only operator; `kernel * kernel` raises `TypeError` (the kernels share `_Scalable`).

- **Hyperparameters.** A kernel's hyperparameters are its fields whose type admits a
  `Param`; a number fixes one. They are fitted in a fixed order: a node's own
  hyperparameters before those of the kernels it holds, so `Scale(Matern52())` fits the output scale, then the
  lengthscales, as `default_gp` did. A lengthscale with `ard` (the default) has one
  entry per input, the count taken from the training data at fit time; a fixed one
  given as one number applies to every input. A fitted GP holds the numbers: a float,
  or a tuple of one per input.
- **Units.** Kernel hyperparameters are in scaled units, as the fit sees them
  (lengthscales relative to inputs scaled onto [0, 1], the output scale relative to standardized
  outputs). `noise_std` is in measured units, the units of the training outputs, so an
  experiment states the noise of its measurement as it is: `fit_gp` and `predict`
  convert it with the output scaler, `K = k(X, X) + ((noise_std * slope)**2 + jitter) I`.
  With `noise_std = 0` the fitted GP nearly interpolates (see **Jitter**). A fitted
  noise is out of scope: on a run's few points it sits at its lower bound; it can
  later be `noise_std=Param(...)`.
- **Closed set.** `Scale`, `Matern12`, `Matern32`, `Matern52`, `SquaredExponential`
  and the `Zero` mean; no composition (`Sum`, a product of kernels) and no user kernels
  until an experiment needs them. A new kernel is a dataclass and a case in `covariance`.
- **Jitter** is 1e-6 (#107), a nugget in the sense of Gramacy & Lee (2012), "Cases
  for the nugget in modeling computer experiments". Once a run converges its inputs
  cluster, and with legacy's 1e-10 the training covariance reached cond(K) of about
  1e11: most GP-fit starts then ran to IPOPT's iteration limit, and `fit_gp` took
  15-25s per iteration on the illustrative example. With 1e-6 the posterior std at
  the data is about 1e-3 in scaled units, negligible for EI. `fit_gp` warns when
  cond(K) at the kept hyperparameters exceeds 1e10 (`CONDITION_THRESHOLD`, #108),
  below the 1e11 that stalled the fit.
- **Output scale** multiplies the kernel unsquared, as legacy `OutputScaleKernel` did,
  so `GP(mean=Zero(), kernel=Scale(Matern52()), noise_std=0.0)` fits the same
  hyperparameters as the legacy GP.
- `fit_gp` fits the scalers (`minmax` onto [0, 1] for X, `standard` for Y), puts
  CasADi symbols in place of each `Param`, builds the negative log marginal
  likelihood, draws starts with `lhs` over the (log-transformed) bounds and calls
  `solve_with_multistart`. If every start fails it keeps the initial values and logs a
  warning. A GP with no `Param` is only conditioned on the data, with no draws. With the
  same data and generator its hyperparameters matched the legacy GP bit for bit, and its
  predictions to rounding (checked until #30; the regression baselines still pin the
  fit). Replacing `default_gp` with `GP(Zero(), Scale(Matern52()))` (#119) left every
  run bit for bit the same.
- `predict` traces the fitted GP's conditional mean and variance once as a
  `cas.Function` of X; a numpy X gives arrays `(n,)`, a symbolic X (SX or MX; the
  conditional mean and variance themselves are built in SX) gives `(n, 1)` columns.
  Both are noise-free: the std is the square root of the variance of the latent
  output, with no observation noise added, so callers need no covariance.
- This removes `Parameterized`, `JoinedParameters`, `k_variable`/`m_variable` and the
  name prefixing, and with them the legacy `LinearKernel` ARD bug. `minmax` uses the
  correct intercept (`lower - min * slope`), evaluated so that [0, 1] reproduces the
  legacy values bit for bit.

### Hybrid model (`hybrid_model.py`)

```python
@dataclass(frozen=True)
class Scenarios:
    z: np.ndarray  # (S, P)
    w: np.ndarray  # (S,), >= 0 and sums to 1, else ValueError


def nominal_scenario(n_y: int) -> Scenarios: ...  # z = 0, w = 1


@dataclass(frozen=True)
class DeterministicEquivalent:
    nlp: NLP
    start_bounds: tuple[np.ndarray, np.ndarray]  # (lower, upper), for lhs
    n_u: int  # the variable order of w, read by _unpack
    n_x: int = 0
    n_scenarios: int = 0  # standard BO: w is u alone
    unknown_states_as_variables: bool = False
    scenario_values: cas.Function | None = None


def _pack(u, x=None, y=None): ...  # numpy or CasADi
def _unpack(problem: DeterministicEquivalent, w) -> tuple[u, x, y | None]: ...
def _unpack_scenarios(problem: DeterministicEquivalent, w) -> tuple[u, x, y]: ...


def build_deterministic_equivalent(
    problem,
    u,
    x,
    y,
    unknown_equations,  # residuals (S, P), or None if y is an expression of u and x
    objective,  # (f, x, y, u) expressions -> (objective, std of f)
    objective_name,
) -> DeterministicEquivalent: ...
def gp_deterministic_equivalent(
    problem,
    fitted,
    scenarios,
    objective,
    unknown_states_as_variables,
    objective_name,
) -> DeterministicEquivalent: ...
def fit_surrogate(
    system, dataset, *, gp, rng, n_starts=10, solver=Ipopt()
) -> tuple[FittedGP, ...]: ...
def optimize_hybrid_model(
    problem, fitted, rng, *, n_starts=10, solver=Ipopt()
) -> NominalOptimum: ...  # u, x, y, f
```

- **Hybrid BO is built on the hybrid model, not the other way round (#197).** Everything
  that solves the hybrid model has the same variables, bounds and `g`, with `y` tied to
  the GP. What differs is the objective, and the scenarios follow from it: an
  acquisition function needs a distribution of `f`, which the scenarios approximate,
  and `f` itself needs none.

  | | Objective | Scenarios it needs |
  |---|---|---|
  | `optimize_hybrid_model` | `f` | the nominal scenario |
  | SAA-EI, SoftSAA-EI, SAA-LCB | an acquisition function of the `f_s` | S scenarios |
  | Lin-EI | EI of `f` with its sensitivity | the nominal scenario |

  So `gp_deterministic_equivalent` takes both, `hybrid_acquisition` supplies an
  objective, and `hybrid_bo` is the loop around them: `fit_surrogate`,
  `solve_with_multistart`, evaluate. `acquisition.py` imports `hybrid_model.py`, never
  the reverse. The objective is a function because it is built from the problem's own
  symbols. It also states the std of `f` that `scenario_values` reports, which is 0 for
  the nominal optimum.
- **The unknown equations are the seam (#211).** The oracle optimum has the same
  variables, bounds and `g` as the nominal optimum, with `y` tied to the oracle instead
  of the GPs, and it is the benchmark the hybrid result is tested against, so the
  formulation exists once. `build_deterministic_equivalent` takes the symbols and the
  residuals that tie the unknown states, as expressions over `(u, x, y)`, and has two
  callers: `gp_deterministic_equivalent`, with `y - prediction(x, u)`, and
  `oracle_optimum`, with `oracle(x, y, u) - y`. The residuals read `y` because the
  unknown equations are `0 = h(x, y, u)`: `y = h(x, u)` is only the form the GP learns,
  and an oracle may be implicit.

  Substituting the prediction for `y` (`unknown_states_as_variables=False`) needs an
  explicit `y`, which residuals don't give, so only `gp_deterministic_equivalent` does
  it: it passes the prediction as `y`, and None for the residuals. The builder takes
  expressions, not a function of the symbols, so that this needs no second entry point,
  and each caller builds its symbols once. `oracle_optimum` and `optimize_hybrid_model`
  share the objective `f` (`_minimize_f`), and the dependency runs one way: `oracle.py`
  imports `hybrid_model.py`.
- **`AcquisitionProblem` is an alias of `DeterministicEquivalent`.** Hybrid BO's
  acquisition problem is a deterministic equivalent whose objective is an acquisition
  function. Standard BO's is no deterministic equivalent, having no hybrid model, but
  needs the same fields with no scenarios (`w` is `u`), so it shares the type.
- **`fit_surrogate` takes a `Dataset`.** A dataset is every state at every point, which
  is what the fit needs and what a caller's sensors don't give. Measurements from
  elsewhere become one through `reconstruct_states`, so there is one way in, and a
  finished run's dataset fits as it is.
- **`hybrid_bo` doesn't call `optimize_hybrid_model`.** They solve the same problem with
  a different objective: an acquisition function of `f`, or `f` itself.
  `gp_deterministic_equivalent` requires the objective (ADR 0009), and
  `optimize_hybrid_model` fixes it to `f` by name. `hybrid_bo` calls
  `gp_deterministic_equivalent` through `hybrid_acquisition` because it needs the
  problem unsolved for the replay.
- **`optimize_hybrid_model` is not a method.** It measures nothing, so it has no run, no
  artifact and no settings, only a solution. It has no uncertainty in it, so it trusts
  the GP's mean even far from the data. Its unknown states are always variables, since
  substituting them would drop their bounds.

### Acquisition (`acquisition.py`)

```python
# The settings and choice dataclasses are keyword-only; what chooses the method has
# no default (ADR 0009)
@dataclass(frozen=True, kw_only=True)
class EI: ...


@dataclass(frozen=True, kw_only=True)
class LCB:
    std_factor: float  # finite and >= 0, else ValueError


def saa_scenarios(n: int, n_y: int, rng) -> Scenarios: ...  # z ~ N(0, I), w = 1/S


AcquisitionProblem = DeterministicEquivalent


@dataclass(frozen=True, kw_only=True)
class SAA:
    n_scenarios: int
    unknown_states_as_variables: bool = True  # a formulation, same optimum


@dataclass(frozen=True, kw_only=True)
class SoftSAA:  # SoftSAA-EI, SAA-EI with a smooth maximum
    temperature: float  # in units of the objective
    n_scenarios: int
    unknown_states_as_variables: bool = True


@dataclass(frozen=True, kw_only=True)
class Linearization:  # Lin-EI, over the nominal scenario
    unknown_states_as_variables: bool = True


def hybrid_acquisition(
    problem,
    gp,
    fitted,
    scenarios,
    incumbent,
    acquisition,
    approximation,  # SAA | SoftSAA | Linearization
) -> AcquisitionProblem: ...
def standard_acquisition(
    problem, gp, fitted, incumbent, acquisition
) -> AcquisitionProblem: ...
```

Each returns an `AcquisitionProblem`: the NLP, its start bounds and the order of its
decision vector `w`.

- **The problem owns its variable order.** `_pack` and `_unpack` (in `hybrid_model.py`)
  are the only code that knows it; hybrid BO's trial point, the scenario states read
  back from a solution, the replay and the tests all go through them (or through the
  problem's own functions of w), so changing the order breaks nothing silently.
  `_unpack_scenarios` is `_unpack` with `y` from `scenario_values`, what a solved
  problem over scenarios gives. Standard BO's `w` is `u`, so its trial point is `w`.
- **The acquisition problem says what it computed.** Besides the objective and the
  constraints, hybrid BO's problem carries `scenario_values(w)`: f under each
  scenario, the scenarios' unknown states and the std of f (linearization's s, else
  the weighted std over the scenarios), built from the same expressions as the
  objective. The loop reads the stored scenario states from it, and `replay`
  (ADR 0005) solves the problem's own constraints with u fixed and evaluates the
  objective and `scenario_values`, so the figures can't drift from what the method
  minimized. `bo.iteration_problem(run, problem, i)` rebuilds an iteration's problem
  from a run, through the function the loop builds it with (`bo._acquisition_problem`,
  which reads the incumbent after the evaluations it is told to count), so the two
  can't drift apart. `_pack` needs no
  layout, so the builders use it for the symbols, the bounds and the start bounds
  alike.

- **Validation.** Every method checks its acquisition function the same way (one
  helper, called by the settings and by the builders): anything but an `EI` or `LCB`
  instance raises `TypeError`. This avoids the legacy `formulation_acq` bug. Hybrid
  BO's approximation is checked the same way, by a second helper next to it, which
  raises `TypeError` for anything but an approximation the library knows,
  `NotImplementedError` for a combination that isn't implemented (LCB with
  linearization), and `ValueError` for one that has no meaning (LCB with SoftSAA).
- **Standard BO** computes EI and LCB analytically from the GP mean and std (EI adds
  1e-10 to the std in `gamma`, as legacy `ei_gp` did) and minimizes -EI or LCB over `u`
  within its bounds.
- **Hybrid BO** with SAA computes them as weighted sample averages over the scenarios (SAA-EI,
  SAA-LCB; paper, Problems 8 and 9). Each scenario has its own `x_s` and `y_s`, with `g`
  as constraints. SAA-LCB uses the weighted variance; `w = 1/S` reproduces the legacy
  `sample_variance` (`ddof=0`).
- **The approximation is a setting of hybrid BO, not a method.** `acquisition`
  (`EI | LCB`) says what is computed and `approximation` how: `SAA`, `SoftSAA` or
  `Linearization` (below). The approximation decides which scenarios the problem is
  built over and how its objective combines them; the scenario machinery (`x_s` and
  `g` per scenario, the optional tie of `y_s` to the GP, `_pack`/`_unpack`,
  `scenario_values`) is shared. Settings that only make sense for one approximation
  live on it (`n_scenarios` on `SAA`).
- `unknown_states_as_variables` (on every approximation): if True, each scenario's unknown states are variables
  of the acquisition problem tied to the GP by an equality constraint (paper, Problem 8),
  which helps IPOPT; if False, the GP prediction is substituted wherever they appear.
- The hybrid acquisition problem's variables are `u`, then the `x_s`, then the `y_s` (if
  variables), each state with one entry per scenario (`_pack`). Every variable is
  boxed by its own bounds and started within its own start bounds, and the GP's
  inputs are the columns of `h_inputs`. This avoids the two legacy
  `setup_acq_problem` bugs (`u` start bounds, number of GP inputs). With one
  mechanistic state, as in the illustrative example, the order matches legacy, so the
  Latin hypercube starts and the baseline are the same.
- The scenarios' unknown states, mean + std * z_s, are computed in one place
  (`_scenario_unknown_states`, in `hybrid_model.py`), symbolically for the scenario constraint and
  numerically when the `y_s` are substituted and read back.
- Hybrid BO only needs "mean and std of `y` at given inputs" from the surrogate, through
  `predict`, so other surrogates or hyperparameter samples can be paired with scenarios
  later.
- **More than one unknown state (#114).** The surrogate model is P independent GPs,
  one per unknown state in the order of `y`, each fitted to its own column of `y` on
  the same `h_inputs` data from the one prior GP in the settings. Unknown state j of
  scenario s is `mean_j + std_j * z_sj`, and Lin-EI's `L` is `diag(std_1, ..., std_P)`.
  A surrogate is a tuple of fitted GPs, and standard BO's a 1-tuple, so every run has
  the same type; `load` reads a run saved with one fitted GP per iteration as 1-tuples.
  Not built: per-equation arguments of h, a prior GP per unknown state, and the GP
  panels of the iteration plots for P > 1 (`ValueError` with `gp=True`).

#### Soft SAA (SoftSAA-EI)

SAA-EI averages `max(a_s, 0)` with `a_s = incumbent − f_s`, so its acquisition
problem is flat (zero gradient) wherever no scenario improves on the incumbent and
has a kink per scenario; IPOPT stalls in both (#63). Following LogEI (Ament et al.
2023), `SoftSAA` replaces the maximum with the smooth maximum
`sp_τ(a) = τ log(1 + e^{a/τ})` at the temperature `τ` and minimizes
`−log Σ_s w_s sp_τ(a_s)`. It is a separate approximation, not an option of `SAA`, so
SAA-EI stays the paper's and the two can be compared as settings of one experiment.

- **Computed in logs.** The objective is a logsumexp over scenarios of
  `log w_s + log τ + log log(1 + e^{a_s/τ})` (`_log_softplus`), which takes
  `a/τ` below −30 as its own log (where `log(1 + e^t)` underflows) and
  `t + log(1 + e^{−t})` above 0 (where `e^t` overflows), so the value and gradient
  are finite for any `a/τ`. Far below the incumbent the gradient is that of a
  softmax-weighted `−a_s/τ`, which points toward improvement.
- **The temperature is explicit.** It is in units of the objective, so a default
  would be wrong for most problems; it has none and is recorded in the run (the
  worked example of [ADR 0009](docs/adr/0009-defaults-never-choose-the-method.md)). The
  log already gives a gradient at any temperature, so it only needs to be small: a
  measured point has no scenario spread and scores `τ log 2`, so a temperature
  near the improvements makes the noise-free loop re-measure the incumbent. Too
  small, and it takes tiny local steps (its EI underflows away from the incumbent).
  The experiments use about 1e-3 times the optimum's magnitude (5e-3 for Langermann,
  50e-3 for the illustrative examples), set from trial runs; Rosenbrock, whose
  optimum is 0, uses 100e-3.
- **EI only.** LCB has no maximum to soften, so `LCB` with `SoftSAA` is a
  `ValueError`, not `NotImplementedError`.
- **The stored value is −log SoftSAA-EI**, the objective the problem minimizes, as
  for every approximation. The iteration plot draws it as −EI, `−exp(−value)`:
  −log EI grows like `(f − incumbent)/τ` far from the incumbent, so on a linear axis
  it squashes the regions that matter, and as −EI it reads like SAA-EI and Lin-EI.

#### Linearization (Lin-EI)

SAA-EI needs S copies of the mechanistic states in every acquisition problem.
Linearization approximates EI from one scenario, the nominal scenario (`z = 0`,
weight 1), built through the same scenario machinery: the variables are
`(u, x̄[, ȳ])` and the constraints `g(x̄, ȳ, u) = 0` (plus the tie of `ȳ` to the GP's
mean). It draws nothing from the generator.

With the scenario draws `z ~ N(0, I)` (the paper's ξ), the states `x(u, z)` and
`y(u, z)` solve `0 = g(x, y, u)` and `y = μ(x, u) + L(x, u) z`, with `L` the GP's std.
Expanding `f` to first order in `z` around the nominal states makes it normal:

- **Nominal states** `(x̄, ȳ)`: `0 = g(x̄, ȳ, u)`, `ȳ = μ(x̄, u)` (`z = 0`).
- **Total sensitivity**, by the implicit function theorem at `z = 0`
  (`_sensitivity`): `dx/dz = −(g_x + g_y μ_x)⁻¹ g_y L`, `dy/dz = μ_x dx/dz + L`, and
  `aᵀ = df/dz = f_x dx/dz + f_y dy/dz`. The term in `dL/dx` vanishes at `z = 0`.
- **Lin-EI** is the analytic EI of `N(f(x̄, ȳ, u), s²)` below the incumbent, with
  `s² = aᵀa` (the same `_expected_improvement` as standard BO, with its epsilon).

**The derivative must be the total one.** Both states are random: `x` moves with `y`
through `g`, and through the GP when `x` is an argument of `h`. The partial
derivative `f_y L` misses that. In the illustrative example (`g = x + eˣ − y`, `f` of
`x` only, `h` of `u` only) it gives `s ≡ 0`, so Lin-EI would never explore, while the
total one gives `s = |f_x| L / (1 + e^x̄)`.

`s` is built symbolically in SX: `cas.jacobian` of `g`, `f` and the GP mean with
respect to `x` and `y` at fresh symbols, one `cas.solve` (M×M) for `dx/dz`, then a
`cas.Function` evaluated at the problem's `(x̄, ȳ, u)`, so it works whether `ȳ` is a
variable or the substituted prediction. It is in matrix form, so P > 1 needed no
rework beyond stacking the P GPs' means and stds.

The expansion is first order only: with the second-order term `f` is a generalized
χ², and the closed form no longer holds. Lin-LCB is not implemented. A run stores
the nominal scenario as its scenarios, and each acquisition solution the nominal
states as one-row `x (1, M)` and `y (1, P)`.

### Methods and the optimization loop (`bo.py`, `settings.py`)

```python
@dataclass(frozen=True, kw_only=True)
class HybridBOSettings:
    gp: GP  # required: the experiment states it
    acquisition: EI | LCB  # what is computed, required (ADR 0009)
    approximation: SAA | SoftSAA | Linearization  # how it is computed, required
    n_starts: int = 10  # acquisition multistart
    n_starts_gp: int = 10
    solver: Ipopt = Ipopt()


@dataclass(frozen=True, kw_only=True)
class StandardBOSettings:
    gp: GP  # required
    acquisition: EI | LCB  # required
    n_starts: int = 10
    n_starts_gp: int = 10
    solver: Ipopt = Ipopt()


TrialPoint = Callable[
    [Problem, Dataset, np.random.Generator],
    tuple[np.ndarray, FittedGP, AcquisitionSolution],
]


def optimization_loop(
    problem,
    measure,
    initial_points,
    n_iterations,
    trial_point: TrialPoint,
    rng,
    artifact,  # from run_artifact, before the method draws
    *,
    settings,  # the method's, kept in the run (#120); None if not replayable
    scenarios,  # the method's, kept in the run; None if it has none
) -> Run: ...


def hybrid_bo(
    problem,
    measure,
    initial_points,
    n_iterations,
    rng,
    *,
    settings: HybridBOSettings,  # required
) -> Run: ...
def standard_bo(
    problem,
    measure,
    initial_points,
    n_iterations,
    rng,
    *,
    settings: StandardBOSettings,  # required
) -> Run: ...
def latin_hypercube_sampling(
    problem, measure, initial_points, n_points, rng
) -> Run: ...  # baselines.py
def uniform_sampling(problem, measure, initial_points, n_points, rng) -> Run: ...
```

The baselines are named `*_sampling` because `hb.sampling.uniform` and
`hb.sampling.lhs` are the sampling primitives.

- `hybrid_bo` draws the scenarios from `rng` (SAA; linearization takes the nominal
  scenario and draws nothing), then runs `optimization_loop` with a
  trial-point function that fits the GP on `(u, x)[h_inputs] -> y`, builds
  `hybrid_acquisition`, and solves it with `solve_with_multistart`. The user never draws
  `z`. With `SAA(n_scenarios=25, unknown_states_as_variables=False)` (the legacy
  illustrative setting) it reproduces the legacy illustrative baseline.
- `standard_bo` fits on `u -> f` (measured), with the incumbent the best measured `f`,
  and solves `standard_acquisition` by `solve_with_multistart` from `lhs` starts over
  the `u` start bounds. With the defaults it reproduces the legacy illustrative
  baseline.
- `optimization_loop` takes the artifact and returns the `Run`, so a `HybridBOError` in
  the loop (an evaluation, or the trial-point function) carries the partial run as
  `error.run` for every method: the points evaluated so far (also `error.dataset`) and
  one surrogate per evaluated trial point (a failed evaluation drops its surrogate). It
  holds hybrid BO's scenarios too and builds the `Run` in one place, so a complete and
  a partial run have the same fields. The baselines don't use the loop: they have no
  surrogates, and build their own `Run`.
- The artifact holds the settings without the GP, plus `gp`, the prior GP's repr; the
  fitted GPs stay in the run. The GP is required, so `settings` is too.
- The baselines are not sequential: they draw their whole design and `evaluate` it after
  the initial points.
- `optimization_loop` is public, for new methods.

### Errors

`HybridBOError` and subclasses (`RootFindingError` from #25, `NoSolutionError`, ...). An
error raised in `evaluate` carries the points evaluated before it (`error.dataset`); an
error raised during a run carries the partial `Run` (`error.run`, from #26), so the
caller can save it. The loop does not catch errors. Warnings are `UserWarning`
subclasses (`ReconstructionWarning`) so they can be filtered.

### Plots

`hb.plots.regret` and `hb.plots.convergence` draw a list of runs per label over the
evaluations. `hb.plots.iteration` draws one iteration of a run, a function of the finished run (#55)
that replays its acquisition problem
([ADR 0005](docs/adr/0005-iteration-plots-replay-the-acquisition-problem.md)). Each run's
GP comes from its stored fitted GPs (#119) and the rest from `run.settings`, so a saved
run plots after the experiment's methods change. matplotlib is imported on first use and
is the extra `hybrid-bo[plots]`. Figures use the repo's style file, `hb.plots.STYLE`
(`src/hybrid_bo/hybrid_bo.mplstyle`); put rcParams there, not in plotting code.

## Order of random draws

Every draw in a run comes from that run's generator, in this order:

1. initial points: `lhs` over the `u` start bounds (none if the caller gives them);
2. method setup: hybrid BO draws the scenarios (`rng.standard_normal((S, P))`); the
   baselines draw their whole design; standard BO draws nothing;
3. each evaluation, initial points first: the measurement (oracle simulation: full-solve
   starts, then noise), then the root-finding starts;
4. each iteration: GP-fit starts, then acquisition starts, then the evaluation of the
   trial point as in 3. Hybrid BO with P > 1 unknown states fits its GPs in the order
   of `y`, each from its own `lhs` starts, before the acquisition starts; with P = 1
   the draws are those of a single GP fit.

Root-finding starts are drawn even without noise, because the reconstruction always
runs. Each consumer draws only what it uses, when it uses it. Outside a run,
`reconstruct_states` draws one set of `lhs` starts per row, and `fit_surrogate` and
`optimize_hybrid_model` draw only their `lhs` starts, in that order.

**A generator has two independent sequences.** `lhs` does not consume `rng`'s stream:
it draws from the next spawned child of `rng`'s `SeedSequence` (`rng.spawn(1)[0]`),
exactly as scipy's `qmc.LatinHypercube(d, rng=rng).random(n)` does. It draws that
itself because importing `scipy.stats` took about a second (#172).
So every `lhs` call (initial
points, fsolve and root-finding starts, multistart starts, LHS designs) is ordered only
against other `lhs` calls and `spawn`s, and the stream draws (`normal`, `uniform`,
`standard_normal`: noise, scenarios, the uniform design) only against each other. The
order above holds within each sequence; the order *between* the two sequences (e.g.
noise before or after the oracle's fsolve starts) has no effect on any value. New code must keep `lhs` drawing
what `LatinHypercube(..., rng=rng)` would, and must not add a `spawn` or an `lhs` call
in the middle of a run, or the baselines move. `tests/test_sampling.py` pins `lhs`
to scipy's draw and its spawning.

## Removed

| What | Why | Where to find it |
|---|---|---|
| `array_operations/` numpy branches and their `@overload` stubs | one CasADi path (SX); numeric use goes through `cas.Function` | tag `pre-refactor` |
| `SciPyLocalOptimizer` | never used; only offered as a commented-out alternative | tag `pre-refactor` |
| `use_jacobian` (fsolve without the CasADi Jacobian) | every equation is CasADi-traceable ([ADR 0002](docs/adr/0002-equations-are-casadi-callables.md)) | tag `pre-refactor` |
| `Problem.solve` (true optimum with `h` known) | called nowhere; would be a function on `solve_with_multistart` | tag `pre-refactor` |
| `Config` and `config.yaml` | functions take what they use; problems are Python | tag `pre-refactor` |
| `hybrid_bo.legacy`: the `do_*` routines, `ResultsBO` pickles, the old `Problem`, GP and optimizers | replaced by the new API (#30); studies are saved as JSON (#53) | tag `pre-refactor` |
| Kernel and mean classes, `Parameterized`, `JoinedParameters` | hyperparameters were named specs and kernels functions; since #119 kernels are dataclasses again, but data with no state or methods ([ADR 0008](docs/adr/0008-gps-are-data.md)) | tag `pre-refactor` |
| `PriorGP`, `Hyper`, `default_gp`, `kernels.py` (with `linear` and `constant`) | GPs are data, from a closed set ([ADR 0008](docs/adr/0008-gps-are-data.md)); the linear and constant kernels were unused | git history before #119 |
| Per-iteration plot callbacks | plots are functions of a finished run ([ADR 0003](docs/adr/0003-runs-are-pure-data.md)) | tag `pre-refactor` |
| `hb.cli`, `hb.studies`, `hb.views`, `hb.save`, `hb.load` | a script runs the methods ([ADR 0013](docs/adr/0013-scripts-run-the-methods.md)) | version 0.5 |
| Strategy classes (planned, never built) | methods are settings plus functions | earlier DESIGN.md in git history |

## Future directions

Not built now; the design leaves room for them.

- **Bayesian hyperparameters.** MAP: an optional `prior` on `Param` (#88), added to the
  negative log marginal likelihood. Sampling: a `log_posterior(gp, X, Y)` seam
  returning a `cas.Function` with its gradient over the log-space hyperparameters,
  usable by slice sampling or NUTS (pymc through a custom op, or a standalone sampler). Each scenario
  then pairs a hyperparameter sample with its `z`. Suggested order: MAP, slice sampling,
  NUTS if the dimension grows.
- **Alternatives to SAA.** Sigma points or cubature produce a different `Scenarios`
  (deterministic `z`, specific `w`); the acquisition builders already take weights.
  A moment-matched second-order linearization could be an option of `Linearization`.
