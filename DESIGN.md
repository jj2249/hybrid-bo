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
- **Saving.** The library returns values. Examples save them (pickle or npz) and loop
  over runs.

## Target example

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

# Test case: an oracle plays the real system
experiment = hb.oracle_experiment(
    problem, oracle=lambda v: [cas.sin(v.u)], noise_std=0.0
)

# Method: how we search
settings = hb.HybridBOSettings(acquisition=hb.EI(), n_scenarios=25)  # default GP

rng = np.random.default_rng(0)
runs = []
for run_rng in rng.spawn(10):
    initial_points = hb.lhs(system.u, n=2, rng=run_rng)
    runs.append(hb.hybrid_bo(problem, experiment, initial_points, 5, settings, run_rng))

hb.plots.convergence(runs)
```

## Target API

All snippets are illustrative.

### Problem

```python
@dataclass(frozen=True)
class Var:
    name: str
    lower: float = -np.inf
    upper: float = np.inf
    start: tuple[float, float] | None = None  # defaults to (lower, upper) if finite


@dataclass(frozen=True, init=False)
class System:
    u: list[Var]  # N decision variables
    x: list[Var]  # M mechanistic states
    y: list[Var]  # P unknown states (P = 1 for now; P > 1 raises)
    g: cas.Function  # built from the callable: g(x, y, u) -> (M,)
    h_inputs: list[Var]  # the arguments of h, from u and x
    measured: list[Var]  # the states the sensors read, len(measured) == P

    def __init__(self, u, x, y, g: Callable, h_inputs, measured): ...


@dataclass(frozen=True, init=False)
class Problem:
    system: System
    f: cas.Function  # built from the callable: f(x, y, u) -> scalar

    def __init__(self, system: System, f: Callable): ...
```

Construction calls each callable once with CasADi symbols (`v.name` for every variable)
and builds the `cas.Function`s in declaration order. It checks: unique names, finite
start bounds, `len(g) == M`, `len(measured) == P`, `h_inputs` drawn from `u` and `x`.
Variables are looked up by identity or name, never with `==`, because CasADi's `==` is
symbolic.

### Evaluation

```python
Experiment = Callable[
    [np.ndarray, np.random.Generator], np.ndarray
]  # u (N,) -> measured


def oracle_experiment(
    problem,
    oracle: Callable,
    noise_std: float = 0.0,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> Experiment: ...


def evaluate(problem, experiment, u: np.ndarray, rng) -> History: ...  # u (n, N)
```

- An **experiment** returns the measured variables for one trial point. A real
  application supplies only this.
- The **oracle simulation** solves `g` with `y = oracle(x, u)` for every state
  (root-finding with CasADi Jacobians), reads the measured variables and adds
  `rng.normal(0, noise_std)`. Its full states are the **oracle values**.
- The library then always solves the **root-finding problem**: `g` for the unmeasured
  states given `u` and the measured variables, then computes `f`. When oracle values
  exist and the reconstruction disagrees with them, it warns (the measured variables do
  not determine the states uniquely).
- `fsolve` always uses the CasADi Jacobian (`use_jacobian` is removed).

### History and Run

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


@dataclass(frozen=True)
class Run:
    history: History
    surrogates: list[FittedGP]  # one per iteration; empty for baselines
    scenarios: Scenarios | None  # hybrid BO only
    record: dict[str, str]  # how the run was produced, as text
```

The **record** is built by the library, not the example: variable names and bounds,
`g`, `f` and the oracle as expression text, the method settings, the GP spec and kernel
expression, the seed and run index, and the versions of `hybrid_bo` (with git commit),
CasADi, NumPy and SciPy. A custom experiment can add a `description`.

A column is found by name with `system.column("x1")`.

### Solvers

```python
@dataclass(frozen=True)
class NLP:
    f: cas.Function  # w -> scalar
    lower: np.ndarray
    upper: np.ndarray
    g: cas.Function | None = None  # g(w) = 0


@dataclass(frozen=True)
class Ipopt:
    plugin_options: dict = field(default_factory=lambda: {"print_time": 0})
    solver_options: dict = field(default_factory=lambda: {"print_level": 0})


def multistart(nlp: NLP, starts: np.ndarray, solver: Ipopt) -> Solution:
    """Best of the starts; raises NoSolutionError if every start fails."""
```

There is no `Solver` protocol until a second solver exists. The caller draws the starts
with `lhs(...)`.

### GP

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
    hyperparameters: list[Hyper]  # declaration order is the optimization order
    kernel: Callable  # (h, X1, X2) -> (n1, n2)
    mean: Callable = zero_mean  # (h, X) -> (n,)
    jitter: float = 1e-10


def default_gp(n_inputs: int) -> GP: ...  # scaled Matern-5/2 with ARD and noise


@dataclass(frozen=True)
class Scaler:  # scaled = slope * value + intercept, per column
    slope: np.ndarray
    intercept: np.ndarray


@dataclass(frozen=True)
class FittedGP:  # data only (ADR 0003)
    hyperparameters: dict[str, np.ndarray]
    X: np.ndarray  # training inputs, scaled
    Y: np.ndarray  # training outputs, scaled
    x_scaler: Scaler
    y_scaler: Scaler


def fit_gp(gp, X, Y, rng, n_starts=10, solver=Ipopt()) -> FittedGP: ...
def predict(
    gp, fitted, X
) -> tuple: ...  # (mean, std) in original units; X numeric or symbolic
```

Kernels are plain functions (`matern12/32/52`, `squared_exponential`, `linear`,
`constant`) combined with arithmetic:

```python
gp = hb.GP(
    hyperparameters=[
        hb.Hyper("scale", 1.0, bounds=(0.1, 10.0)),
        hb.Hyper("lengthscale", 1.0, bounds=(1e-3, 1e2), size=2),
        hb.Hyper("noise_std", 1e-2, bounds=(1e-6, 1.0)),
    ],
    kernel=lambda h, X1, X2: h.scale**2 * hb.kernels.matern52(X1, X2, h.lengthscale),
)
```

`fit_gp` fits the scalers, calls `kernel` and `mean` once with CasADi symbols for the
hyperparameters, builds the negative log marginal likelihood, draws starts with `lhs`
over the (log-transformed) hyperparameter bounds and calls `multistart`. If every start
fails it keeps the initial values and logs a warning. This removes `Parameterized`,
`JoinedParameters`, `k_variable`/`m_variable` and the name prefixing, and with them the
`LinearKernel` ARD Known issue. `minmax` uses the correct intercept
(`min_t - min * slope`).

### Acquisition

```python
@dataclass(frozen=True)
class EI: ...


@dataclass(frozen=True)
class LCB:
    std_factor: float = 1.0


@dataclass(frozen=True)
class Scenarios:
    z: np.ndarray  # (S, P)
    w: np.ndarray  # (S,), sums to 1


def saa_scenarios(n: int, P: int, rng) -> Scenarios: ...  # z ~ N(0, I), w = 1/S


def hybrid_acquisition(
    problem,
    gp,
    fitted,
    scenarios,
    incumbent,
    acquisition,
    unknown_states_as_variables=True,
) -> tuple[NLP, np.ndarray]: ...
def standard_acquisition(
    problem, gp, fitted, incumbent, acquisition
) -> tuple[NLP, np.ndarray]: ...
```

Each returns the NLP and its start bounds.

- **Standard BO** computes EI and LCB analytically from the GP mean and std.
- **Hybrid BO** computes them as weighted sample averages over the scenarios (SAA-EI,
  SAA-LCB; paper, Problems 8 and 9). Each scenario has its own `x_s` and `y_s`, with `g`
  as constraints. SAA-LCB uses the weighted variance; `w = 1/S` reproduces today's
  `sample_variance` (`ddof=0`).
- `unknown_states_as_variables`: if True, each scenario's unknown states are variables
  of the acquisition problem tied to the GP by an equality constraint (paper, Problem 8),
  which helps IPOPT; if False, the GP prediction is substituted wherever they appear.
- Hybrid BO only needs "mean and std of `y` at given inputs" from the surrogate, through
  `predict`, so other surrogates or hyperparameter samples can be paired with scenarios
  later.

### Methods and the optimization loop

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
    [Problem, History, np.random.Generator], tuple[np.ndarray, FittedGP]
]


def optimization_loop(
    problem, experiment, initial_points, n_iterations, trial_point: TrialPoint, rng
) -> tuple[History, list[FittedGP]]: ...


def hybrid_bo(
    problem, experiment, initial_points, n_iterations, settings, rng
) -> Run: ...
def standard_bo(
    problem, experiment, initial_points, n_iterations, settings, rng
) -> Run: ...
def latin_hypercube(problem, experiment, initial_points, n_points, rng) -> Run: ...
def uniform(problem, experiment, initial_points, n_points, rng) -> Run: ...
```

- `hybrid_bo` draws the scenarios from `rng`, then runs `optimization_loop` with a
  trial-point function that fits the GP on `(u, x)[h_inputs] -> y`, builds
  `hybrid_acquisition`, and solves it with `multistart`. The user never draws `z`.
- `standard_bo` fits on `u -> f`.
- The baselines are not sequential: they draw their whole design and `evaluate` it after
  the initial points.
- `optimization_loop` is public, for new methods.

### Errors

`HybridBOError` and subclasses (`RootFindingError`, `NoSolutionError`, ...). An error
raised during a run carries the partial `Run` (`error.run`), so the caller can save it.
The loop does not catch errors.

### Plots

`hb.plots` (matplotlib optional): convergence of incumbents across runs, the surrogate
with its uncertainty band for one-dimensional `u`, and the scenarios at an iteration.
All are functions of a finished `Run` and the specs from the example.

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
in the middle of a run, or the baselines move. `tests/test_problem.py` pins this
scipy behavior.

## Removed

| What | Why | Where to find it |
|---|---|---|
| `array_operations/` numpy branches and their `@overload` stubs | one CasADi path (SX or MX); numeric use goes through `cas.Function` | tag `pre-refactor` |
| `SciPyLocalOptimizer` | never used; only offered as a commented-out alternative | tag `pre-refactor` |
| `use_jacobian` (fsolve without the CasADi Jacobian) | every equation is CasADi-traceable ([ADR 0002](docs/adr/0002-equations-are-casadi-callables.md)) | tag `pre-refactor` |
| `Problem.solve` (true optimum with `h` known) | called nowhere; would be a function on `multistart` | tag `pre-refactor` |
| `Config` and `config.yaml` | functions take what they use; problems are Python | tag `pre-refactor` |
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
- **Benchmark metric.** The oracle objective at the point the optimizer believes is
  best, computed from the history. Its own logic PR, checked against the paper's figures.

## Restructure PRs

Every PR keeps `tests/data/regression/` unchanged unless it is marked *logic*. Each PR
updates `hybrid-bo-src`, `hybrid-bo-examples`, the README and this file where its change
affects them.

The old code is moved to `hybrid_bo/legacy/` and the new modules are built at their
final paths. New code never depends on old code (old code may call new helpers such
as `lhs`), so no shims are needed and deleting `legacy/` touches nothing new. The
switch PR ports the
regression test and the illustrative example and must reproduce the baselines from the
draw-order PR exactly.

0. **Design** (docs): CONTEXT.md, ADRs, this file, ROADMAP.md, CLAUDE.md conventions.
1. **R1 trimmed** (#17). *Done.* flash_unit excluded from ty (it targets the
   `pre-refactor` tag); `lhs(lower, upper, n, rng)`/`uniform(...)` in `sampling.py`
   (PR 4 changes them to take `Var`s); `Problem.solve` and `use_jacobian` deleted;
   `examples/main.py` dispatches with `match`. Two hand-written LHS copies remain until
   their PRs: the fixed-x root-finding guesses (PR 5) and
   `MultiStartOptimizer.create_lhs_samples` (PR 6).
2. **Draw order** (*logic*). *Done.* The old routines draw in the order above:
   noise per evaluation (`noise_std` instead of a pre-drawn array), root-finding
   always runs, `measurement_noise.pkl` dropped (it is `x - x_no_noise` on the
   measured states). Baselines regenerated.
3. **Legacy**. *Done.* The old code is in `hybrid_bo/legacy/`, imported as
   `hybrid_bo.legacy`; the top-level `hybrid_bo` exports only the new API. Legacy may
   import new code, never the reverse. Result pickles saved before this PR
   (`hybrid_bo.results.ResultsBO`) need the old path to unpickle.
4. **Problem**: `Var`, `System`, `Problem`.
5. **Evaluation**: `oracle_experiment`, root-finding, `evaluate`, `History`, error types.
6. **Solvers**: `NLP`, `Ipopt`, `multistart`.
7. **GP**: `Hyper`, kernel functions, `GP`, `fit_gp`, `predict`, scalers, `default_gp`;
   port the kernel, GP and scaler tests.
8. **Acquisition**: `EI`, `LCB`, `Scenarios`, the two builders.
9. **Switch**: `optimization_loop`, the four methods, `Run` and its record; port the
   regression test and the illustrative example; delete `legacy/`, `Config` and YAML.
   If too big, the baselines (which only need PR 5) go first.
10. **Plots and CLI**: `hb.plots`; a CLI for the examples.
