# Design: target API

This is where the restructure is heading. [ROADMAP.md](ROADMAP.md) lists the PRs (R0 to R9).
Each PR may refine names or signatures. When one does, it updates this file in the same PR.

## Why

The code is hard to change for three reasons:

1. **Duplication.** The four routines in `routines/` are near-copies of one BO loop. For
   example, `latin_hypercube_sampling.py` and `uniform_sampling.py` differ in 13 of about
   385 lines.
2. **Ungrouped data.** Related values travel as separate arrays: `u`, `x`, `x_no_noise`,
   `f`, `f_no_noise`, `incumbents` and `incumbents_no_noise`, each in `_initial`, `_bo`
   and `_complete` versions. `Problem` has 8 separate bound arrays.
3. **Stateful objects, scikit-learn style.** You configure an object, call methods in the
   right order, and read the results off its state:
   - Training a GP is `fit` the transformers → `set_XY_train` → `create_training_problem`
     → poke `gp.optimizer.lower_bounds_starting_points`/`X0` → `solve_training_problem`.
   - Optimisers are `set_problem` → `X0` → `solve()`.
   - One GP object is retrained in place across runs, so storing its state needs
     `deepcopy`.
   - `Problem` has 9 evaluation entry points.
   - `Config` is passed everywhere, so a signature doesn't show what a function uses.

## Principles

1. **Data in, data out.** No set-then-call sequences. Fitting and solving return new
   values and never mutate their inputs.
2. **Frozen dataclasses** for data and configuration. Classes exist only where users swap
   implementations (kernels, means, solvers, strategies), and those classes are immutable.
3. **Functions take what they use.** There is no `Config` in the library.
4. **One way to do each thing.** One evaluation function. One kernel code path that works
   for numeric and CasADi-symbolic values.
5. **Explicit randomness.** Anything random takes an `rng: np.random.Generator`.
6. **Errors raise** specific exceptions. The caller decides whether to carry on.
   Expected failures (e.g. one multistart start not converging) are logged without
   tracebacks.
7. **Rows are points.** A point is 1-D, shape `(n,)`. A set of points is
   `(n_points, n)`, and `f` over a set is `(n_points,)`. `(n, 1)` column vectors appear
   only inside CasADi glue code.

## Decisions

- **Randomness (R0).** Before the restructure, one *logic* PR changes how randomness is
  drawn and regenerates the baselines (see R0). From then on, every restructure PR keeps
  the regression baselines unchanged, with no compatibility shims.
- **Saving.** The library returns values. The example/CLI saves them (pickle or npz),
  loops over runs and handles failures. The library has no `results_dir`, overwrite
  checks or try/except per run.
- **flash_unit.** Before R1, tag the current commit `pre-refactor` and say in
  `examples/flash_unit/README.md` that the example targets that tag. Exclude it from ty
  and tests. After the refactor it is ported, and its custom evaluation becomes a custom
  simulator.

## Target API

All snippets are illustrative. `hb` is `import hybrid_bo as hb`.

### Bounds and sampling (R1, done; `src/hybrid_bo/bounds.py`)

```python
@dataclass(frozen=True, init=False)
class Bounds:
    lower: np.ndarray  # (n,), read-only float
    upper: np.ndarray  # (n,), read-only float

    # Accepts array-likes; rejects non-1-D input, shape mismatch and lower > upper
    def __init__(self, lower: ArrayLike, upper: ArrayLike): ...
    @classmethod
    def unbounded(cls, n: int) -> "Bounds": ...


# Each returns (n, d)
def lhs(bounds: Bounds, n: int, rng) -> np.ndarray: ...
def uniform(bounds: Bounds, n: int, rng) -> np.ndarray: ...
```

`lhs` draws exactly like the code it replaced: `scipy.stats.qmc.LatinHypercube(d, rng=rng).random(n)`,
scaled as `lower + (upper - lower) * samples`. Two hand-written copies remain:
- the fixed-x solve in `problem.py`, which keeps its Known issue until R2;
- `MultiStartOptimizer.create_lhs_samples`, which goes in R3.

`stack(*bounds)` arrives with its first user, R6.

### Problem (R1, done)

```python
@dataclass(frozen=True)
class Problem:
    f: cas.Function  # f(u, x) -> (1, 1)
    h_known: cas.Function  # h_known(u, x) -> (n_h_known, 1)
    h_unknown: cas.Function  # the true system; only the simulator uses it
    u_bounds: Bounds  # box for u in the acquisition problem
    x_bounds: Bounds  # box for x; Bounds.unbounded(n_x) if none
    u_start_bounds: Bounds  # finite; initial points, multistart starts
    x_start_bounds: Bounds  # finite; fsolve guesses, multistart starts
    threshold_equality: float = 1e-10  # moves to Simulator.tol in R2

    n_u: int  # property, from f (also n_x, n_h_known, n_h_unknown)
    # __post_init__: function arities and sizes, bound sizes, finite start bounds
```

All four bounds are required, so there are no defaults computed from other arguments
(CLAUDE.md Conventions). `Problem.solve` was deleted, since nothing called it. If someone
needs the true optimum later, it becomes a function built on `minimize`.

### Simulation (R2)

```python
@dataclass(frozen=True)
class Measurement:
    indices: list[int]  # measured states
    noise_std: float = 0.0


@dataclass(frozen=True)
class Evaluation:  # rows are points
    u: np.ndarray  # (n, n_u)
    x: np.ndarray  # (n, n_x)  as measured (noise applied, other states re-solved)
    f: np.ndarray  # (n,)
    x_true: np.ndarray  # (n, n_x)  noise-free
    f_true: np.ndarray  # (n,)


@dataclass(frozen=True)
class Simulator:
    n_starts: int = 100  # fsolve restarts (was n_starts_max_evaluate_problem)
    use_jacobian: bool = True
    tol: float = 1e-10  # was Problem.threshold_equality

    def __call__(self, problem, u, rng, measurement=None, noise=None) -> Evaluation:
        """u: (n, n_u). noise: (n, len(measurement.indices)), drawn by the caller."""
```

This replaces `evaluate_with_simulation`, `evaluate_with_fixed_x` and
`evaluate_with_noisy_simulation`, their `single_*` and `single_*_original` variants, and
the `original` flag. A custom simulator is any callable with the same signature. That is
how flash_unit's fallback logic will come back, with no subclassing. Solving `h_known`
with some states fixed becomes a private helper. The draw order inside a call must stay
the same as today: per point, LHS guesses for the full solve, then (if there is noise)
LHS guesses for the fixed-x solve.

### Optimisation (R3)

```python
@dataclass(frozen=True)
class NLP:
    f: cas.Function  # w -> (1, 1)
    bounds: Bounds  # on w
    h: cas.Function | None = None  # h(w) = 0
    g: cas.Function | None = None  # g(w) <= 0


@dataclass(frozen=True)
class Solution:
    x: np.ndarray  # (n,)
    f: float


class Solver(Protocol):
    def __call__(self, nlp: NLP, starts: np.ndarray) -> list[Solution | None]: ...


@dataclass(frozen=True)
class Ipopt:  # was CasadiOptimizer + MultiStartOptimizer
    plugin_options: dict = field(default_factory=lambda: {"print_time": 0})
    solver_options: dict = field(default_factory=lambda: {"print_level": 0})
    # builds one cas.Opti per call, then loops over the starts (same as today)


@dataclass(frozen=True)
class Scipy:  # was SciPyLocalOptimizer
    method: str = "L-BFGS-B"
    options: dict | None = None


def minimize(nlp: NLP, starts: np.ndarray, solver: Solver) -> Solution:
    """Best of the starts; raises NoSolutionError if every start fails."""
```

The `set_problem`, `X0`, `set_x0`, `create_lhs_samples` and
`lower_bounds_starting_points` attributes, and the `__getstate__` workarounds, all go.
The caller draws starting points with `lhs(start_bounds, n, rng)`.

### Hyperparameters, kernels, means (R4)

```python
@dataclass(frozen=True)
class Param:
    value: Any  # np.ndarray, float, or a CasADi symbol during training
    bounds: tuple[float, float]
    log: bool = True  # optimise in log space
    trainable: bool = True


@dataclass(frozen=True)
class Matern52:
    lengthscale: Param = Param(1.0, (1e-3, 1e2))
    ard: bool = True  # fit_gp expands a scalar lengthscale to one per input

    def __call__(self, X1, X2):  # numeric or symbolic, (n1, d), (n2, d) -> (n1, n2)
        ...


@dataclass(frozen=True)
class Scaled:  # was OutputScaleKernel
    base: Kernel
    scale: Param = Param(1.0, (0.1, 10.0))


# Trainable Params, depth-first over dataclass fields
def params(obj) -> list[Param]: ...


# Same structure with new values (via dataclasses.replace)
def with_values(obj, values: Sequence) -> obj: ...
```

Training builds the objective by calling the *same* `__call__` on
`with_values(gp, symbols)`. That removes `k_variable`/`m_variable`, `Parameter.symbol`,
`Parameterized`, `JoinedParameters` and the name prefixing that `GP`/`OutputScaleKernel`
apply to other objects' parameters.

- **Order matters.** The flattened order must match today's `JoinedParameters`
  (`noise_std`, then the kernel's parameters depth-first, outer before inner, then the
  mean's), so the multistart starting points stay the same.
- **Kernels to port:** Matern12/32/52/Inf (keep the aliases), Linear, Constant, Zero.
- **Means to port:** Zero, Constant.

### GP and scaling (R5)

```python
@dataclass(frozen=True)
class Scaler:  # y_scaled = slope * y + intercept, per column
    slope: np.ndarray
    intercept: np.ndarray

    def apply(self, X): ...
    def invert(self, X): ...


# minmax keeps today's intercept (a Known issue, fixed in its own PR)
def minmax(X) -> Scaler: ...
def standard(Y) -> Scaler: ...


@dataclass(frozen=True)
class GP:  # the model spec
    kernel: Kernel
    mean: Mean = ZeroMean()
    noise_std: Param = Param(1e-2, (1e-6, 1.0))
    jitter: float = 1e-10


@dataclass(frozen=True)
class FittedGP:
    gp: GP  # with fitted values
    X: np.ndarray  # training inputs (scaled)
    Y: np.ndarray  # training outputs (scaled)
    x_scaler: Scaler
    y_scaler: Scaler

    # (mean (n,), cov (n, n)) in original units; X may be numeric or symbolic
    def predict(self, X) -> tuple: ...


def fit_gp(gp, X, Y, rng, n_starts=10, solver=Ipopt()) -> FittedGP: ...
```

`fit_gp` fits the scalers, builds the negative log marginal likelihood NLP over the
trainable `Param`s (in log space where `log=True`), draws starts with
`lhs(transformed param bounds, n_starts, rng)` and calls `minimize`. If every start fails,
it keeps the initial values and logs a warning, as today.

**Numerics.** Today the hybrid acquisition samples in scaled space and then inverts. In
original units that is the same affine map, but rounding may differ. If a baseline moves
beyond tolerance, compute the samples in scaled space as before.

### Acquisition (R6)

```python
@dataclass(frozen=True)
class EI: ...


@dataclass(frozen=True)
class LCB:
    std_factor: float = 1.0


@dataclass(frozen=True)
class GPStructure:  # which variables the hybrid GP maps
    u_inputs: list[int]
    x_inputs: list[int]
    x_outputs: list[int]  # one output for now (single-output GP)


@dataclass(frozen=True)
class Acquisition:
    nlp: NLP
    start_bounds: Bounds
    decode: Callable  # w -> u (n_u,)


def hybrid_acquisition(
    problem, model: FittedGP, structure, incumbent, z, kind, outputs_as_vars=False
) -> Acquisition: ...
def standard_acquisition(problem, model: FittedGP, incumbent, kind) -> Acquisition: ...
```

`hybrid_acquisition` replaces `routines/hybrid_bo.py:setup_acq_problem`. The four
repeated bound concatenations become `stack(...)` over per-block `Bounds` (add `stack` to
`bounds.py` here). Keep the Known issue (`problem.u_bounds.upper` used as an upper start
bound, marked with a comment in `setup_acq_problem`) until its fix PR. The new structure makes it a one-word fix. `formulation_acq` strings become
`EI()`/`LCB()`, and both routines validate them the same way.

### The loop (R7)

```python
@dataclass(frozen=True)
class History:  # one run, every evaluated point in order
    u: np.ndarray  # (n, n_u)
    x: np.ndarray  # (n, n_x) measured
    f: np.ndarray  # (n,)
    x_true: np.ndarray
    f_true: np.ndarray
    n_initial: int
    incumbents: np.ndarray  # property: np.minimum.accumulate(f)
    incumbents_true: np.ndarray

    def append(self, evaluation: Evaluation) -> "History": ...  # returns a new History
    def best(self) -> tuple[np.ndarray, float]: ...


@dataclass(frozen=True)
class Proposal:
    u: np.ndarray  # (n_u,)
    model: FittedGP | None = None


class Strategy(Protocol):
    # Per-run state, e.g. z or an LHS design
    def init(self, problem, n_iterations, rng) -> Any: ...
    def propose(self, problem, history, state, rng) -> Proposal: ...


@dataclass(frozen=True)
class Run:
    history: History
    models: list[FittedGP]  # the model behind each proposal (empty for sampling)


def run_bo(
    problem,
    strategy,
    u_initial,
    n_iterations,
    rng,
    simulator=Simulator(),
    measurement=None,
    callback=None,
) -> Run:
    """Draw noise for all points, state = strategy.init(...), evaluate u_initial, then
    repeat: proposal = strategy.propose(...); evaluate; history = history.append(...);
    callback(history, proposal.model)."""
```

Strategies are frozen dataclasses:
- `HybridBO(gp, structure, acquisition=EI(), n_samples=25, n_starts=10, n_starts_gp=10, solver=Ipopt(), outputs_as_vars=False)`.
  Its `init` draws `z` (n_samples). `propose` fits the GP on `(history.u, history.x)`,
  builds `hybrid_acquisition`, calls `minimize` from `lhs(acq.start_bounds, n_starts, rng)`
  and decodes the result.
- `StandardBO(gp, acquisition=EI(), n_starts=10, n_starts_gp=10, solver=Ipopt())`.
  Fits on `(history.u, history.f)`.
- `LatinHypercube()` / `Uniform()`. `init` draws the whole design, and `propose` returns
  the next row.

This replaces the four routines, `ResultsBO`, `ResultsGP` and `get_sampling_based_data`.
Plotting is a `callback`, which the example supplies. A plot that needs randomness takes
its own `rng`, not the run's.

**Order of random draws for one run**, set in R0 and fixed from then on. Every draw
comes from the run's own generator, `config.rng.spawn(n_runs)[i]`:
0. the caller draws the initial points: `qmc.LatinHypercube(n_u, rng=rng).random(n_initial)`,
   scaled to the u start bounds;
1. noise for all `n_initial + n_iterations` points:
   `rng.normal(0.0, noise_std, size=(n_points, len(measured)))`;
2. `strategy.init`:
   - hybrid: `z = rng.standard_normal((n_samples, 1))`;
   - standard BO: no draw;
   - LHS: `qmc.LatinHypercube(n_u, rng=rng).random(n_iterations)`, scaled;
   - uniform: `rng.uniform(size=(n_iterations, n_u))`, scaled;
3. evaluate the initial points;
4. per iteration, `propose` (GP fit starts, then acquisition starts), then evaluate.

The new code must make exactly these calls, in this order (e.g. `lhs()` is
`LatinHypercube(...).random(n)` scaled, not some other sampler). Otherwise the
baselines move.

Today the GP is also retrained after the last evaluation. That extra fit comes last, so
dropping it doesn't change the history.

### What the example does (R7, R8)

```python
rng = np.random.default_rng(seed)
runs = []
for run_rng in rng.spawn(n_runs):
    u0 = hb.lhs(problem.u_start_bounds, n_initial, run_rng)
    try:
        runs.append(
            hb.run_bo(problem, strategy, u0, n_iterations, run_rng, measurement=...)
        )
    except hb.HybridBOError as e:
        log.error("run failed: %s", e)
save(runs, results_dir)  # the example's choice of format
```

### Config mapping (R8)

| `Config` field | Goes to |
|---|---|
| `n_u`, `n_x` | dropped (read from `problem.f`) |
| `indices_u_input_gp`, `indices_x_input_gp`, `indices_x_output_gp` | `GPStructure` |
| `indices_x_no_gp`, `indices_x_opt` | dropped (derived inside `hybrid_acquisition`; `indices_x_opt` is unused) |
| `use_output_gp_as_opt_var` | `HybridBO.outputs_as_vars` |
| `n_starts_training_gp` | `HybridBO.n_starts_gp` / `StandardBO.n_starts_gp` |
| `n_starts_acq_optimization` | `HybridBO.n_starts` / `StandardBO.n_starts` |
| `n_samples_gp` | `HybridBO.n_samples` |
| `formulation_acq`, `factor_lcb_std` | `EI()` / `LCB(std_factor)` |
| `epsilon_ei_standard` | dropped (unused; `ei` keeps its own default) |
| `indices_x_measured`, `std_measurement_noise` | `Measurement` |
| `n_starts_max_evaluate_problem`, `use_jacobian_evaluate_problem` | `Simulator` |
| `n_iterations_bo` | `run_bo(n_iterations=...)` |
| `seed`, `n_runs_bo`, `create_plots`, `save_bo_results`, `save_extended_results` | the example |
| `rng` | the example (`np.random.default_rng(seed)`, `.spawn(n_runs)`) |

Whether the example keeps YAML is decided in Stage 5 (examples).

### Target example (end of Stage 5)

```python
import casadi as cas
import numpy as np
import hybrid_bo as hb

u, x = cas.SX.sym("u"), cas.SX.sym("x", 2)
problem = hb.Problem(
    f=cas.Function("f", [u, x], [(6 * x[0] - 2) ** 2 * cas.sin(12 * x[0] - 4)]),
    h_known=cas.Function("h_known", [u, x], [x[0] + cas.exp(x[0]) - x[1]]),
    h_unknown=cas.Function("h_unknown", [u, x], [cas.sin(u) - x[1]]),
    u_bounds=hb.Bounds([-2.0], [2.0]),
    x_bounds=hb.Bounds.unbounded(2),
    u_start_bounds=hb.Bounds([-2.0], [2.0]),
    x_start_bounds=hb.Bounds([-2.5, -2.0], [1.0, 2.0]),
)
strategy = hb.HybridBO(
    gp=hb.GP(hb.kernels.Scaled(hb.kernels.Matern52())),
    structure=hb.GPStructure(u_inputs=[0], x_inputs=[], x_outputs=[1]),
)
rng = np.random.default_rng(0)
run = hb.run_bo(
    problem, strategy, hb.lhs(problem.u_bounds, 2, rng), n_iterations=5, rng=rng
)
print(run.history.best())
```

## Restructure PRs

Every PR keeps `tests/data/regression/` unchanged unless it is marked *logic*. The
regression test is ported to each new API but keeps the JSON keys: the `ResultsBO`-style
field names are mapped from the new objects inside the test (`u_initial = history.u[:n_initial]`
and so on). Changes to how expressions are built can shift results by about 1e-12,
which is within `rtol=1e-6`. If a baseline fails, find out why. Never regenerate the
baselines in a non-logic PR.

Each PR also updates `hybrid-bo-src` (module map, contracts), `hybrid-bo-examples` if
the example API changes, the README if usage changes, and this file if the design moves.

- **R0 (logic): per-run randomness. Done.**
  - Each run gets its own generator, and the draws follow the order above.
  - The routines take `rngs` after `u_initial_complete`.
    `get_u_initial(problem, rng)` returns one run's points.
  - LHS/uniform no longer train a GP or take one. Their plots show only the true function
    and the sampled points. A GP can come back in the plot callback once `fit_gp` exists
    (R5).
  - Noise has one column per measured state (fixes that Known issue).
  - Plot functions still use `config.rng`, which no longer affects the runs.
- **R1: `Bounds`, `lhs`/`uniform`, `Problem` as a frozen dataclass. Done.**
  - `pre-refactor` tag (at `main`'s `94396db`) pushed; flash_unit excluded from ty.
  - `Problem.solve` deleted; `tests/test_bounds.py` added.
  - `examples/main.py` dispatches with `match` and looks up plot functions only when
    plotting is on.
- **R2: `Simulator`, `Measurement`, `Evaluation`.** Delete the 9 evaluation methods.
- **R3: `NLP`, `Solution`, `Ipopt`, `Scipy`, `minimize`.**
  - Routines and `GP` stop setting optimiser attributes.
  - Expected failures stop printing tracebacks (`optimizers.py:295`).
- **R4: `Param`, functional kernels and means, `params`/`with_values`.**
  - Port `test_kernels.py`.
  - The single code path fixes the `LinearKernel` ARD Known issue, so remove that
    xfail marker and say so in the PR. Examples don't use `LinearKernel`, so the
    baselines don't change.
- **R5: `Scaler`, `GP`, `FittedGP`, `fit_gp`.**
  - Delete `affine_transformers.py`, the `GP` methods and `parameterization.py`.
  - Port `test_gp.py` and `test_transformers.py`.
- **R6: `EI`, `LCB`, `GPStructure`, `Acquisition`, the two acquisition builders.**
- **R7: `History`, `Proposal`, `Run`, `run_bo`, the four strategies.**
  - Delete `routines/` and `results.py`.
  - The example saves the results and supplies plot callbacks.
- **R8: remove `Config` from the library.** The example builds objects from its config.
- **R9: drop the `@overload` stubs** (169 of them) and decide whether `array_operations/`
  stays.
- **Then (logic):** exception types (`HybridBOError` and subclasses), then the Known
  issues in ROADMAP.md, one PR each.
