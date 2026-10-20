# hybrid-bo

Bayesian optimization of partially known systems: some equations of the system are
known, the rest are unknown and learned by a Gaussian process, and the two together form
a hybrid model that the optimizer plans with. Notation follows
[arXiv 2603.11199](https://arxiv.org/abs/2603.11199), Problem 4.

## Language

### The problem

**Problem**:
A system together with an objective. The same problem is solved by different methods.
_Avoid_: case, benchmark

**System**:
The physical process: its decision variables, states, known equations, the arguments
of its unknown equations, and the bounds on each.
_Avoid_: plant (except for a real one), model

**Method**:
How a problem is solved: hybrid BO, standard BO, or a baseline (Latin hypercube or
uniform sampling). A BO method includes its choice of surrogate model.
_Avoid_: strategy, routine, algorithm

**Surrogate model**:
A GP that stands in for an unknown part of the problem: the unknown equations in
hybrid BO, the whole map from decision variables to objective in standard BO.
_Avoid_: GP model, emulator

**Prior GP**:
A surrogate model before it sees any data: its hyperparameters with their initial
values and bounds, its kernel and its mean, all written as functions of the
hyperparameters.
_Avoid_: GP spec, GP (on its own, when the prior is meant)

**Fitted GP**:
A prior GP conditioned on its training data, at hyperparameters fitted by maximizing
the marginal likelihood. Its predictions are the conditional mean and standard
deviation of the noise-free output.
_Avoid_: trained GP, conditioned GP, posterior

**Jitter** (`jitter`):
A small constant added to the diagonal of a GP's training covariance so it stays
well conditioned, also called a nugget. It is a numerical safeguard, not observation
noise, which a `noise_std` hyperparameter models.
_Avoid_: epsilon

**Hybrid model**:
The known equations together with a surrogate model of the unknown equations.

### Variables and equations

**Decision variables** (`u`, dimension N):
The variables the optimizer chooses.
_Avoid_: inputs, controls, design variables

**States**:
The variables the system settles at once the decision variables are fixed: the
mechanistic states and the unknown states together.
_Avoid_: outputs, process variables, dependent variables

**Mechanistic states** (`x`, dimension M):
The states determined by the known equations.

**Unknown states** (`y`, dimension P):
The states determined by the unknown equations; the GP predicts them.
_Avoid_: GP outputs, unknown variables

**Known equations** (`g`, M equations):
The equations `0 = g(x, y, u)` whose form is known.
_Avoid_: h_known, mechanistic model

**Unknown equations** (`h`, P equations):
The equations `0 = h(x, y, u)` whose form is not known, written explicitly as
`y = h(x, u)`; hybrid BO learns them with a GP.
_Avoid_: h_unknown

**Bounds**:
The box a variable must stay in, possibly infinite. Bounds on decision variables are
part of the problem statement, not a setting of the method.

**Start bounds**:
The finite box from which starting points are drawn: initial points, multistart
starts and root-finding guesses. They default to the bounds when those are finite.

**Objective** (`f`):
The scalar `f(x, y, u)` to minimize.
_Avoid_: cost, target

**Oracle**:
A known stand-in for the unknown equations, used only in benchmark experiments to play
the part of the real system. Real systems have no oracle. An in silico simulation of a
plant is not an oracle: to the library it is a real system, run by a custom measurement.
_Avoid_: ground truth, true system

**Oracle values**:
The states and objective an evaluation would give with the oracle and no measurement
noise; for benchmarking only.
_Avoid_: true values, noise-free values, no_noise

### Evaluation

**Trial point**:
The decision variables to evaluate next: the solution of the acquisition problem.
_Avoid_: proposal, candidate, next point

**Measurement** (`measure`):
Running the system at given decision variables and reading its measured variables. One
per evaluation, for initial points and trial points alike. The oracle simulation plays
this part in benchmark experiments.
_Avoid_: experiment, query, black box, trial

**Evaluation**:
Running the system at given decision variables and obtaining its states and objective.
_Avoid_: simulation (except the oracle simulation), query

**Measured variables**:
The states the system's sensors read at each evaluation, as many as there are unknown
states. Part of the system's description.

**Oracle simulation**:
The measurement in a benchmark experiment: solving the known equations with the oracle
in place of the unknown equations, then reading the measured variables, with optional
noise.

**Root-finding problem**:
Solving the known equations for the states that were not measured, given the decision
variables and the measured variables. Part of every evaluation, after the measurement.
_Avoid_: re-solve, fixed-x evaluation

### Bayesian optimization

**Experiment**:
A problem, how it is measured, and the methods to compare, stated together so they can
be run as studies: "the flash unit experiment". Each time it is run it gives a new study.
_Avoid_: example, case, test case, application

**Benchmark experiment**:
An experiment whose unknown equations are known to us (the oracle) but hidden from the
methods, so its studies can be scored: oracle incumbents, and regret if the oracle
optimum is stated.
_Avoid_: test case, toy problem

**Run**:
One complete optimization from the initial points to the end of the optimization loop.
Runs are repeated with different seeds for statistics.

**Study**:
A collection of runs of an experiment's methods, where run i of every
method starts from the same initial points. The unit that is saved, loaded and analyzed.
_Avoid_: benchmark, experiment, results, batch

**Artifact**:
What a saved study leaves to say what it ran and how: the problem, the measurement,
each method with its settings, the seed, the oracle optimum if stated, and the versions
of the code and packages. One per study. It lists the study's runs but doesn't hold
them.
_Avoid_: record, metadata, manifest, provenance

**Reproducible**:
A study is reproducible when its artifact records a clean code version (every file
committed, none changed or untracked) and running its experiment again at that version,
with the same seed, gives the same runs within a tolerance. It covers the computation
only: a measurement of a real system can't be repeated, so such a study can be
reproduced only up to its measurements. A study run from an unclean tree is still
saved, but it isn't reproducible.
_Avoid_: replicable, deterministic

**Iteration**:
One step of the optimization loop.
_Avoid_: run, step

**Incumbent**:
The best measured objective value so far. The point that measured it is the
incumbent's point, the one the optimizer believes is best.
_Avoid_: best value, current best

**Oracle incumbent** (`oracle_incumbents`):
The oracle objective at the incumbent's point: what acting on the run would give.
The benchmark metric. Under noise it can be worse than the best oracle objective so far;
without noise it equals it.
_Avoid_: incumbents_no_noise, true incumbent

**Oracle optimum** (`f*`):
The best objective value the oracle allows: the reference that regret is measured
against. Benchmark experiments only; always stated by the experiment, never computed by
a study, since a search can miss it.
_Avoid_: true optimum, global optimum

**Regret**:
The oracle incumbent minus the oracle optimum, after each evaluation. Counted over
evaluations, not iterations, because initial points are not free.
_Avoid_: simple regret, error

**Initial points**:
The decision variables evaluated before the optimization loop starts.
_Avoid_: initial samples, initial design

**Optimization loop** (`optimization_loop`):
The sequence of iterations after the initial points; each iteration solves the
acquisition problem and evaluates the result.
_Avoid_: BO loop, routine

**Acquisition problem**:
The optimization problem solved each iteration; its solution is the next trial point.
Its known equations are constraints, so it involves no root-finding.
_Avoid_: acquisition optimization

**Acquisition function**:
The objective of the acquisition problem: expected improvement (EI) or lower confidence
bound (LCB). Standard BO computes it analytically from the GP's mean and variance;
hybrid BO computes it by an approximation. SAA-EI and Lin-EI are the same acquisition
function under two approximations.
_Avoid_: formulation_acq, acquisition formulation

**Approximation**:
How hybrid BO computes its acquisition function, which has no closed form under the
hybrid model: the sample-average approximation or linearization. A setting of hybrid
BO, not a method of its own, chosen apart from the acquisition function (though
linearization gives EI only).
_Avoid_: formulation, estimator

**Sample-average approximation** (SAA):
The approximation that averages the objective over drawn scenarios, each weighted
`1/S`. Gives SAA-EI and SAA-LCB.
_Avoid_: Monte Carlo approximation, sampling

**Scenario**:
One realization of the GP's prediction of the unknown states, with its own mechanistic
states, in hybrid BO's acquisition problem: many drawn ones for the sample-average
approximation, the nominal scenario alone for linearization.
_Avoid_: GP sample, posterior sample, sample

**Scenario draws** (`z`, one per scenario):
The standard normal draws that define the scenarios (the paper's ξ). Drawn once per run
and the same in every iteration's acquisition problem.
_Avoid_: xi, gaussian_standard_samples

**Scenario weights** (`w`, one per scenario):
The weight of each scenario in the acquisition function's average; `1/S` for SAA.

**Nominal states** (`x̄`, `ȳ`):
The states the hybrid model gives when the GP is replaced by its mean: the mechanistic
states solve the known equations with the unknown states at the GP's mean there. They
are not the expected states, since the known equations are nonlinear.
_Avoid_: mean states, mean realization

**Nominal scenario**:
The scenario with zero scenario draws and weight 1, whose states are the nominal states:
linearization's only scenario.
_Avoid_: mean scenario, zero scenario

**Linearization**:
The approximation that expands the objective to first order in the scenario draws
around the nominal states, accounting for how the mechanistic states move with the
unknown states, so the objective is normal with a mean and standard deviation in closed
form. Gives Lin-EI.
_Avoid_: linear approximation, delta method, Taylor approximation
