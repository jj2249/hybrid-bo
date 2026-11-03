# hybrid-bo

Bayesian optimization of partially known systems: some equations of the system are
known, the rest are unknown and learned by a Gaussian process, and the two together form
a hybrid model that the optimizer plans with. Notation follows
[arXiv 2603.11199](https://arxiv.org/abs/2603.11199), Problem 4.

## Language

### The problem

**Problem**:
A system together with an objective and its known inequality constraints, if any. The
same problem is solved by different methods.
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
What stands in for an unknown part of the problem. In hybrid BO it models the unknown
equations, with one independent GP per unknown state, all sharing the arguments of the
unknown equations. In standard BO it is one GP of the whole map from decision variables
to objective.
_Avoid_: GP model, emulator

**Prior GP**:
A surrogate model before it sees any data: its mean, its kernel with each
hyperparameter fixed or to be fitted, and its observation noise (`hb.gp.GP`).
_Avoid_: GP spec, GP (on its own, when the prior is meant)

**Fitted GP**:
A prior GP with every hyperparameter fitted to its training data, held with that data
(`hb.gp.FittedGP`). The hyperparameters are fitted by maximizing the marginal
likelihood. Its predictions are the conditional mean and standard deviation of the
noise-free output.
_Avoid_: trained GP, conditioned GP, posterior

**Hyperparameter** (`hb.gp.Param` when fitted):
A field of a GP's kernel, such as a lengthscale or an output scale: a number fixes it,
a `Param` (an initial value and bounds) is fitted to the training data. Kernel
hyperparameters are in scaled units (inputs scaled onto [0, 1], outputs standardized);
the observation noise `noise_std` is in measured units.
_Avoid_: parameter (on its own), kernel parameter

**Jitter** or **nugget** (`jitter`):
A small constant added to the diagonal of a GP's training covariance so it stays
well conditioned. It is a numerical safeguard, not observation noise, which the prior
GP states.
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
The states determined by the known equations; there are none when there are no
known equations (M = 0).

**Unknown states** (`y`, dimension P):
The states determined by the unknown equations; in hybrid BO, each has its own GP
predicting it.
_Avoid_: GP outputs, unknown variables

**Known equations** (`g`, M equations):
The equations `0 = g(x, y, u)` whose form is known. There may be none (M = 0, and
then no mechanistic states): the known part of the problem is the objective alone,
`f(y, u)` with `y = h(u)`, and hybrid BO is BO of a composite function (Astudillo &
Frazier, [arXiv 1906.01537](https://arxiv.org/abs/1906.01537)).
_Avoid_: h_known, mechanistic model

**Unknown equations** (`h`, P equations):
The equations `0 = h(x, y, u)` whose form is not known, written explicitly as
`y = h(x, u)`; hybrid BO learns them with a GP.
_Avoid_: h_unknown

**Known inequality constraints** (`c`, K constraints):
The requirements `c(x, y, u) ≤ 0` whose form is known. An evaluation that violates one
still succeeds, with states and an objective; it is an infeasible evaluation. There may
be none (K = 0).
_Avoid_: known constraints, constraints (on their own), g_known

**Unknown inequality constraints**:
Conditions on the system whose form is not known and which are never measured: an
evaluation that violates one fails, and only that it failed is observed. They are not
part of the problem; in a benchmark experiment the oracle holds them.
_Avoid_: unknown constraints, hidden constraints (an S2MPJ term), failure conditions,
g_unknown

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
A known stand-in for the unknown equations, and for the unknown inequality constraints if
there are any, used only in benchmark experiments to play
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
It either succeeds, as a feasible or an infeasible evaluation, or fails.
_Avoid_: simulation (except the oracle simulation), query

**Feasible evaluation**:
An evaluation that succeeded and satisfies every known inequality constraint.

**Infeasible evaluation**:
An evaluation that succeeded but violates a known inequality constraint. It has states
and an objective, and the surrogate model learns from it like any other.
_Avoid_: failed evaluation, violating evaluation

**Failed evaluation**:
An evaluation at decision variables that violate an unknown inequality constraint, so the
system gives no states and no objective. Only its success label is observed. Whether it
satisfies the known inequality constraints can't be known.
_Avoid_: infeasible evaluation, infeasible trial, crashed evaluation

**Success label**:
Whether an evaluation succeeded (1) or failed (0): evidence about the unknown
inequality constraints alone, since only they fail an evaluation.
_Avoid_: feasibility label, feasibility flag

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
A problem, how it is measured, and the configurations to compare, stated together in
a script that passes them to `hb.cli.main`, so they can be run as studies: "the flash
unit experiment". The script's file name is the experiment's name. Each time it is run
it gives a new study. The shipped ones are in `experiments/`; a person's own can live
anywhere, such as the untracked `scripts/`.
_Avoid_: example, case, test case, application

**Benchmark experiment**:
An experiment whose unknown equations are known to us (the oracle) but hidden from the
methods, so its studies can be scored: oracle incumbents, and regret if the oracle
optimum is stated.
_Avoid_: test case, toy problem

**Configuration**:
A method with its own evaluation budget, labeled in a study. One method can have
several configurations, with different settings or budgets.
_Avoid_: variant, arm

**Evaluation budget** (`n_evaluations`):
How many evaluations a run of a configuration makes in all, its initial points
(`n_initial`) included; the rest are iterations.
_Avoid_: horizon, total iterations

**Initial fraction** (ε):
The share of the evaluation budget spent on initial points, `n_initial /
n_evaluations`.

**Run**:
One complete optimization from the initial points to the end of the optimization loop.
Runs are repeated with different seeds for statistics.

**Study**:
A collection of runs of an experiment's configurations, where run i of every
configuration with the same number of initial points starts from the same initial
points. The number of runs and the seed are the same for every configuration. The unit
that is saved, loaded and analyzed.
_Avoid_: benchmark, experiment, results, batch

**Artifact**:
What a study or a run leaves to say what it ran and how, as text for people. One per
study and one per run. A study's artifact holds the problem, the measurement, each
method with its settings (a BO method's prior GP included), the seed, the oracle optimum if stated, and the versions of
the code and packages; it lists the study's runs but doesn't hold them. A run's
artifact holds the problem, the measurement, its method with its settings, its
generator's seed and state, and the versions, so a run made without a study still says
how it was made.
_Avoid_: record, metadata, manifest, provenance

**Task**:
One run of a configuration in a study: the configuration's label, the configuration,
the run's index and the seed spawned for it (`hb.Task`). A study is its tasks, each
run on its own (`hb.run_task`), then assembled (`hb.assemble`); `hb.cli` runs each
task in a process of its own.
_Avoid_: job, work item

**Task list**:
The file `tasks.json` that `hb.cli` writes when it prepares a study: what the study
will run (the experiment's name, the seed, the number of runs, each configuration's
budget, the oracle optimum, the code version, the problem and the measurement) and its
tasks in order. These are what the script passes to `hb.cli.main`, not where it is:
resuming checks the experiment against them, so a study resumes from its script moved
elsewhere. Unlike the artifact, it is written before anything runs.
_Avoid_: manifest, job list

**Unfinished study**:
A study folder without an artifact: its task list, its study log, and the run files
and per-run logs of the tasks done so far. It can't be loaded or plotted; `resume`
runs the tasks left and assembles it. Its folder keeps its study id taken. Saving
each run as it finishes, so a study can be resumed, is *checkpointing*; the unit is a
run, never part of one.
_Avoid_: partial study

**Launcher**:
What starts a study's tasks and waits for them: `hb.cli`'s `run` and `resume`, which
keep `--n-workers` task processes running, or anything that runs `run-task` commands
(a SLURM array, GNU parallel).
_Avoid_: scheduler (SLURM's role), pool, worker

**Study log**:
What a study logged as it ran: each task as it started and finished, then every run's
lines (each run and iteration with its time, and every warning raised), each starting
with the run's tag. Saved with the study, beside its artifact. It is for reading, not
analysis: the runs hold the results.
_Avoid_: run log

**Per-run log**:
The lines one run logged, saved as the run's task runs, in the study's
`logs/<label>/run-003.log`. When the study is assembled, they are added to the study
log, each starting with the run's tag, `<label>/run-003`.
_Avoid_: run log, worker log, task log

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
The best measured objective value among the feasible evaluations so far; there is none
until a feasible evaluation exists. The point that measured it is the incumbent's point,
the one the optimizer believes is best.
_Avoid_: best value, current best

**Oracle incumbent** (`oracle_incumbents`):
The oracle objective at the incumbent's point: what acting on the run would give.
The benchmark metric. Under noise it can be worse than the best oracle objective so far;
without noise it equals it. There is none until there is an incumbent.
_Avoid_: incumbents_no_noise, true incumbent

**Oracle feasibility**:
Whether an evaluation's oracle values satisfy the known inequality constraints. Under
noise, a feasible evaluation, even the incumbent's, can be infeasible by its oracle
values.
_Avoid_: true feasibility

**Oracle optimum** (`f*`):
The best objective value the oracle allows: the reference that regret is measured
against. Benchmark experiments only; always stated by the experiment, never computed by
a study, since a search can miss it.
_Avoid_: true optimum, global optimum

**Regret**:
The oracle incumbent minus the oracle optimum, after each evaluation. Counted over
evaluations, not iterations, because initial points are not free. Like the oracle
incumbent, there is none before the first feasible evaluation.
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
hybrid BO computes it by an approximation. SAA-EI, SoftSAA-EI and Lin-EI are the same
acquisition function under three approximations.
_Avoid_: formulation_acq, acquisition formulation

**Success classifier**:
The model of the probability of success, learned from the success labels.
_Avoid_: feasibility classifier, constraint model

**Probability of success** (`p_C`):
The success classifier's probability that an evaluation at given decision variables
would not violate any unknown inequality constraint.
_Avoid_: probability of feasibility (for this alone)

**Probability of feasibility** (`p_f`):
The probability that an evaluation at given decision variables would be a feasible
evaluation: the probability of success combined with the known inequality constraints.
_Avoid_: feasibility probability, P(feasible)

**Imputation**:
Giving each failed evaluation a made-up objective, worse than the worst measured one, so
standard BO's GP can learn from it.
_Avoid_: padding, penalty value

**CBO**:
Standard BO with a success classifier: a baseline for hybrid BO's feasibility handling.
_Avoid_: constrained standard BO, CEI-BO

**Feasibility handling**:
How a BO method's acquisition problem accounts for the probability of feasibility:
*explicit*, a chance constraint that it is at least the minimum probability of
feasibility (β), or *implicit*, weighting EI by it (constrained EI). Until there is an
incumbent, the acquisition problem maximizes the probability of feasibility instead.
_Avoid_: explicit/implicit formulation, constraint handling

**Aggregation**:
How hybrid BO combines the scenarios' known inequality constraints with the probability
of success into the probability of feasibility. The *indicator mean* multiplies the
probability of success by the weighted mean of the scenarios' constraint indicators.
_Avoid_: known-constraint handling, Monte Carlo estimate, Independent (the thesis's name)

**Constraint indicator**:
Whether a scenario satisfies every known inequality constraint, either exactly (1 or 0)
or softened by a logistic function of each constraint with a stated slope.
_Avoid_: feasibility indicator, step function

**Approximation**:
How hybrid BO computes its acquisition function, which has no closed form under the
hybrid model: the sample-average approximation, the soft sample-average approximation
or linearization. A setting of hybrid BO, not a method of its own, chosen apart from the
acquisition function (though the soft sample-average approximation and linearization
give EI only).
_Avoid_: formulation, estimator

**Sample-average approximation** (SAA):
The approximation that averages the objective over drawn scenarios, each weighted
`1/S`. Gives SAA-EI and SAA-LCB.
_Avoid_: Monte Carlo approximation, sampling

**Soft sample-average approximation** (SoftSAA):
The sample-average approximation of EI with each scenario's improvement softened by a
smooth maximum at a temperature, so the acquisition function has no flat regions or
kinks. Gives SoftSAA-EI.
_Avoid_: smoothed SAA, LogEI, soft EI, SAA-Soft-EI

**Temperature** (`τ`):
How far the soft sample-average approximation softens the maximum, in units of the
objective; the smaller it is, the closer SoftSAA-EI comes to SAA-EI where that is
positive.
_Avoid_: smoothing, beta

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
