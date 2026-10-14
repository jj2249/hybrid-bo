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
A known stand-in for the unknown equations, used only in test cases to play the part of
the real system. Real applications have no oracle. An in silico simulation of a plant
is not an oracle: to the library it is a real system, run by a custom experiment.
_Avoid_: ground truth, true system

**Oracle values**:
The states and objective an evaluation would give with the oracle and no measurement
noise; for benchmarking only.
_Avoid_: true values, noise-free values, no_noise

### Evaluation

**Trial point**:
The decision variables to evaluate next: the solution of the acquisition problem.
_Avoid_: proposal, candidate, next point

**Experiment**:
In a real application, running the system at a trial point and reading its measured
variables. The oracle simulation plays this part in test cases.
_Avoid_: query, black box

**Evaluation**:
Running the system at given decision variables and obtaining its states and objective.
_Avoid_: simulation (except the oracle simulation), query

**Measured variables**:
The states the system's sensors read at each evaluation, as many as there are unknown
states. Part of the system's description.

**Oracle simulation**:
The experiment in a test case: solving the known equations with the oracle in place of
the unknown equations, then reading the measured variables, with optional noise.

**Root-finding problem**:
Solving the known equations for the states that were not measured, given the decision
variables and the measured variables. Part of every evaluation, after the experiment.
_Avoid_: re-solve, fixed-x evaluation

### Bayesian optimization

**Run**:
One complete optimization from the initial points to the end of the optimization loop.
Runs are repeated with different seeds for statistics.

**Iteration**:
One step of the optimization loop.
_Avoid_: run, step

**Incumbent**:
The best measured objective value so far.
_Avoid_: best value, current best

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
hybrid BO computes it as a sample average over the scenarios (SAA-EI, SAA-LCB).
_Avoid_: formulation_acq, acquisition formulation

**Scenario**:
One realization of the GP's prediction of the unknown states, with its own mechanistic
states, in the sample-average approximation of the acquisition problem.
_Avoid_: GP sample, posterior sample, sample

**Scenario draws** (`z`, one per scenario):
The standard normal draws that define the scenarios (the paper's ξ). Drawn once per run
and the same in every iteration's acquisition problem.
_Avoid_: xi, gaussian_standard_samples

**Scenario weights** (`w`, one per scenario):
The weight of each scenario in the acquisition function's average; `1/S` for SAA.
