Walkthrough: the illustrative example
=====================================

This page teaches hybrid BO through the illustrative example of the paper,
`Bayesian optimization for partially known systems using hybrid models
<https://arxiv.org/abs/2603.11199>`_. The code shown is
``examples/illustrative_example.py``; the page quotes it rather than copying it. The
API reference has the signatures.

The problem class
-----------------

A *problem* is a *system* together with an *objective*. The optimizer chooses the
*decision variables* ``u``. Once they are fixed, the system settles at its *states*:
the *mechanistic states* ``x``, determined by the *known equations* ``g``, and the
*unknown states* ``y``, determined by the *unknown equations* ``h``, whose form is not
known. The problem is

.. math::

   \min_u \; f(x, y, u) \quad \text{s.t.} \quad
   0 = g(x, y, u) \;\; \text{(known)}, \qquad
   y = h(x, u) \;\; \text{(unknown)}

with each variable inside its *bounds*. Each *evaluation* runs the system at some
``u`` and reads its *measured variables*, as many as there are unknown states.

The illustrative example has one variable of each kind:

- the decision variable ``u`` in ``[-2, 2]``,
- the mechanistic state ``x``, with the known equation ``x + exp(x) = y``,
- the unknown state ``y``, with the unknown equation ``y = sin(u)``,
- the objective ``f = (6x - 2)^2 sin(12x - 4)``, which depends on ``x`` only.

The sensors measure ``y``. Nobody tells the optimizer that ``y = sin(u)``: in this
test case an *oracle* knows it and plays the real system.

Why a hybrid model
------------------

*Standard BO* treats the system as a black box. It learns the map ``u -> f`` directly
with a GP, the *surrogate model*. Here that map is ``f(x(sin(u)))``: the oscillating
objective composed with the known equation and the unknown one. The GP has to learn
all of it from evaluations.

*Hybrid BO* learns only what is unknown. Its GP models the unknown equation,
here ``u -> y``, a smooth map that a few points describe well. The *hybrid model* is
that GP together with the known equations. To judge a decision ``u``, hybrid BO
propagates the GP's prediction of ``y`` through ``g`` to ``x`` and then through ``f``,
so the structure that is known is never learned, and the GP's uncertainty about ``y``
becomes uncertainty about ``f``.

Stating it in code
------------------

Each variable is a :py:class:`~hybrid_bo.problem.Var` with its bounds. ``x`` and
``y`` are unbounded, so they need *start bounds*: the finite box from which starting
points (initial points, multistart starts, root-finding guesses) are drawn. ``u`` has
finite bounds, which are its start bounds.

.. literalinclude:: ../examples/illustrative_example.py
   :language: python
   :start-at: u = hb.Var(
   :end-before: system = hb.System(

The :py:class:`~hybrid_bo.problem.System` lists the variables, the known equations,
the arguments of the unknown equations (``h_inputs``) and the measured variables. The
equations are functions of ``v``, which holds the variables by name, and return a
list with one CasADi expression per equation. The
:py:class:`~hybrid_bo.problem.Problem` adds the objective.

.. literalinclude:: ../examples/illustrative_example.py
   :language: python
   :start-at: system = hb.System(
   :end-before: # Test case:

The *experiment* runs the system at a trial point and returns the measured variables.
A real application passes a function that does this. A test case uses the *oracle
simulation*, :py:func:`~hybrid_bo.evaluation.oracle_experiment`, which solves the
known equations with the oracle in place of the unknown equations, then reads the
measured variables, optionally with noise.

.. literalinclude:: ../examples/illustrative_example.py
   :language: python
   :start-at: # Test case:
   :end-before: # Methods:

What an iteration does
----------------------

A *run* evaluates the *initial points*, then runs the *optimization loop*. Each
*iteration* of hybrid BO does four things.

1. **Fit the GP** to the unknown states of every point evaluated so far, over the
   arguments of ``h`` (here ``u -> y``).
2. **Solve the acquisition problem.** Its solution is the next *trial point*. The GP's
   prediction at ``u`` is a distribution over ``y``, which the *sample-average
   approximation* (SAA) represents by *scenarios*: scenario ``s`` takes
   ``y_s = mean(u) + std(u) * z_s``, where the *scenario draws* ``z_s`` are standard
   normal draws made once per run. (By default ``y_s`` is also a variable of the
   acquisition problem, tied to this prediction by an equality constraint; the example
   substitutes the prediction instead, with ``unknown_states_as_variables=False``.)
   Each scenario has its own mechanistic states ``x_s``, constrained by the known
   equations ``0 = g(x_s, y_s, u)``, and its own objective ``f_s``. The *acquisition function* is a weighted average over the
   scenarios: expected improvement over the *incumbent* (SAA-EI) or a lower
   confidence bound (SAA-LCB). The known equations are constraints of the acquisition
   problem, so it needs no root-finding. It is solved with IPOPT from several starts.
3. **Run the experiment** at the trial point, which measures ``y``.
4. **Solve the root-finding problem**: the known equations, for the states that were
   not measured, given ``u`` and the measured variables. Here that is
   ``x + exp(x) = y`` for ``x``. Then compute ``f``.

Steps 3 and 4 together are an *evaluation*; the initial points are evaluated the same
way. Standard BO differs only in steps 1 and 2: its GP models ``u -> f``, and it
computes EI or LCB analytically from the GP's mean and variance. The paper has the
formulations (Problems 8 and 9) and the analysis.

Running it and reading a run
----------------------------

A *method* is how a problem is solved. The example compares hybrid BO with standard
BO and two baselines that sample without a model, Latin hypercube and uniform
sampling. Each method is a function with the same arguments, so the example keeps
them in a dictionary, with their settings bound by ``functools.partial``.

.. literalinclude:: ../examples/illustrative_example.py
   :language: python
   :start-at: # Methods:
   :end-before: # The study

Every run gets its own random generator, which draws everything the run needs: the
initial points, the scenarios, the multistart starts and any noise. A *study*
(:py:func:`~hybrid_bo.studies.study`) runs every method ``n_runs`` times from
generators spawned from one seed, the same ones for every method, so run ``i`` of
each method starts from the same initial points, and the same seed gives the same
study. A method that fails raises a
:py:class:`~hybrid_bo.errors.HybridBOError` carrying the partial run, which the study
keeps. The examples' CLI, ``examples/main.py``, runs a study of the example's
methods with its :py:class:`~hybrid_bo.studies.StudySettings` and saves it:

.. literalinclude:: ../examples/main.py
   :language: python
   :start-at: study = hb.study(
   :end-at: hb.save(study, study_dir)
   :dedent:

A method returns a :py:class:`~hybrid_bo.run.Run`, which is plain data, so
:py:func:`~hybrid_bo.saving.save` writes a study's runs as JSON and
:py:func:`~hybrid_bo.saving.load` reads them back without the code that produced
them.

- ``run.history`` (:py:class:`~hybrid_bo.history.History`) holds every evaluated
  point in order, the initial points first: ``u``, ``x``, ``y`` and ``f``, measured or
  reconstructed by the root-finding problem. With an oracle it also holds the
  *oracle values*, what the evaluation would give without measurement noise.
  :py:func:`~hybrid_bo.history.oracle_incumbents` turns them into the benchmark
  metric: the oracle objective at the incumbent's point after each evaluation.
- ``run.method`` and ``run.index`` are the label the run was run under and its
  position among that method's runs.
- ``run.surrogates`` holds one fitted GP per iteration, the one behind that
  iteration's trial point, and ``run.acquisitions`` the acquisition problem's
  solution: the acquisition value at the trial point and, for hybrid BO, the
  scenario states there. Both are empty for the baselines.
- ``run.scenarios`` holds hybrid BO's scenario draws and weights, the same in every
  iteration, and is ``None`` for the other methods.
- ``run.record`` describes, as text, how the run was produced: the problem's
  variables and equations, the experiment, the method and its settings, the seed and
  the package and code versions.

``pixi run example illustrative_example`` runs every method and saves the study to
``results/illustrative_example/``; ``pixi run plot illustrative_example``
draws the newest one (see ``examples/main.py``).

Plotting a study
----------------

The plots read a finished study, so a saved one is plotted without running it again.
:py:func:`~hybrid_bo.plots.convergence` draws, per method, the median oracle incumbent
over the evaluations with the interquartile range across runs shaded;
:py:func:`~hybrid_bo.plots.regret` draws the same against the oracle optimum on a log
scale, as in the paper. Both count evaluations, initial points included, since those
are not free.

:py:func:`~hybrid_bo.plots.iteration` draws one iteration of one run: the state before
its trial point. For hybrid BO it has three panels, the steps of the iteration above:
the GP of ``y`` over ``u``; ``f`` under the scenarios, with ``x_s`` solving the known
equations; and the acquisition function, the objective of the acquisition problem with
``u`` fixed at each grid point. The last two replay the acquisition problem the run
solved, from its stored GP, scenarios and incumbent, so the curve passes through the
acquisition value stored at the trial point. Standard BO's figure has the GP of ``f``
and its analytic acquisition function; a baseline's has the points only.
