Writing an experiment
=====================

An experiment is one Python file in ``experiments/``. It states a problem, how the
system is measured, and the methods to compare, then hands them to the command line,
which runs them as a study and plots it. This page builds one block by block, quoting
``experiments/illustrative_example.py``. The quickest start is to copy that file,

.. code-block:: bash

   cp experiments/illustrative_example.py experiments/my_experiment.py

change each block as described below, and run it with
``pixi run experiment my_experiment``. The file name is the experiment's name; it must
be a valid Python module name (letters, digits and underscores).

The problem
-----------

The library minimizes an objective over the *decision variables* ``u``, the values
you choose. Once ``u`` is fixed the system settles at its *states*: the *mechanistic
states* ``x``, set by *known equations* ``g`` you can write down, and the *unknown
states* ``y``, set by *unknown equations* ``h`` you can't:

.. math::

   \min_u \; f(x, y, u) \quad \text{s.t.} \quad
   0 = g(x, y, u) \;\; \text{(known)}, \qquad
   y = h(x, u) \;\; \text{(unknown)}

A GP learns ``h`` from measurements. In the illustrative example, ``u`` is in
``[-2, 2]``, the known equation is ``x + exp(x) = y``, the unknown one is
``y = sin(u)`` and the objective is ``f = (6x - 2)^2 sin(12x - 4)``.

A system may have several unknown states; hybrid BO then fits one GP per unknown
state, all over the same arguments of ``h``. It may also have no mechanistic states
and no known equations: leave out ``x`` and ``g``, and the objective ``f(y, u)`` is the
only known part (BO of a composite function).

CasADi in two minutes
---------------------

You write ``g``, ``f`` (and, for a benchmark, the oracle) as Python functions of ``v``,
which holds one variable per name: ``v.u``, ``v.x``, ``v.y``. The library calls each
function once, with `CasADi <https://web.casadi.org/>`_ symbols rather than numbers, and
keeps the expression it returns. From that expression CasADi computes exact
derivatives, which the NLP solver (IPOPT) needs to solve the acquisition problem. The
same ``g`` serves twice: as constraints of the acquisition problem, and as the
root-finding problem that recovers the unmeasured states after each measurement.

What that means when you write the equations:

- Arithmetic (``+``, ``*``, ``**``, ...) works as usual.
- Use CasADi's functions: ``cas.exp``, not ``np.exp`` or ``math.exp`` (likewise
  ``cas.sin``, ``cas.log``, ``cas.sqrt``). ``math.exp(v.x)`` silently gives ``nan``,
  and numpy functions work for some operations, with a warning, and fail for others.
- No Python ``if``, ``max`` or ``min`` on variables, since a symbol has no value to
  compare: use ``cas.if_else(v.x > 0, a, b)``, ``cas.fmax`` and ``cas.fmin``. Prefer
  smooth expressions, since the solver relies on derivatives.
- ``g`` returns a list with one expression per mechanistic state, each equal to zero at
  the solution (``x + exp(x) = y`` becomes ``v.x + cas.exp(v.x) - v.y``). ``f``
  returns one expression.

Variables, system and problem
-----------------------------

Each variable is a :py:class:`~hybrid_bo.problem.Var`: a name (a valid Python
identifier, read as ``v.<name>``), its *bounds*, and its *start bounds*, the finite
box from which initial points and solver starting points are drawn. Bounds default to
infinite; start bounds default to the bounds when both are finite, so an unbounded
variable needs ``start``. Make start bounds a realistic range of the variable.

.. literalinclude:: ../experiments/illustrative_example.py
   :language: python
   :start-at: u = hb.Var(
   :end-before: system = hb.System(

The :py:class:`~hybrid_bo.problem.System` takes keyword arguments: the variables, the
known equations ``g`` (with ``x``, both optional), the arguments of the unknown
equations (``h_inputs``, from ``u`` and ``x``) and
the ``measured`` variables, the states your sensors read (from ``x`` and ``y``). There
must be as many measured variables as unknown states, so the root-finding problem can
recover the rest. The :py:class:`~hybrid_bo.problem.Problem` adds the objective ``f``,
which is minimized.

.. literalinclude:: ../experiments/illustrative_example.py
   :language: python
   :start-at: system = hb.System(
   :end-before: # Measurement:

Both raise a ``ValueError`` straight away if the shapes don't fit (say ``g`` has more
entries than there are mechanistic states).

The measurement
---------------

The *measurement* runs the system at a trial point ``u`` and returns the measured
variables. It is a function ``(u, rng) -> measured``
(:py:data:`~hybrid_bo.evaluation.Measure`): ``u`` is an array with one entry per
decision variable, in the order of ``system.u``, and it returns an array with one
entry per measured variable, in the order of ``system.measured``. ``rng`` is a numpy
random generator, for any noise you add.

A benchmark experiment: an oracle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test a method you write the unknown equations in closed form, the *oracle*, and
:py:func:`~hybrid_bo.evaluation.oracle_measure` simulates the system from ``g`` and
the oracle together. The methods only see the measured variables, with normal noise
of the standard deviation you state (``noise_std``, 0 for none); the study also keeps
the true (*oracle*) values, so it can score the methods.

.. literalinclude:: ../experiments/illustrative_example.py
   :language: python
   :start-at: # Measurement:
   :end-before: # Methods:

The *oracle optimum* is the true minimum of the problem, which the study measures
regret from. State it: the study never searches for it, since a search can miss it.
Find it once with :py:func:`~hybrid_bo.evaluation.oracle_optimum` (with many starts),
check it, and paste the value as above. If a run later evaluates a point below it, the
study warns with :py:class:`~hybrid_bo.errors.OracleOptimumWarning`.

Your own system
~~~~~~~~~~~~~~~

For a system whose unknown equations you don't know, write the measurement yourself.
Here a simulator stands in for the plant (any Python code: a flowsheet, an ODE solve,
a call to another program):

.. code-block:: python

   import numpy as np

   from my_lab import simulate_reactor  # your simulator


   def measure(u, rng):
       """The outlet concentration of the reactor simulation."""
       result = simulate_reactor(temperature=u[0], residence_time=u[1])
       return np.array([result.outlet_concentration])

The docstring (else the function's name) is how the measurement is recorded with every
run, so say what it measures. A measurement written as a dataclass is recorded by its
repr instead, as the oracle simulation is. Without an oracle the library knows only what was
measured:

- there is no oracle optimum, so no regret and no ``regret.png``;
- the convergence figure shows the incumbent, the lowest ``f`` the run measured,
  instead of the oracle incumbent, and the summary's oracle and regret columns are
  empty;
- don't pass ``optimum`` or ``true_curves`` to the command line (below).

The measurement is called once per evaluation, so its cost sets the cost of the
study: ``n_runs * n_evaluations`` calls per configuration (below).

Methods and settings
--------------------

A *method* is how the problem is solved. ``methods`` maps a label to a method; every
method takes the same arguments. A BO method's settings, which name its GP, are bound
with ``functools.partial``:

.. literalinclude:: ../experiments/illustrative_example.py
   :language: python
   :start-at: # The prior GP
   :end-before: # Configurations

- :py:func:`~hybrid_bo.bo.hybrid_bo` with
  :py:class:`~hybrid_bo.settings.HybridBOSettings`: the prior GP (``gp``; see
  below), the acquisition function (:py:class:`~hybrid_bo.acquisition.EI`, or
  :py:class:`~hybrid_bo.acquisition.LCB` with a ``std_factor``), how it is
  approximated (``approximation``, e.g. :py:class:`~hybrid_bo.acquisition.SAA`), and
  the solver starts (``n_starts``). The first three are required: a default may set
  how hard or how precisely we search, never what we search for, so an experiment
  shows which method it runs. ``SAA`` takes the number of scenarios
  (``n_scenarios``, required) and ``unknown_states_as_variables``: ``False`` (as
  above) makes the acquisition problem smaller; the default, ``True``, often helps
  the solver.
  :py:class:`~hybrid_bo.acquisition.Linearization` (Lin-EI, as
  ``hybrid_bo_lin`` above) builds the acquisition problem from one scenario, the
  nominal states, so it is smaller still; it gives EI only, and takes
  ``unknown_states_as_variables`` too.
  :py:class:`~hybrid_bo.acquisition.SoftSAA` (SoftSAA-EI, as ``hybrid_bo_soft``
  above) is SAA-EI with each scenario's improvement softened by a smooth maximum,
  so the solver doesn't stall where no scenario improves. It takes a
  ``temperature``, in units of the objective, with no default (smaller is closer to
  SAA-EI; keep it below the improvements you want: the experiments take about
  ``1e-3`` times the optimum's magnitude, e.g. ``50e-3`` for an optimum near -50), and
  ``SAA``'s settings; it gives EI only. Its stored acquisition value
  is -log EI; the iteration plot draws it as -EI.
- :py:func:`~hybrid_bo.bo.standard_bo` with
  :py:class:`~hybrid_bo.settings.StandardBOSettings`: BO with a GP of the objective, as the
  usual comparison. Its GP and acquisition function are required too.
- :py:func:`~hybrid_bo.baselines.latin_hypercube_sampling` and
  :py:func:`~hybrid_bo.baselines.uniform_sampling`, which sample without a model and
  take no settings.

The label names the method in the results, so one method can appear twice with
different settings:

.. code-block:: python

   methods = {
       "hybrid_bo_ei": functools.partial(
           hb.hybrid_bo,
           settings=hb.HybridBOSettings(
               gp=gp, acquisition=hb.EI(), approximation=hb.SAA(n_scenarios=25)
           ),
       ),
       "hybrid_bo_lcb": functools.partial(
           hb.hybrid_bo,
           settings=hb.HybridBOSettings(
               gp=gp,
               acquisition=hb.LCB(std_factor=2.0),
               approximation=hb.SAA(n_scenarios=25),
           ),
       ),
   }

Configurations
--------------

A *configuration* is a method with its own evaluation budget:
:py:class:`~hybrid_bo.studies.Configuration` takes the method, ``n_evaluations``,
the evaluations of a run in all, and ``n_initial``, how many of them are initial
points. The rest, ``n_evaluations - n_initial``, are iterations (design points for
the baselines). Neither has a default, and ``n_initial`` must be at least 1 and at
most ``n_evaluations``. A study runs ``configurations``, which maps a label to a
configuration; the shipped experiments give every method the same budget:

.. literalinclude:: ../experiments/illustrative_example.py
   :language: python
   :start-at: # Configurations
   :end-before: # The study

To compare splits of one budget between initial points and iterations, give one
method several configurations, e.g. *initial fractions* ε = ``n_initial /
n_evaluations`` of 0.2 and 0.5:

.. code-block:: python

   configurations = {
       f"hybrid_bo_{n_initial}": hb.Configuration(
           method=methods["hybrid_bo"], n_evaluations=20, n_initial=n_initial
       )
       for n_initial in (4, 10)
   }

Run ``i`` of every configuration with the same ``n_initial`` starts from the same
initial points. A BO method doesn't depend on how many evaluations it has left, so
with the same seed and ``n_initial`` a run with a larger budget passes through the
same first evaluations as one with a smaller budget: the longest run gives the regret
of every smaller budget. The baselines draw their whole design at the start, so Latin
hypercube sampling doesn't have this property, and uniform sampling has it only
without measurement noise.

A sweep of one method over many ``n_initial`` makes the convergence figure
unreadable, so pass the sweep's figures to the command line instead (see
:doc:`reading_results`):

.. code-block:: python

   cli.main(
       "my_sweep", problem, measure, configurations, study_settings,
       optimum=optimum,
       figures={
           "budget": hb.plots.budget_heatmap,
           "at_budget_20": functools.partial(hb.plots.regret_at_budget, budget=20),
           "regret_by_initial_fraction": hb.plots.regret_by_initial_fraction,
       },
   )

``figures`` maps each figure's file name to a function of the study that draws it;
the default is ``convergence`` and ``regret``.

The GP
------

Each BO method states its prior GP, :py:class:`~hybrid_bo.gp.GP`: a mean, a kernel
and the observation noise. Hybrid BO's GP maps the arguments of ``h`` to the unknown
state; standard BO's maps ``u`` to ``f``. The means and kernels are a fixed set in
``hb.gp``: the :py:class:`~hybrid_bo.gp.Zero` mean, the kernels
:py:class:`~hybrid_bo.gp.Matern12`, :py:class:`~hybrid_bo.gp.Matern32`,
:py:class:`~hybrid_bo.gp.Matern52` and :py:class:`~hybrid_bo.gp.SquaredExponential`,
which multiply by an output scale written in front of them:

.. code-block:: python

   outputscale = hb.gp.Param(1.0, bounds=(0.1, 10.0))
   lengthscale = hb.gp.Param(1.0, bounds=(1e-3, 1e2))
   gp = hb.gp.GP(
       mean=hb.gp.Zero(),
       kernel=outputscale * hb.gp.Matern52(lengthscale=lengthscale),
       noise_std=0.0,
   )

A kernel's hyperparameters are its fields, and the output scale is the factor in
front. A number fixes one (``2.0 * hb.gp.Matern52(lengthscale=0.3)``); a
:py:class:`~hybrid_bo.gp.Param` (an initial value and bounds) is fitted to the data
in each iteration. Each method fits its own GP to its own data, so methods can share
one prior GP. Left out, a lengthscale is the ``Param`` above; without a factor there
is no output scale. A lengthscale has one entry per input unless ``ard=False``. There
is no product or sum of two kernels. The hyperparameters are in *scaled units*: the GP sees its
inputs scaled onto [0, 1] and its outputs scaled to zero mean and unit standard
deviation, so a lengthscale of 0.5 is half the range of the data.

``noise_std`` is the known standard deviation of the measurement noise, in the units
of the measured variable, the same number as the measurement's noise. It has no
default, here or in ``oracle_measure``: 0 is noise-free. With noise of 0.01 on the
measured state:

.. code-block:: python

   measure = hb.oracle_measure(system, oracle, noise_std=0.01)
   gp = hb.gp.GP(
       mean=hb.gp.Zero(),
       kernel=outputscale * hb.gp.Matern52(lengthscale=lengthscale),
       noise_std=0.01,
   )

A run saves its fitted GPs whole, so the iteration figures of a saved study need
nothing from the experiment.

The study and the command line
------------------------------

``study_settings`` (:py:class:`~hybrid_bo.studies.StudySettings`) are the study's
defaults, which the command-line flags override: the number of runs of each
configuration and the seed. Both are the same for every configuration, so the runs
are paired. Every random draw comes
from the seed, so the same seed gives the same study; ``None`` draws a fresh one, which
the saved study keeps.

.. literalinclude:: ../experiments/illustrative_example.py
   :language: python
   :start-at: # The study
   :end-before: # Run or plot

The file ends by passing everything it defines to ``experiments/cli.py``. The first
argument is the experiment's name, which names its results folder:

.. literalinclude:: ../experiments/illustrative_example.py
   :language: python
   :start-at: # Run or plot

For your own system, leave out ``optimum`` and ``true_curves``:

.. code-block:: python

   if __name__ == "__main__":
       cli.main("my_experiment", problem, measure, configurations, study_settings)

``true_curves=True`` draws the true curves in the iteration figures by running the
measurement on a grid, so use it only with an oracle.

Running
-------

.. code-block:: bash

   pixi run experiment my_experiment --n-runs 1   # a quick check first
   pixi run experiment my_experiment --plot
   pixi run plot my_experiment

The study is saved in ``results/my_experiment/<study id>/``; :doc:`reading_results`
explains what is in it. Results are git-ignored.

The command line is a convenience. In a script or notebook, run the study and save it
yourself (the library never writes files unless asked):

.. code-block:: python

   study = hb.study(configurations, problem, measure, study_settings, optimum=optimum)
   hb.save(study, "results/my_experiment/first-try")

or run one method once, from initial points drawn with
:py:func:`~hybrid_bo.sampling.initial_points`:

.. code-block:: python

   rng = np.random.default_rng(0)
   u0 = hb.initial_points(system.u, n=2, rng=rng)
   run = hb.hybrid_bo(problem, measure, u0, 5, rng)

When something goes wrong
-------------------------

- **No module named my_experiment**: the name after ``experiment`` must match a file
  in ``experiments/``, without ``.py``.
- **A run fails** with a :py:class:`~hybrid_bo.errors.HybridBOError` (for example the
  root-finding problem has no solution): the study logs it, keeps the run up to that
  point, and goes on. Check the equations and the start bounds.
- **ReconstructionWarning**: with an oracle, the measured variables don't pin the
  states down to the oracle's values (``g`` has several roots). Narrow the start
  bounds of the unmeasured states to the right branch, or measure other variables.
- **ReplayWarning** when plotting: an iteration figure's replay of hybrid BO's
  acquisition function found another root of the known equations at the trial
  point than the run's solution (or none within the bounds), so the acquisition
  panel may show another branch than the one the run optimized. The run itself is
  unaffected; the figure is still drawn, and its 2D acquisition panel notes the
  stored and replayed values.
- **Starts stopped at IPOPT's iteration limit** (a warning naming the NLP's
  objective, such as ``negative_log_marginal_likelihood`` for a GP fit): the fit or
  acquisition problem is hard to solve, and the run is slow. A GP fit that also warns
  that its **training covariance is nearly singular** usually has a ``jitter`` that
  is too small for clustered inputs, or badly scaled data.
- **nan in the results**: look for ``math`` functions in ``g`` or ``f``.
