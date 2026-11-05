Writing an experiment
=====================

An experiment is one Python script: a problem, how the system is measured, and the
methods to compare. It runs on its own, with nothing around it. This page builds one
block by block, quoting ``tests/illustrative_example.py``. The quickest start is
to copy that file, change each block as described below, and run it:

.. code-block:: bash

   cp tests/illustrative_example.py my_experiment.py
   python my_experiment.py first-try

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

.. literalinclude:: ../tests/illustrative_example.py
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

.. literalinclude:: ../tests/illustrative_example.py
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
:py:func:`~hybrid_bo.oracle.oracle_measure` simulates the system from ``g`` and
the oracle together. The methods only see the measured variables, with normal noise
of the standard deviation you state (``noise_std``, 0 for none); a run also keeps the
true (*oracle*) values, so the methods can be scored.

.. literalinclude:: ../tests/illustrative_example.py
   :language: python
   :start-at: # Measurement:
   :end-before: # Methods:

The *oracle optimum* is the true minimum of the problem, which regret is measured from.
State it: nothing searches for it, since a search can miss it. Find it once with
:py:func:`~hybrid_bo.oracle.oracle_optimum` (with many starts), check it, and paste
the value as above. :py:func:`~hybrid_bo.dataset.check_oracle_optimum` warns if a run
evaluates a point below it.

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
repr instead, as the oracle simulation is. Without an oracle the library knows only what was measured: there is no oracle
optimum, so no regret. Use :py:func:`~hybrid_bo.plots.convergence` instead, which draws
the incumbent, the lowest ``f`` a run measured.

The measurement is called once per evaluation, so its cost sets the cost of a run.

Methods
-------

A *method* is how the problem is solved. Each is a function you call directly, and a
script usually wraps each call in a small function of its own:

.. literalinclude:: ../tests/illustrative_example.py
   :language: python
   :start-at: # Every method's budget
   :end-before: def hybrid_bo_lin

The prior GP is defined once above them:

.. literalinclude:: ../tests/illustrative_example.py
   :language: python
   :start-at: # The prior GP
   :end-before: # Every method's budget

The methods take the problem and the measurement, then keyword arguments:

- :py:func:`~hybrid_bo.bo.hybrid_bo`: the prior GP (``gp``; see below), the acquisition function (:py:class:`~hybrid_bo.acquisition.EI`, or
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
- :py:func:`~hybrid_bo.bo.standard_bo`: BO with a GP of the objective, as the usual
  comparison. Its GP and acquisition function are required too.
- :py:func:`~hybrid_bo.baselines.latin_hypercube_sampling` and
  :py:func:`~hybrid_bo.baselines.uniform_sampling`, which sample without a model.

Initial points and the budget
-----------------------------

Every method takes ``n_iterations`` (design points for the baselines) and either
``n_initial``, how many initial points to draw from ``rng``, or ``initial_points``, an
array with a row per point, to start from points of your own. They are evaluated with
the measurement.

The same seed gives every method the same initial points. A BO method doesn't depend on
how many evaluations it has left, so with the same seed a run with a larger budget
passes through the same first evaluations as one with a smaller budget: the longest run
gives the regret of every smaller budget. The baselines draw their whole design at the
start, so Latin hypercube sampling doesn't have this property, and uniform sampling has
it only without measurement noise.

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

A run saves its fitted GPs whole, so the iteration figures of a saved run need nothing
from the experiment's methods.

Running
-------

The end of the file repeats each method over seeds, then saves and plots:

.. literalinclude:: ../tests/illustrative_example.py
   :language: python
   :start-at: # Every method's runs from the same seeds

The script takes the study's name, ``python my_experiment.py first-try``, so versions of
an experiment sit side by side. It is plain Python, so change it as you like. To run in parallel, ``joblib`` works:

.. code-block:: python

   from joblib import Parallel, delayed

   runs = Parallel(n_jobs=4)(delayed(hybrid_bo)(np.random.default_rng(s)) for s in range(10))

To keep finished runs through a long study, save each with
:py:func:`~hybrid_bo.saving.save_run` as it finishes, and skip the files that exist.
:doc:`reading_results` explains what is saved.

Optimizing data you already have
--------------------------------

With measurements of your own, you can get an optimum with no ``measure`` and no
optimization loop. :py:func:`~hybrid_bo.evaluation.reconstruct` solves the known equations
for the unmeasured states, :py:func:`~hybrid_bo.hybrid_model.fit_surrogate` fits the prior
GP to each unknown state, and :py:func:`~hybrid_bo.hybrid_model.optimize_hybrid_model`
solves the hybrid model once with each GP replaced by its mean:

.. code-block:: python

   rng = np.random.default_rng(0)
   dataset = hb.reconstruct(problem, u=u, measured=measured, rng=rng)
   fitted = hb.fit_surrogate(system, dataset, gp=gp, rng=rng)
   solution = hb.optimize_hybrid_model(problem, fitted, rng, n_starts=30)
   print(solution.u, solution.x, solution.y, solution.f)

The measurements have a row per point: ``u`` of shape ``(n, N)`` and ``measured``
``(n, P)``, its columns in the order of ``system.measured``. A ``u`` or a measured variable
outside its bounds is kept, with an ``OutOfBoundsWarning``. If some rows have no root or
hold a NaN, a ``RootFindingError`` names them all and its ``dataset`` holds the others, so
you can carry on without them. The GP's ``noise_std`` applies to all the data. The search is local from
each start, so raise ``n_starts`` if ``f`` has several minima.

.. warning::

   There is no uncertainty in this problem, so it trusts the GP's mean even far from the
   data, and the optimum can sit where the model is only extrapolating. Check the
   solution against your data, or measure it. To choose where to measure next, use
   hybrid BO, whose acquisition function accounts for what the GP doesn't know.

When something goes wrong
-------------------------

- **A run fails** with a :py:class:`~hybrid_bo.errors.HybridBOError` (for example the
  root-finding problem has no solution, or a measurement returned NaN or an infinity):
  it carries the run up to that point as ``error.run``. Check the equations, the start
  bounds and the measurement.
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
