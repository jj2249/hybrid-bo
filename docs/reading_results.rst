Reading results
===============

A method returns a run, which is plain data. To compare methods, keep a list of runs
per label, which the library saves and plots.

One run
-------

A :py:class:`~hybrid_bo.run.Run` holds:

- ``run.method``: the method's name.
- ``run.history`` (:py:class:`~hybrid_bo.history.History`): every evaluated point in
  order, initial points first, as arrays ``u``, ``x``, ``y`` and ``f`` with one row
  per evaluation, and their oracle values (``None`` without an oracle).
  :py:func:`~hybrid_bo.history.incumbents` and
  :py:func:`~hybrid_bo.history.oracle_incumbents` give the incumbents after each
  evaluation.
- ``run.surrogates``: the fitted GPs (:py:class:`~hybrid_bo.gp.FittedGP`) of each
  iteration, a tuple of one per unknown state for hybrid BO and of one for standard BO;
  each is its prior GP with the fitted values in place of each ``Param``
  (``fitted.gp``), for :py:func:`~hybrid_bo.gp.predict`; and ``run.acquisitions``: the
  acquisition problem's solution at each trial point. Both are empty for the baselines.
- ``run.scenarios``: hybrid BO's scenario draws and weights.
- ``run.settings``: the BO method's settings in full.
- ``run.artifact``: how the run was produced, as text, including its seed.

A method that fails raises a :py:class:`~hybrid_bo.errors.HybridBOError`, whose ``run``
holds the run up to the point where it stopped.

Saving a run
~~~~~~~~~~~~

:py:func:`~hybrid_bo.saving.save_run` writes a run as JSON, and
:py:func:`~hybrid_bo.saving.load_run` reads it back without the code that produced it:

.. code-block:: python

   hb.save_run(run, "hybrid_bo.json")
   run = hb.load_run("hybrid_bo.json")

The library never writes files unless asked.

Iteration figures
~~~~~~~~~~~~~~~~~

:py:func:`~hybrid_bo.plots.iteration` draws the state before a trial point: the GP, the
objective under the scenarios and the acquisition function, over one or two decision
variables. It reads the run, so a saved run draws without running it again, but it
needs the problem, from your script. Pass the measurement to draw the true curves
too, which runs it over a grid, so only for an oracle:

.. code-block:: python

   figure = hb.plots.iteration(run, problem, 4, measure=measure)
   figure.savefig("iteration-4.png")

It needs ``hybrid-bo[plots]``, and uses the repo's matplotlib style, ``hb.plots.STYLE``.
With more than one unknown state, hybrid BO's figures need ``gp=False``.

Many runs
---------

A list of runs per label is all the library needs:

.. code-block:: python

   runs = {"hybrid_bo": [...], "standard_bo": [...]}

:py:func:`~hybrid_bo.saving.save_runs` writes them to a new folder, as
``<label>/run-000.json``, and :py:func:`~hybrid_bo.saving.load_runs` reads them back:

.. code-block:: python

   hb.save_runs(runs, "results/first-try")
   runs = hb.load_runs("results/first-try")

:py:func:`~hybrid_bo.plots.regret` draws the median regret of each label over the
evaluations, with its interquartile range, and
:py:func:`~hybrid_bo.plots.convergence` the incumbent the same way. Each returns the
matplotlib axes, so adjust the figure as you like:

.. code-block:: python

   import matplotlib.pyplot as plt

   ax = hb.plots.regret(runs, optimum=-50.4608)
   ax.set_title("Illustrative example")
   plt.savefig("regret.pdf")

:py:func:`~hybrid_bo.history.check_oracle_optimum` warns with
:py:class:`~hybrid_bo.errors.OracleOptimumWarning` if a run evaluated a point below the
stated oracle optimum, which would make its regret wrong.
