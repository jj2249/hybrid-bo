Reading results
===============

A method returns a run, which is plain data. A study of many runs is Benchopt's, which
saves a table of results beside each run in full.

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
needs the problem, from your experiment. Pass the measurement to draw the true curves
too, which runs it over a grid, so only for an oracle:

.. code-block:: python

   figure = hb.plots.iteration(run, problem, 4, measure=measure)
   figure.savefig("iteration-4.png")

It needs ``hybrid-bo[plots]``, and uses the repo's matplotlib style, ``hb.plots.STYLE``.
With more than one unknown state, hybrid BO's figures need ``gp=False``.

A study
-------

``pixi run benchmark`` writes to ``experiments/outputs/``:

.. code-block:: text

   benchopt_run_<time>.parquet     the results: one row per evaluation of every run
   benchopt_run_<time>.html        the report, unless --no-html
   runs/<module>/<code>/<label>/   each run in full, as seed-<seed>.json

The report has Benchopt's plots and two of ours, ``regret`` (the median regret of each
method over the evaluations, with its interquartile range; lower is better) and
``summary``: each method's runs, complete runs, the quartiles of the final regret and
the runs below the stated oracle optimum, whose regret is wrong. **Check the runs in ``summary``.**
A run that stops with a ``HybridBOError`` counts as incomplete, and one that fails with
any other exception prints a traceback and is missing from the results, without
changing the exit code.

Read the results with pandas, or with a Parquet viewer in your editor:

.. code-block:: python

   from benchopt.results import read_results

   df = read_results("experiments/outputs/benchopt_run_<time>.parquet")
   last = df[df["objective_complete"] == 1.0].groupby(["solver_name", "idx_rep"]).last()
   last.groupby("solver_name")["objective_regret"].median()

Each row names its method (``solver_name``, ``p_solver_label``), its run (``idx_rep``)
and its experiment (``p_dataset_module``). Its scores are ``objective_evaluation``,
``objective_incumbent`` (the lowest ``f`` measured so far), ``objective_value`` (the
oracle incumbent: the objective without noise at that point), ``objective_regret`` (how
far that is above the oracle optimum), ``objective_complete`` and
``objective_below_optimum``.

``p_dataset_code`` is a hash of the experiments and the library, which names the
folder of the saved runs, so results say which code produced them.
