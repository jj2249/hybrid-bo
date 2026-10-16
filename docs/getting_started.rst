Getting started
===============

Import the package as ``hb``. Its top level holds what a typical script uses; the rest
is in its module (``hb.gp.fit_gp``, ``hb.solvers.multistart``), and the
:doc:`API reference <autoapi/index>` covers every function.

.. code-block:: python

   import hybrid_bo as hb

Define a problem
----------------

- :py:class:`~hybrid_bo.problem.Var`: a variable with its bounds and start bounds.
- :py:class:`~hybrid_bo.problem.System`: the variables (``u``, ``x``, ``y``), the known
  equations ``g``, the inputs of the unknown equations and the measured variables.
- :py:class:`~hybrid_bo.problem.Problem`: a system and the objective ``f``.
- :py:data:`~hybrid_bo.evaluation.Measure`: a function ``(u, rng) -> measured``,
  the real system. :py:func:`~hybrid_bo.evaluation.oracle_measure` simulates one
  from known unknown equations, for benchmark experiments.

Run a method
------------

- :py:func:`~hybrid_bo.sampling.initial_points`: the points every method starts from.
- :py:func:`~hybrid_bo.bo.hybrid_bo` with
  :py:class:`~hybrid_bo.bo.HybridBOSettings`, and :py:func:`~hybrid_bo.bo.standard_bo`
  with :py:class:`~hybrid_bo.bo.StandardBOSettings`. The settings choose the
  acquisition function, :py:class:`~hybrid_bo.acquisition.EI` or
  :py:class:`~hybrid_bo.acquisition.LCB`.
- The baselines :py:func:`~hybrid_bo.baselines.latin_hypercube_sampling` and
  :py:func:`~hybrid_bo.baselines.uniform_sampling`.

A method raises a :py:class:`~hybrid_bo.errors.HybridBOError` when it fails; the error
carries the partial run.

Look at the results
-------------------

- :py:class:`~hybrid_bo.run.Run`: what a method returns, plain data.
- :py:class:`~hybrid_bo.history.History`: every evaluated point (``run.history``).
- :py:func:`~hybrid_bo.history.incumbents` and
  :py:func:`~hybrid_bo.history.oracle_incumbents`: the best point after each
  evaluation, and the oracle objective there (the benchmark metric).
- :py:func:`~hybrid_bo.views.evaluations`, :py:func:`~hybrid_bo.views.summary` and
  :py:func:`~hybrid_bo.views.hyperparameters`: Polars data frames of a study, one row
  per evaluation, per method or per fitted hyperparameter (Polars is optional).
- :py:mod:`hybrid_bo.plots`: convergence and per-iteration figures of finished runs.

The :doc:`walkthrough` puts these together on the illustrative example.
