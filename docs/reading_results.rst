Reading results
===============

A *study* is every run of every method of one experiment. ``pixi run experiment``
saves it, and ``pixi run plot`` draws it, without running anything again. This page
explains the saved files, the figures, and how to work with a study in Python.

The study folder
----------------

``results/<experiment>/<study id>/`` holds:

- ``artifact.json``: the study's settings (with the seed) and its artifact, which says
  what it ran and how: the problem's variables and equations, the measurement, each
  method with its settings, the oracle optimum if stated, and the package and code
  versions. The code version ends in ``-dirty`` or ``-untracked`` when the study ran
  with uncommitted or untracked files, and then the study isn't reproducible. Studies
  saved before this was named have ``study.json`` instead, and still load.
- ``runs/<method>/run-000.json``: one file per run.
- ``summary.csv``: one row per method, the median and interquartile range (IQR) over
  its runs of the final values, and how many runs failed.
- ``study.log``: the study log, the study's progress as it ran: each run and
  iteration with its time and the incumbent, the time of each GP fit and multistart,
  and every warning. Each line a run logged starts with the run, as
  ``hybrid_bo/run-003``, which tells apart the lines of runs that ran in parallel
  (``--n-workers``). While the study runs it is ``<study id>.log`` beside the study
  folders; a study that doesn't finish leaves it there.
- ``logs/<method>/run-000.log``: one log per run, the study log's lines of that run.
  While the study runs they are in ``<study id>.logs/``.
- ``figures/``: written by ``plot`` (or ``--plot``).

Every *evaluation* runs the measurement at a point ``u``, recovers the unmeasured
states and computes ``f``. A run evaluates its *initial points*, then one trial point
per *iteration*. The values the summary and the figures report, after each evaluation:

- ``incumbent``: the lowest ``f`` the run has measured so far.
- ``oracle_incumbent`` (with an oracle): the objective at the incumbent's point,
  computed from the oracle values rather than the noisy measurement. This is the fair
  score of a method.
- ``regret`` (with an oracle optimum): the oracle incumbent minus the oracle optimum.

The figures
-----------

- ``convergence.png``: per method, the median oracle incumbent over the evaluations
  (the incumbent, without an oracle), with the IQR across runs shaded. The dotted line
  marks the end of the initial points. Evaluations, not iterations, are counted, since
  initial points cost a measurement too.
- ``regret.png``: the median regret, on a log scale, as in the paper. Only with an
  oracle optimum.
- ``<method>/run-000/iteration-00.png``: one figure per iteration of a run, the
  state before its trial point. Only for problems with one or two decision variables,
  and by default only for run 0 of each method: ``--plot-runs 0 3`` chooses the runs,
  ``--plot-runs all`` draws every one.

Hybrid BO's iteration figure has three panels, one per step of the iteration: the GP of
the unknown state over ``u``; the objective under the *scenarios* (draws from the GP,
each propagated through ``g`` to ``f``); and the acquisition function, whose minimum is
the next trial point. Standard BO's figure has the GP of ``f`` and its acquisition
function; a baseline's has the points only. Each shows the points so far, the
incumbent and the trial point, and, with ``true_curves=True``, the true curves.

With two decision variables each panel is a map over ``u``: hybrid BO has a row for
the unknown state (oracle, GP mean, GP std) and a row for ``f`` (oracle, the mean and
std over the scenarios, and EI); standard BO the row for ``f`` with its GP. The oracle
panels, with ``true_curves=True``, share the color scale of the mean beside them. EI
is drawn as log10 EI, with the points where it is 0, SAA-EI's flat regions, in gray.
Its color scale covers at most 10 decades below the largest EI; smaller EI, which
SoftSAA-EI at a low temperature has over most of the map, takes the lowest color.

The study in Python
-------------------

:py:func:`~hybrid_bo.saving.load` reads a saved study back, without the code that
produced it:

.. code-block:: python

   import hybrid_bo as hb

   study = hb.load("results/illustrative_example/<study id>")

A :py:class:`~hybrid_bo.studies.Study` holds ``runs``, ``settings``
(:py:class:`~hybrid_bo.studies.StudySettings`), ``name`` and ``artifact``; each run
records its configuration's budget as ``n_evaluations``. Three
functions turn it into `Polars <https://pola.rs>`_ data frames:

- :py:func:`~hybrid_bo.views.evaluations`: one row per evaluation of every run, with
  the method, run, evaluation and iteration, every variable (``u_<name>``,
  ``x_<name>``, ``y_<name>``) and ``f``, their oracle values, the incumbents, the
  regret and the acquisition value.
- :py:func:`~hybrid_bo.views.summary`: one row per configuration, as in
  ``summary.csv``; a run failed if it stopped short of its ``n_evaluations``.
- :py:func:`~hybrid_bo.views.hyperparameters`: the fitted GP hyperparameters of every
  iteration, for checking the GP (a lengthscale stuck at a bound, say); ``output``
  names the variable each GP models (an unknown state, or ``f`` for standard BO).

.. code-block:: python

   import polars as pl

   rows = hb.evaluations(study)
   best = rows.filter(pl.col("method") == "hybrid_bo").sort("f").head(1)

The figures are functions of the study, to draw into your own figures or with
different options: :py:func:`~hybrid_bo.plots.convergence`,
:py:func:`~hybrid_bo.plots.regret` and :py:func:`~hybrid_bo.plots.iteration` (which
also needs the problem, from your experiment). They use the repo's matplotlib style,
``hb.plots.STYLE``.

.. code-block:: python

   import matplotlib.pyplot as plt

   ax = hb.plots.convergence(study)
   ax.set_title("Illustrative example")
   plt.savefig("convergence.pdf")

One run
~~~~~~~

Each :py:class:`~hybrid_bo.run.Run` in ``study.runs`` is plain data:

- ``run.method`` and ``run.index``: its label and its position among that method's
  runs.
- ``run.history`` (:py:class:`~hybrid_bo.history.History`): every evaluated point in
  order, initial points first, as arrays ``u``, ``x``, ``y`` and ``f`` with one row
  per evaluation, and their oracle values (``None`` without an oracle).
  :py:func:`~hybrid_bo.history.incumbents` and
  :py:func:`~hybrid_bo.history.oracle_incumbents` give the incumbents after each
  evaluation.
- ``run.surrogates``: the fitted GPs (:py:class:`~hybrid_bo.gp.FittedGP`) of each
  iteration, a tuple of one per unknown state for hybrid BO and of one for standard BO;
  each is its prior GP with the fitted values in place of each ``Param``
  (``fitted.gp``), for :py:func:`~hybrid_bo.gp.predict`; and ``run.acquisitions``: the acquisition problem's solution at each
  trial point. Both are empty for the baselines.
- ``run.scenarios``: hybrid BO's scenario draws and weights.
- ``run.artifact``: how the run was produced, as text, including its seed.

A run that failed, with a :py:class:`~hybrid_bo.errors.HybridBOError`, is kept up to
the point where it stopped, so it is shorter than the others; ``summary`` counts it
under ``failed``.
