Getting started
===============

This page runs the illustrative example of the paper and looks at what it produced.
The next pages explain how to write your own experiment and how to read the results.

Install
-------

The project is managed with `pixi <https://pixi.sh>`_, which installs Python, the
library and every dependency into the project folder. Install pixi, then fork the
repository on GitHub and clone your fork:

.. code-block:: bash

   git clone https://github.com/<you>/hybrid-bo.git
   cd hybrid-bo

There is no separate install step: the first ``pixi run`` installs the environment it
needs.

Run the illustrative experiment
-------------------------------

An *experiment* is one Python file in ``experiments/``: a problem, how it is measured,
and the methods to compare. ``experiments/illustrative_example.py`` is the
illustrative example of the paper: one decision variable ``u``, the known equation
``x + exp(x) = y`` and the unknown equation ``y = sin(u)``, which the methods have to
learn. Run every method on it three times and plot the results:

.. code-block:: bash

   pixi run experiment illustrative_example --n-runs 3 --plot

It takes about a minute. It runs a *study*: every method (hybrid BO, standard BO, Latin hypercube sampling and
uniform sampling) ``n_runs`` times, each run 7 evaluations of which the first 2 are
initial points, run ``i`` of every method from the same initial points. ``pixi run experiment illustrative_example --help`` lists the flags:

- ``--method NAME`` runs only the configuration with that label (repeat it for
  several; default all);
- ``--n-runs`` and ``--seed`` override the experiment's study settings. Each
  configuration has its own budget, set in the experiment, so no flag changes the
  number of iterations;
- ``--n-workers 4`` runs the runs in 4 processes at once, with the same results
  (default 1: one after another, each still in its own process);
- ``--name`` names the study (default the current UTC time); a name already taken
  is refused before anything runs;
- ``--plot`` draws the figures straight after, and ``--plot-runs 0 3`` (or ``all``)
  chooses the runs that get a figure per iteration (default run 0 of each method);
- ``--require-clean`` refuses to run with uncommitted or untracked files (without it,
  they get a warning), since the study wouldn't be reproducible.

Each run is saved as soon as it finishes. Ctrl-C, or a crash, stops the study but
keeps its finished runs, and prints the command that finishes it:

.. code-block:: bash

   pixi run experiment illustrative_example resume STUDY [--n-workers 4]

which runs only the runs that are missing. It refuses if the experiment, the seed or
the code changed since the study started, so a resumed study equals one run in one go.

:doc:`running_large_studies` covers long studies: many runs on one machine, and
SLURM clusters.

Look at the study
-----------------

When it finishes, the command prints a summary with one row per method: the median and
interquartile range (IQR), over the runs, of the final ``incumbent`` (the lowest
objective measured), ``oracle_incumbent`` (the objective there without measurement
noise) and ``regret`` (how far that is above the oracle optimum, the problem's
minimum). Lower is better.

The study is saved in ``results/illustrative_example/<study id>/``:

.. code-block:: text

   artifact.json       what the study ran and how: settings, problem, methods
   tasks.json          the task list: what the study runs, written before it starts
   runs/<method>/      one JSON file per run: every evaluated point, the GPs, ...
   summary.csv         the summary table printed above
   study.log           the study's progress: each run and iteration, times, warnings
   logs/<method>/      the same, one log per run: run-000.log, ...
   figures/            convergence.png, regret.png and a figure per iteration of run 0

Open ``figures/regret.png``: the median regret of each method over the evaluations,
with the interquartile range across runs shaded. Hybrid BO's median regret should fall
by orders of magnitude within the five iterations (the axis is logarithmic), while
the other methods stay well above it.
To draw the figures again later, or for an older study:

.. code-block:: bash

   pixi run plot illustrative_example            # the newest study
   pixi run plot illustrative_example <study id>
   pixi run experiment illustrative_example plot <study id> --plot-runs all

:doc:`reading_results` explains every file and figure, and how to load a study in
Python.

Build these docs
----------------

.. code-block:: bash

   pixi run docs

and open ``docs/_build/html/index.html``. Next, :doc:`writing_an_experiment`.
