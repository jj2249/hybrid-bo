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
uniform sampling) ``n_runs`` times, run ``i`` of every method from the same initial
points. ``pixi run experiment illustrative_example --help`` lists the flags:

- ``--method NAME`` runs only that method (repeat it for several; default all);
- ``--n-runs``, ``--n-iterations`` and ``--seed`` override the experiment's study
  settings;
- ``--name`` names the study (default the current UTC time);
- ``--plot`` draws the figures straight after.

Look at the study
-----------------

When it finishes, the command prints a summary with one row per method: the median and
interquartile range (IQR), over the runs, of the final ``incumbent`` (the best
objective measured), ``oracle_incumbent`` (the true objective there) and ``regret``
(how far that is above the true optimum). Lower is better.

The study is saved in ``results/illustrative_example/<study id>/``:

.. code-block:: text

   study.json          the settings and the record of the problem and methods
   runs/<method>/      one JSON file per run: every evaluated point, the GPs, ...
   summary.csv         the summary table printed above
   figures/            convergence.png, regret.png and one figure per iteration

Open ``figures/regret.png``: the median regret of each method over the evaluations,
with the interquartile range across runs shaded. Hybrid BO's median regret should drop
to about zero within the five iterations, while the other methods stay well above it.
To draw the figures again later, or for an older study:

.. code-block:: bash

   pixi run plot illustrative_example            # the newest study
   pixi run plot illustrative_example <study id>

:doc:`reading_results` explains every file and figure, and how to load a study in
Python.

Build these docs
----------------

.. code-block:: bash

   pixi run docs

and open ``docs/_build/html/index.html``. Next, :doc:`writing_an_experiment`.
