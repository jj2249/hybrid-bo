Getting started
===============

This page runs hybrid BO and standard BO on the illustrative example of the paper, in
a plain script. The next pages explain how to state your own problem and how to read
the results.

Install
-------

The library isn't on PyPI yet, so install a release from the repository, with Python
3.12 or later and git:

.. code-block:: bash

   pip install "hybrid-bo[plots] @ git+https://github.com/jj2249/hybrid-bo.git@v0.6.0"

``plots`` adds matplotlib, for the figures. The tag at the end fixes the version, so the
library can't change under your work. To work on the repository, use
`pixi <https://pixi.sh>`_, which installs Python, the library and every dependency into
the project folder: fork and clone the repository, and the first ``pixi run`` installs
the environment it needs.

Run a script
------------

The illustrative example has one decision variable ``u``, the known equation
``x + exp(x) = y`` and the unknown equation ``y = sin(u)``, which the methods have to
learn. A script states the problem, how it is measured and the prior GP, then calls a
method:

.. code-block:: python

   import casadi as cas
   import numpy as np

   import hybrid_bo as hb

   # Problem: what we know about the physics, and what we want
   u = hb.Var("u", lower=-2.0, upper=2.0)
   x = hb.Var("x", start=(-2.5, 1.0))
   y = hb.Var("y", start=(-2.0, 2.0))
   system = hb.System(
       u=[u],
       x=[x],
       y=[y],
       g=lambda v: [v.x + cas.exp(v.x) - v.y],  # 0 = g(x, y, u)
       h_inputs=[u],  # y = h(u), which the GP learns
       measured=[y],
   )
   problem = hb.Problem(system, f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4))

   # Measurement: an oracle plays the real system
   measure = hb.oracle_measure(system, lambda v: [cas.sin(v.u)], noise_std=0.0)

   # The prior GP, which each method fits to its own data
   gp = hb.gp.GP(
       mean=hb.gp.Zero(),
       kernel=hb.gp.Param(1.0, bounds=(0.1, 10.0))
       * hb.gp.Matern52(lengthscale=hb.gp.Param(1.0, bounds=(1e-3, 1e2))),
       noise_std=0.0,
   )
   # Run both from the same seed, so from the same initial points
   hybrid = hb.hybrid_bo(
       problem,
       measure,
       n_initial=2,
       n_iterations=5,
       rng=np.random.default_rng(0),
       gp=gp,
       acquisition=hb.EI(),
       approximation=hb.SAA(n_scenarios=25, unknown_states_as_variables=False),
   )
   standard = hb.standard_bo(
       problem,
       measure,
       n_initial=2,
       n_iterations=5,
       rng=np.random.default_rng(0),
       gp=gp,
       acquisition=hb.EI(),
   )

   # Results: a run is plain data
   for run in (hybrid, standard):
       print(f"{run.method}: best f = {hb.incumbents(run.dataset)[-1]:.4f}")

   # A figure of the last iteration
   figure = hb.plots.iteration(hybrid, problem, 4, measure=measure)
   figure.savefig("hybrid_bo_iteration.png")

It takes a few seconds. Each method spends 2 initial points and 5 iterations, and the
true minimum is about -50.46. The figure shows the state before the last trial point:
what the GP has learned, the objective under the scenarios and the acquisition function.

Compare methods
---------------

One run says little, since a method's result depends on its initial points. To compare
methods, repeat them over seeds in a loop, and keep a list of runs per label:

.. code-block:: python

   import matplotlib.pyplot as plt

   def hybrid(seed):
       return hb.hybrid_bo(...)  # as above, with rng=np.random.default_rng(seed)

   def standard(seed):
       return hb.standard_bo(...)

   runs = {
       "hybrid_bo": [hybrid(seed) for seed in range(10)],
       "standard_bo": [standard(seed) for seed in range(10)],
   }
   hb.save_runs(runs, "results/first-try")

   figure, ax = plt.subplots()
   hb.plots.regret(runs, -50.4608, ax)
   figure.savefig("regret.png")

``tests/illustrative_example.py`` does this in full. :doc:`reading_results` explains what is
saved.

Build these docs
----------------

.. code-block:: bash

   pixi run docs

and open ``docs/_build/html/index.html``. Next, :doc:`writing_an_experiment`.
