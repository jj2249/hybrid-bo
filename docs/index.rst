hybrid-bo
=========

Bayesian optimization of partially known systems with hybrid models
(`arXiv 2603.11199 <https://arxiv.org/abs/2603.11199>`_): the known equations of a system
are used as they are, and a Gaussian process learns the unknown ones.

The library fits GPs with `CasADi <https://web.casadi.org/>`_ and solves one iteration of
hybrid or standard BO. You write the loop and the measurement, a function of ``(u, rng)``
that returns the measured variables as an array, whether it runs a plant, an external
simulator or an oracle of your own:

.. code-block:: python

   import hybrid_bo as hb

   scenarios = hb.saa_scenarios(25, system.n_y, rng)  # once, before the loop
   for _ in range(10):
       fitted = hb.fit_surrogate(system, dataset, gp=gp, rng=rng)
       solution = hb.hybrid_trial_point(
           problem,
           fitted,
           scenarios,
           hb.incumbents(dataset)[-1],
           rng,
           approximation=hb.SAA(),
       )
       measured = measure(solution.u, rng)
       new = hb.reconstruct_states(
           problem, u=solution.u[None], measured=measured[None], rng=rng
       )
       dataset = hb.extend(dataset, new)

The `README <https://github.com/jj2249/hybrid-bo#readme>`_ has a complete example, with
standard BO and the baselines. The API reference covers the rest.

.. toctree::
   :hidden:

   API reference <autoapi/index>
