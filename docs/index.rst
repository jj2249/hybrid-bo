hybrid-bo
=========

Bayesian optimization of partially known systems with hybrid models
(`arXiv 2603.11199 <https://arxiv.org/abs/2603.11199>`_): the known equations of a system
are used as they are, and a Gaussian process learns the unknown ones.

The library fits GPs with `CasADi <https://web.casadi.org/>`_ and solves one
iteration of hybrid or standard BO. You write the loop and the measurement, whether it runs
a plant, an external simulator or an oracle of your own:

.. code-block:: python

   import hybrid_bo as hb

   saa = hb.SAA(rng.standard_normal((25, 1)))  # once, before the loop
   for _ in range(10):
       surrogate = [gp.fit(data, inputs=[u], output=y, rng=rng)]
       solution = nlp.solve(
           saa,
           rng,
           surrogate=surrogate,
           objective=hb.obj.SoftEI(data.incumbents[-1], temperature=0.05),
       )
       new = hb.Dataset.from_measurements(
           nlp, u=solution.u, y=measure(solution.u), rng=rng
       )
       data = data.extend(new)

The `README <https://github.com/jj2249/hybrid-bo#readme>`_ has a complete example, with
standard BO and the baselines. The API reference covers the rest.

.. toctree::
   :hidden:

   API reference <autoapi/index>
