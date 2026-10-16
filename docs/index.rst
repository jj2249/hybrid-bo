hybrid-bo
=========

Bayesian optimization of partially known systems with hybrid models
(`arXiv 2603.11199 <https://arxiv.org/abs/2603.11199>`_): the known equations of a
system are used as they are, and a Gaussian process learns the unknown ones. The paper
explains the method; these pages explain how to use the library.

- :doc:`getting_started`: install, run the illustrative experiment and look at what it
  produced.
- :doc:`writing_an_experiment`: state your own problem and measurement, choose the
  methods and run them.
- :doc:`reading_results`: the saved study, the summary, the figures, and the study in
  Python.

Import the package as ``import hybrid_bo as hb``: the top level holds what a typical
experiment uses, the rest is in its module (``hb.gp.default_gp``). The
:doc:`API reference <autoapi/index>` covers every function.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   writing_an_experiment
   reading_results
