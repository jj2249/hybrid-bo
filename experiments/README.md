# Experiments

Each file here is an experiment: a plain script with a problem, how it is measured, and
the methods to compare. It repeats each method over seeds, saves the runs to
`results/<name>/` and draws `regret.png`.

- `illustrative_example.py`: the illustrative example of the
  [paper](https://arxiv.org/abs/2603.11199), with an oracle.
- `illustrative_example_2d.py`: the illustrative example with two decision
  variables, y = sin(u1) cos(u2), with an oracle, comparing SAA-EI, Lin-EI,
  SoftSAA-EI, standard BO and Latin hypercube sampling.
- `rosenbrock.py`: the 2D Rosenbrock function as a composite function, with no known
  equations and two unknown states (adapted from Astudillo & Frazier,
  [arXiv 1906.01537](https://arxiv.org/abs/1906.01537), App. D.2), with an oracle,
  comparing SAA-EI, SoftSAA-EI, standard BO and Latin hypercube sampling.
- `langermann.py`: the Langermann function as a composite function, with no known
  equations and five unknown states (Astudillo & Frazier, App. D.1), with an oracle,
  comparing the same methods.

## Running

```bash
pixi run -e experiments python experiments/illustrative_example.py
```

`N_RUNS`, `N_INITIAL` and `N_ITERATIONS` are at the top of each file. `hb.save_runs`
refuses an existing folder, so move or delete `results/<name>/` to run one again.
