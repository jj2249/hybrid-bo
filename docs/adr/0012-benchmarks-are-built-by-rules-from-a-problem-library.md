# Benchmarks are built by rules from an external problem library, never copied

**A benchmark experiment taken from an external problem library is built by rules from
the library's own code, one experiment per problem name, and the library's problems
are never copied or transcribed.** `benchmarks/s2mpj.py` is the first: `python -m
s2mpj HS27` (`pixi run benchmark s2mpj HS27`) loads HS27 from S2MPJ, the Python
version of the CUTEst problems, and turns it into a hybrid problem. It traces
S2MPJ's own element and group functions with CasADi symbols (ADR 0002), so the known
equations and the oracle are S2MPJ's code, and it applies the same rules to every
problem: which constraint's element sum is hidden as the unknown equation, which
variables are mechanistic states, the box, the stated oracle optimum, the budget and
the settings. Every build compares the traced functions with S2MPJ's float
evaluation, since a symbol converted to a float is NaN and nothing raises. A problem
outside the rules raises an error that names the reason (#181).

This refines ADR 0006. A person still writes the module, and everything the study
depends on is in it, but the module builds one experiment for each problem name it
is given rather than being one experiment. The name is the script's argument: `hb.cli`
passes it to every task process (`python s2mpj.py HS27 run-task ...`), and a study
of it is saved under `results/s2mpj/HS27/`. Benchmarks live in `benchmarks/`, apart
from the experiments people write in `experiments/`, with a pixi environment of their
own. S2MPJ is fetched there only, at a pinned commit, so the library, the experiments
and the tests never need it (the tests that read it skip without it).

## Considered Options

- **A module per problem, transcribing its equations**: what the other benchmark
  experiments do (the Rosenbrock and Langermann functions). For a suite of 30 to 50
  problems it is a lot of copying, each copy a chance of a typo, and it would copy
  CUTEst's problems into this repository.
- **Wrapping S2MPJ's float functions as a `cas.Callback`**: needs no patch, but needs
  MX (#90), and each constraint would be one black box, so no part of it could be
  hidden as the unknown equation.
- **Installing S2MPJ as a dependency**: it isn't on PyPI or conda-forge, and the
  repository is about 150 MB. The benchmarks fetch a sparse checkout of the two paths
  they read.

## Consequences

The rules, not a person, decide each hybrid problem, so a rule that suits one problem
badly suits it badly in every study; the rules are in one module, where a change is
visible, and the task list and artifact record the S2MPJ commit and the split. A
problem the rules can't build is refused rather than built differently.
