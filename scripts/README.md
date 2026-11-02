# Scripts

Your own study scripts: quick experiments, sweeps and one-off checks that don't
belong in `experiments/`. Git ignores everything here except this README, so nothing
in this folder reaches the tests, the docs or the CHANGELOG. Commit a script you want
to keep to `experiments/` instead (see
[Writing an experiment](../docs/writing_an_experiment.rst)).

A script is an experiment like any other: it defines a problem, how it is measured
and the configurations to compare, and ends by passing them to `hb.cli.main`. Its
file name, without `.py`, names its studies. For example, `scripts/short_budget.py`
runs the illustrative example's methods with a smaller budget:

```python
"""The illustrative example's methods with 4 evaluations instead of 7."""

import sys
from dataclasses import replace
from pathlib import Path

import hybrid_bo as hb

sys.path.append(str(Path(__file__).parents[1] / "experiments"))
import illustrative_example as ex  # noqa: E402

configurations = {
    label: replace(configuration, n_evaluations=4)
    for label, configuration in ex.configurations.items()
}

if __name__ == "__main__":
    hb.cli.main(
        ex.problem,
        ex.measure,
        configurations,
        hb.StudySettings(n_runs=5, seed=0),
        optimum=ex.optimum,
        true_curves=True,
    )
```

Run it from the repo root in the `experiments` environment, which has matplotlib and
Polars; studies go to `results/short_budget/` in the folder you run it from:

```bash
pixi run -e experiments python scripts/short_budget.py --n-workers 4 --plot
pixi run -e experiments python scripts/short_budget.py plot [STUDY] --plot-runs all
pixi run -e experiments python scripts/short_budget.py resume STUDY --n-workers 4
pixi run -e experiments python scripts/short_budget.py --help   # every command and flag
```

The commands are those of `pixi run experiment` (see
[Running large studies](../docs/running_large_studies.rst)). Each task runs the
script again, so keep everything the study needs in the script or what it imports,
and don't rename it while a study of it is unfinished: the name is the study's. You
can move it, though: `resume` checks what the script passes to `hb.cli.main`, not
where it is.
