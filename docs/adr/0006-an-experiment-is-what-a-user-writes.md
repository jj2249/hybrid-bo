# An experiment is what a user writes; a measurement is one call to the system

The module a user writes (a problem, how it is measured, and the methods to compare) is
an **experiment**: "the flash unit experiment", "a benchmark experiment". The per-point
callable that ADR 0001 called the experiment is now the **measurement**:
`measure(u, rng) -> measured variables`, with `hb.oracle_measure` for benchmark
experiments. ADR 0001's split is unchanged; only the names move.

A **benchmark experiment** is one whose unknown equations we know (the oracle) but hide
from the methods. The folder of experiments is `experiments/` and the shipped ones serve
as examples.

## Considered Options

- **"Example"** for the user's module: wrong once the library runs on industrial data;
  a plant engineer's script is not an example of anything.
- **"Case"**: fits benchmarks and applications, but carries no scientific meaning and
  was already on the glossary's avoid list for Problem.
- **"Trial"** or **"Plant"** for the per-point callable: `hybrid_bo(problem, trial, ...)`
  reads as a single trial, and a plant collides with the System. "Measurement" matches
  the measured variables and `OracleMeasurement` it returns.

## Consequences

The rename lands before students fork the repo; after that it would break their code.
A run's record (its artifact since #123) now holds the measurement under `measure`. Records are text, so studies
saved with the old `experiment` key still load.
