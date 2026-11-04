# Studies run with Benchopt, not the library

**The library is the optimization: problems, methods, runs and iteration figures. Running
many repetitions of many methods is [Benchopt](https://benchopt.github.io)'s job.**
`experiments/` is a Benchopt benchmark, and `hb.cli`, `hb.studies` and `hb.views` are
gone, with `hb.plots.convergence`, `hb.plots.regret` and saving a whole study.
Supersedes [ADR 0004](0004-studies-are-saved-as-json.md) and
[ADR 0010](0010-parallel-studies-are-processes-over-files.md).

More than half of the code wasn't the optimization, and a study runner has no place in
a package meant to sit beside sklearn or PyMC. It was also a second product to maintain:
task lists, resuming, a process launcher, per-run logs. Benchopt, a maintained benchmark
framework, gives repetitions, parallel runs, resuming, an interactive report and
comparisons across languages. An adapter of about 270 lines runs the experiments under
it, and only that adapter uses Benchopt's classes, so the library stays functional.

A script calls `hb.hybrid_bo`, `hb.standard_bo` and the baselines directly. A run is
still pure data ([ADR 0003](0003-runs-are-pure-data.md)) and saves as JSON with
`hb.save_run`; that is what a study's run files always were. The adapter saves every
run this way, so the iteration figures and everything else a run holds survive beside
Benchopt's table of results.

## Considered Options

- **Keep the runner, in a package of its own**: separates it from the library and could
  serve other libraries, but leaves us maintaining it, competing with Benchopt.
- **Snakemake, or Hydra with submitit**: closer to the runner's files-as-state design,
  but neither knows about solvers or repetitions, so more of the work stays with us.
- **Sacred, SEML, MLflow**: track runs rather than launch them, or need MongoDB or a
  server.

## Consequences

- **The hash of the code has to stay.** Benchopt's cache only sees its own classes, so
  it returns old results for a changed experiment or library. A hash of both is a
  parameter of the adapter's dataset, so it is part of the cache's key and it records
  which code produced the results. It has one value: Benchopt pairs a selected module
  with each value of another parameter. Editing one experiment reruns them all.
- A changed experiment reruns instead of being refused, as `resume` refused it.
- A run failing with an unexpected exception prints a traceback, is missing from the
  results and leaves the exit code at 0. Benchopt has no option for this. A
  `HybridBOError` keeps the partial run, which the `summary` view counts as incomplete.
- Benchopt's curves need a point per solver call, and a run is one call, so `regret` is
  a view of ours. Its traces have no vertical band: it draws the median alone.
- Per-run logs and `--require-clean` are gone, and so is `git` in a run's artifact: a
  git lookup from an installed package answers for the user's repository (#198).
- Studies saved by earlier versions don't load: read them with version 0.5, or rerun.
- SLURM support is Benchopt's and untested here.
