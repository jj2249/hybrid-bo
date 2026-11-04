# A script runs the methods; the library has no study runner

**The library is the optimization: problems, methods, runs, saving and figures. Repeating
a method over seeds is a loop in the script.** `hb.cli`, `hb.studies` and `hb.views` are
gone. An experiment is a plain script that calls `hb.hybrid_bo`, `hb.standard_bo` and the
baselines directly, then `hb.save_runs` and `hb.plots.regret` on a list of runs per
label. Supersedes [ADR 0004](0004-studies-are-saved-as-json.md) and
[ADR 0010](0010-parallel-studies-are-processes-over-files.md).

More than half of the code wasn't the optimization, and a study runner has no place in a
package meant to sit beside sklearn or PyMC. A student should be able to write a short
script and reach the hybrid logic through a few function calls, with nothing to learn
about how it is run.

The methods take keyword arguments, so a script reads as one call. `HybridBOSettings` and
`StandardBOSettings` stay as the record a run keeps for the iteration figures
(`Run.settings`), which the method builds itself. They were an argument so that
`functools.partial` and `Configuration` could carry them.

## Considered Options

- **Keep the runner, in a package of its own**: separates it from the library, but leaves
  us maintaining it.
- **Benchopt** (tried, #196): gives repetitions, parallel runs, resuming and a report. It
  runs your code, so an experiment has to be split into pieces it can call, and an
  adapter of about 190 lines has to connect them. Its cache also returns old results for
  a changed experiment, as it only sees its own classes. The plain script does the same
  work in about 15 lines, and stays a script.
- **A `run_<label>` convention, or one solver file per method**: variations on the same
  adapter.
- **Snakemake, or Hydra with submitit; Sacred, SEML, MLflow**: more machinery than the
  loop they would replace.

## Consequences

- Parallel runs are the script's: `joblib` works, with results identical to serial ones.
- Resuming is the script's too: save each run as it finishes and skip the files that
  exist. Delete those files after changing the experiment, or they are reused.
- A failed run stops a script that doesn't catch it, loudly. `error.run` holds the
  partial run.
- A run's artifact no longer records `git`: from an installed package git answers for the
  user's repository (#198).
- Studies saved by earlier versions don't load: read them with version 0.5, or rerun.
