# Parallel studies are processes over files, in the experiment CLI

**Parallelism and checkpointing live in the experiment CLI, as one process per task
that saves its own files, not in the library.** A study's runs are its tasks
(`hb.study_tasks`). `hb.study` runs them one after another in the caller's process
(`hb.run_task`) and assembles the study (`hb.assemble`). The experiment CLI instead
runs each task as `python -m <experiment> run-task STUDY --task-index K`, a new
process that rebuilds the experiment from its module, saves its run file as soon as
the run finishes, and writes its own per-run log. The study is assembled from the run
files when every task is done. The files on disk are the checkpoint: `resume` runs only
the tasks without a saved run.

The process boundary does the work a pool needed plumbing for. A task logs to its own
files and console, so no records cross processes. Its warnings are logged where they
are raised. BLAS reads its thread variables from the task's environment. Stopping a
task is `terminate()`, which works while CasADi swallows Ctrl-C inside IPOPT. The CLI's
launcher uses only public `subprocess` calls, so it runs the same on Windows and
macOS, and `run-task` is an ordinary command that a SLURM array or GNU parallel can
run too. The library still never writes files on its own: every file is written by the
CLI through explicit saving calls (`hb.save_run`, `hb.saving.save_artifact`).

## Considered Options

- **A process pool in `hb.study`** (`n_workers`, #145; what this replaces): kept
  everything in one call, but results, log records, warnings and Ctrl-C all had to come
  back through the parent. That took about 200 lines: a pickling check, a log queue
  with a listener thread, warnings replayed in the parent, BLAS variables set while
  the pool started workers, and stopping workers through the pool's private
  `_processes`. It saved nothing until every run finished, and couldn't spread a study
  over a cluster.
- **GNU parallel as the local launcher**: needs Perl and a POSIX shell, so it doesn't
  run natively on Windows. It still works as an external launcher, since `run-task` is
  an ordinary command.
- **joblib/loky, submitit, Ray or Dask**: none removes the handling of logs, warnings
  and Ctrl-C the way separate processes over files do, and submitit's pickled results
  don't give resumable run files.

## Consequences

- Each task re-imports CasADi and scipy and rebuilds the problem, about 1–2 s per
  task: noticeable for the illustrative example's sub-second runs, negligible for
  Langermann's (about 90 s each).
- A task rebuilds the experiment from its module, so nothing is pickled: lambdas and
  nested functions work. In return, `cli.main` must be called from the experiment's
  module run as `__main__`, and a resumed or scheduled task refuses to run unless the
  experiment, the flags and the code version match the study's task list.
- A run killed partway starts again from scratch; checkpointing within a run is out of
  scope.
- In-process `hb.study` is serial again and stays the easiest form to debug, but it
  doesn't stop on Ctrl-C while IPOPT runs (#155); the CLI does.
