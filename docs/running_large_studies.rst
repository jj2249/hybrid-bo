Running large studies
=====================

A study's runs are its *tasks*: one task is one run of one configuration, so
``langermann`` (4 configurations) with ``--n-runs 50`` has 200 tasks. The command line
runs each task in its own process and saves its run as soon as it finishes, so a long
study can be spread over the cores of one machine or over a cluster, and a study that
stops partway loses only the runs in progress.

Check first
-----------

Run one configuration once before a long study, to see that it runs, saves and plots:

.. code-block:: bash

   pixi run experiment langermann --method hybrid_bo --n-runs 1 --name smoke --plot

``--help`` lists the configuration labels. Delete ``results/langermann/smoke``
afterwards, or the name stays taken.

On one machine
--------------

.. code-block:: bash

   pixi run experiment langermann --n-runs 50 --seed 1 --n-workers 8 \
       --name big-run --require-clean

``--n-workers 8`` keeps 8 tasks running, each with BLAS and OpenMP on one thread
(unless you set ``OMP_NUM_THREADS`` and the like yourself), so it uses about 8 cores;
more workers than cores only slows each down. The results are the same for any number
of workers. ``--require-clean`` refuses to start with uncommitted files, so the
study's artifact records the exact code version.

The console shows each run's progress, tagged with its run (``hybrid_bo/run-003``),
and each task as it starts and finishes ("37 of 200 done"). Ctrl-C, a crash, or a
laptop going to sleep stops the study, keeps every finished run, and prints the
command that finishes it:

.. code-block:: bash

   pixi run experiment langermann resume big-run --n-workers 8

``resume`` runs only the tasks without a saved run, possibly with another number of
workers or on another machine, then saves the study, which equals the study run in
one go. It takes the study's methods, runs and seed from its task list, and refuses
if the experiment or the code changed since the study started. Until it finishes,
``plot`` skips the study.

On a SLURM cluster
------------------

Prepare the study, run its tasks as an array job, then assemble it:

.. code-block:: bash

   pixi run experiment langermann prepare --n-runs 10 --seed 0 --name sweep \
       --require-clean
   # Prepared .../results/langermann/sweep with 40 tasks. Run task K (0 to 39):

``prepare`` reserves the study and writes its task list (``tasks.json``) without
running anything. Take the array's range from the number of tasks it prints, and
submit one array element per task, for example with this ``sweep.sbatch``:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=sweep
   #SBATCH --array=0-39
   #SBATCH --cpus-per-task=1
   #SBATCH --mem=2G
   #SBATCH --time=00:30:00
   #SBATCH --output=slurm-%A_%a.out

   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   pixi run experiment langermann run-task sweep --task-index "$SLURM_ARRAY_TASK_ID"

.. code-block:: bash

   sbatch sweep.sbatch
   # once the array has finished:
   pixi run experiment langermann resume sweep

Things to know:

- The results folder must be on storage every node shares (the default is
  ``results/`` in the repository).
- Each array element runs one task on one core. Set the thread variables yourself, as
  above: the single-threading of ``--n-workers`` applies only to the local launcher.
- Every element gets the same time limit, so size ``--time`` for the slowest run, not
  the average.
- ``--array=0-39%20`` runs at most 20 at once.
- A task whose run is already saved exits straight away, so after a failure you can
  resubmit the whole array. A task that fails names its per-run log,
  ``logs/<method>/run-003.log``, which holds the traceback.
- ``resume`` assembles the study, after running any task still missing itself, one
  after another, so rerun failed tasks through the array rather than leave them to
  it.
- ``run-task`` is an ordinary command, so GNU parallel or any other launcher can run
  the tasks too.

How the work is spread
----------------------

The tasks are ordered by configuration, in the order the experiment defines them,
then by run index, and each runs whole in one process: a run's iterations are never
split. The local launcher starts the next task in that order as soon as one finishes;
a SLURM array starts them as the scheduler allows. So:

- a study never finishes faster than its slowest run;
- when runs differ in cost, define the expensive configurations first (for a sweep
  over initial points, fewest initial points first, since those run the most
  iterations), so the cheap ones fill the gaps at the end instead of leaving a long
  tail with most workers idle;
- each task spends 1--2 s starting up, negligible next to a run that takes minutes.
