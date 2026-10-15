"""The CLI for the examples: run an example's methods, and plot saved runs.

    python examples/main.py run illustrative_example [--method hybrid_bo ...]
        [--n-runs 10] [--seed 0] [--n-iterations 5]
    python examples/main.py plot examples/results/illustrative_example/<UTC time>
        [--run 0] [--iteration -1] [--show]

`plot --show` opens the figures, each with - and + buttons over the run's iterations.

An example is a module in this folder that defines `problem`, `experiment`,
`methods`, `settings` (of the BO methods, for the plots), the default
`study_settings` and, for a test case, the `oracle` (for the oracle optimum). `run` runs a study and saves it with `hb.save` to
`<results root>/<example>/<UTC time>/`; `plot` loads it with `hb.load` and saves the
figures next to it.
"""

import argparse
import importlib
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np

import hybrid_bo as hb

RESULTS_ROOT = Path(__file__).parent / "results"


def run(
    example: ModuleType,
    method_names: Sequence[str],
    settings: hb.StudySettings,
    results_dir: Path,
) -> None:
    """Runs a study of the chosen methods and saves it to `results_dir`.

    Run i of each method starts from the same initial points. A run that fails with a
    `HybridBOError` is saved as far as it got. Seed None draws a fresh seed, which
    the saved settings hold. The oracle optimum is computed from the example's
    `oracle`, if it defines one.
    """
    study = hb.study(
        {name: example.methods[name] for name in method_names},
        example.problem,
        example.experiment,
        settings,
        name=example.__name__,
        oracle=getattr(example, "oracle", None),
    )
    hb.save(study, results_dir)
    for name in method_names:
        n_saved = sum(run.method == name for run in study.runs)
        print(f"{name}: saved {n_saved} of {settings.n_runs} runs to {results_dir}")


def plot(
    example: ModuleType, results_dir: Path, run_index: int, iteration: int
) -> None:
    """Saves the convergence of every method's runs to `convergence.png`, and the
    iterations of one run of each method, showing `iteration`, to `<method>.png`, in
    `results_dir`. Returns nothing; the figures stay open for `plt.show`, where
    each has - and + buttons to step through its iterations.

    The figures' root-finding starts draw from a generator seeded with 0, so plotting
    again gives the same figures.
    """
    study = hb.load(results_dir)
    runs: dict[str, list[hb.Run]] = {}
    for run in study.runs:
        runs.setdefault(run.method, []).append(run)

    # Convergence: every method on one axes
    figure, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    figure.suptitle("convergence")
    for name, method_runs in runs.items():
        hb.plots.convergence(method_runs, ax=ax, label=name)
    figure.legend(loc="outside right upper")
    figure.savefig(results_dir / "convergence.png")

    # The iterations of one run per method; a figure that can't be drawn is skipped
    for name, method_runs in runs.items():
        selected = [run for run in method_runs if run.index == run_index]
        if not selected:
            print(f"{name}: no run {run_index} to plot")
            continue
        try:
            figure = hb.plots.iterations(
                selected[0],
                example.problem,
                np.random.default_rng(0),
                example.settings.get(name),
                example.experiment,
                i=iteration,
            )
        except (ValueError, TypeError, IndexError, hb.HybridBOError) as error:
            print(f"{name}: no figure: {error}")
            continue
        figure.savefig(results_dir / f"{name}.png")
    print(f"Saved the figures to {results_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    """Parses the command line (`argv`, default `sys.argv`) and runs or plots."""
    # The `run` command
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run an example's methods")
    run_parser.add_argument("example", help="module name, e.g. illustrative_example")
    run_parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="a method to run (repeatable); default every method of the example",
    )
    run_parser.add_argument("--n-runs", type=int, help="default: the example's")
    run_parser.add_argument("--seed", type=int, help="default: the example's")
    run_parser.add_argument("--n-iterations", type=int, help="default: the example's")
    run_parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)

    # The `plot` command
    plot_parser = commands.add_parser("plot", help="plot the runs in a results folder")
    plot_parser.add_argument(
        "results_dir", type=Path, help="<results root>/<example>/<UTC time>"
    )
    plot_parser.add_argument("--run", type=int, default=0, help="default: 0")
    plot_parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="the iteration shown first and saved; default: -1, the last",
    )
    plot_parser.add_argument("--show", action="store_true", help="open the figures")

    # Run: the example's defaults unless given, into a new folder per invocation
    args = parser.parse_args(argv)
    if args.command == "run":
        example = importlib.import_module(args.example)
        methods = args.methods or list(example.methods)
        unknown = sorted(set(methods) - set(example.methods))
        if unknown:
            parser.error(f"{args.example} has no method {', '.join(unknown)}")
        results_dir = (
            args.results_root
            / args.example
            / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        )
        given = {
            "n_runs": args.n_runs,
            "seed": args.seed,
            "n_iterations": args.n_iterations,
        }
        settings = replace(
            example.study_settings,
            **{name: value for name, value in given.items() if value is not None},
        )
        run(example, methods, settings, results_dir)
    # Plot: the results folder is <results root>/<example>/<UTC time>
    else:
        example = importlib.import_module(args.results_dir.resolve().parent.name)
        with plt.style.context(hb.plots.STYLE):
            plot(example, args.results_dir, args.run, args.iteration)
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()
