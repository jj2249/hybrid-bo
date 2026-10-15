"""The CLI for the examples: run an example's methods, and plot saved runs.

    python examples/main.py run illustrative_example [--method hybrid_bo ...]
        [--n-runs 10] [--seed 0] [--n-iterations 5]
    python examples/main.py plot examples/results/illustrative_example/<UTC time>
        [--run 0] [--iteration -1] [--show]

An example is a module in this folder that defines `problem`, `experiment`,
`methods`, `settings` (of the BO methods, for the plots) and the defaults `seed`,
`n_runs`, `n_initial` and `n_iterations`. `run` pickles each method's runs to
`<results root>/<example>/<UTC time>/<method>.pkl`; `plot` reads them back and saves
the figures next to them.
"""

import argparse
import importlib
import pickle
from collections.abc import Sequence
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
    n_runs: int,
    seed: int | None,
    n_iterations: int,
    results_dir: Path,
) -> None:
    """Runs each method `n_runs` times and pickles its list of runs to
    `results_dir/<method>.pkl`.

    One seed for every method, so run i of each method starts from the same initial
    points. A run that fails with a `HybridBOError` is saved as far as it got.
    """
    # One seed for every method, so run i of each method starts from the same
    # initial points
    results_dir.mkdir(parents=True)
    entropy = np.random.SeedSequence(seed).entropy
    for name in method_names:
        runs = []
        for rng in np.random.default_rng(entropy).spawn(n_runs):
            # A failed run still saves the points evaluated before the error
            initial_points = hb.initial_points(
                example.problem.system.u, example.n_initial, rng
            )
            try:
                runs.append(
                    example.methods[name](
                        example.problem,
                        example.experiment,
                        initial_points,
                        n_iterations,
                        rng=rng,
                    )
                )
            except hb.HybridBOError as error:
                print(f"{name}: run failed: {error!r}")
                if error.run is not None:
                    runs.append(error.run)

        # One pickle per method: its list of runs
        with (results_dir / f"{name}.pkl").open("wb") as file:
            pickle.dump(runs, file)
        print(f"{name}: saved {len(runs)} of {n_runs} runs to {results_dir}")


def plot(
    example: ModuleType, results_dir: Path, run_index: int, iteration: int
) -> None:
    """Saves the convergence of every method's runs to `convergence.png`, and the
    figure of one iteration of one run of each method to `<method>.png`, in
    `results_dir`.

    The figures' GP fits and root-finding starts draw from a generator seeded with 0,
    so plotting again gives the same figures.
    """
    runs = {
        path.stem: pickle.loads(path.read_bytes())
        for path in sorted(results_dir.glob("*.pkl"))
    }

    # Convergence: every method on one axes
    figure, ax = plt.subplots()
    for name, method_runs in runs.items():
        if method_runs:
            hb.plots.convergence(method_runs, ax=ax, label=name)
    ax.legend()
    figure.savefig(results_dir / "convergence.png")

    # One iteration of one run per method; a figure that can't be drawn is skipped
    for name, method_runs in runs.items():
        if len(method_runs) <= run_index:
            print(f"{name}: no run {run_index} to plot")
            continue
        try:
            figure = hb.plots.iteration(
                method_runs[run_index],
                iteration,
                example.problem,
                np.random.default_rng(0),
                example.settings.get(name),
                example.experiment,
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
        "--iteration", type=int, default=-1, help="default: -1, the last"
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
        run(
            example,
            methods,
            example.n_runs if args.n_runs is None else args.n_runs,
            example.seed if args.seed is None else args.seed,
            example.n_iterations if args.n_iterations is None else args.n_iterations,
            results_dir,
        )
    # Plot: the results folder is <results root>/<example>/<UTC time>
    else:
        example = importlib.import_module(args.results_dir.resolve().parent.name)
        plot(example, args.results_dir, args.run, args.iteration)
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()
