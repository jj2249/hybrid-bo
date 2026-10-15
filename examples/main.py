"""The CLI for the examples: run an example's methods as a study, and plot studies.

    pixi run example illustrative_example [--method hybrid_bo ...] [--n-runs 3]
        [--seed 0] [--n-iterations 5] [--name lcb-vs-ei] [--plot]
    pixi run plot illustrative_example [STUDY]

An example is a module in this folder that defines `problem`, `experiment`,
`methods` (label -> method), its default study `settings` and, for a test case, the
`oracle` (the study computes the oracle optimum from it) or the known `optimum`.
`run` saves a study with `hb.save` to `<results root>/<example>/<study id>/`, where
the study id is `--name` or the UTC time, with `-2`, `-3`, ... appended if taken.
`plot` loads a study (default: the newest, the largest id) with `hb.load` and saves
its figures to its `figures/` folder.
"""

import argparse
import importlib
import pkgutil
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt

import hybrid_bo as hb

EXAMPLES_DIR = Path(__file__).parent
RESULTS_ROOT = EXAMPLES_DIR / "results"


def examples() -> dict[str, ModuleType]:
    """The example modules in this folder, those that define `methods`, by name."""
    names = [
        info.name
        for info in pkgutil.iter_modules([str(EXAMPLES_DIR)])
        if not info.ispkg and info.name != "main"
    ]
    modules = [importlib.import_module(name) for name in names]
    return {module.__name__: module for module in modules if hasattr(module, "methods")}


def new_study_dir(example_dir: Path, name: str | None) -> Path:
    """The folder of a new study in `example_dir`: `name`, else the UTC time to the
    second, with `-2`, `-3`, ... appended if it is taken."""
    stem = name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path, suffix = example_dir / stem, 2
    while path.exists():
        path, suffix = example_dir / f"{stem}-{suffix}", suffix + 1
    return path


def run(
    example: ModuleType,
    method_names: Sequence[str],
    settings: hb.StudySettings,
    study_dir: Path,
) -> None:
    """Runs a study of the chosen methods and saves it to `study_dir`.

    Run i of each method starts from the same initial points. A run that fails with a
    `HybridBOError` is saved as far as it got. Seed None draws a fresh seed, which
    the saved settings hold. The oracle optimum is the example's `optimum`, else
    computed from its `oracle`, if it defines either.
    """
    study = hb.study(
        {name: example.methods[name] for name in method_names},
        example.problem,
        example.experiment,
        settings,
        name=example.__name__,
        optimum=getattr(example, "optimum", None),
        oracle=getattr(example, "oracle", None),
    )
    hb.save(study, study_dir)
    for name in method_names:
        n_saved = sum(run.method == name for run in study.runs)
        print(f"{name}: saved {n_saved} of {settings.n_runs} runs to {study_dir}")


def plot(example: ModuleType, study_dir: Path) -> None:
    """Saves the figures of the study in `study_dir` to its `figures/` folder:
    `convergence.png`, `regret.png` (when the study has an oracle optimum) and
    `<method>/run-000/iteration-00.png` for every iteration of every run. A figure
    that can't be drawn is skipped with a message."""
    study = hb.load(study_dir)
    figures_dir = study_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Convergence, and regret when the study has an oracle optimum
    hb.plots.convergence(study)
    plt.savefig(figures_dir / "convergence.png")
    plt.close("all")
    try:
        hb.plots.regret(study)
    except ValueError as error:
        print(f"no regret figure: {error}")
    else:
        plt.savefig(figures_dir / "regret.png")
        plt.close("all")

    # Every iteration of every run. The true curves come from the experiment only in
    # a test case, since a real experiment would run at every grid point
    experiment = example.experiment if hasattr(example, "oracle") else None
    for run in study.runs:
        run_dir = figures_dir / run.method / f"run-{run.index:03d}"
        n_iterations = len(run.history.u) - run.history.n_initial
        for i in range(n_iterations):
            try:
                figure = hb.plots.iteration(
                    run, example.problem, i, experiment=experiment
                )
            except (ValueError, TypeError, hb.HybridBOError) as error:
                print(f"{run.method} run {run.index}: no iteration figures: {error}")
                break
            run_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(run_dir / f"iteration-{i:02d}.png")
            plt.close(figure)
    print(f"Saved the figures to {figures_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    """Parses the command line (`argv`, default `sys.argv`) and runs or plots."""
    # The examples, listed in the help of `run`
    found = examples()
    listing = "\n".join(
        f"  {name}: {', '.join(module.methods)}" for name, module in found.items()
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    # The `run` command, `pixi run example`
    run_parser = commands.add_parser(
        "run",
        prog="pixi run example",
        help="run an example's methods as a study",
        epilog=f"examples and their methods:\n{listing}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "example", choices=found, metavar="EXAMPLE", help="one of the examples below"
    )
    run_parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        metavar="METHOD",
        help="a method to run (repeatable); default every method of the example",
    )
    run_parser.add_argument("--n-runs", type=int, help="default: the example's")
    run_parser.add_argument("--seed", type=int, help="default: the example's")
    run_parser.add_argument("--n-iterations", type=int, help="default: the example's")
    run_parser.add_argument("--name", help="the study id; default: the UTC time")
    run_parser.add_argument("--plot", action="store_true", help="plot after running")
    run_parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)

    # The `plot` command, `pixi run plot`
    plot_parser = commands.add_parser(
        "plot", prog="pixi run plot", help="plot a saved study of an example"
    )
    plot_parser.add_argument(
        "example", choices=found, metavar="EXAMPLE", help=", ".join(found)
    )
    plot_parser.add_argument("study", nargs="?", help="the study id; default: newest")
    plot_parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)

    # Run: the example's settings, overridden by the flags given
    args = parser.parse_args(argv)
    example = found[args.example]
    example_dir = args.results_root / args.example
    if args.command == "run":
        methods = args.methods or list(example.methods)
        unknown = sorted(set(methods) - set(example.methods))
        if unknown:
            parser.error(f"{args.example} has no method {', '.join(unknown)}")
        given = {
            "n_runs": args.n_runs,
            "seed": args.seed,
            "n_iterations": args.n_iterations,
        }
        settings = replace(
            example.settings,
            **{name: value for name, value in given.items() if value is not None},
        )
        study_dir = new_study_dir(example_dir, args.name)
        run(example, methods, settings, study_dir)

    # Plot: the study given, else the newest
    else:
        ids = sorted(path.parent.name for path in example_dir.glob("*/study.json"))
        study_id = args.study or (ids[-1] if ids else None)
        if study_id not in ids:
            parser.error(f"{example_dir} holds no study {study_id or ''}")
        study_dir = example_dir / study_id

    if args.command == "plot" or args.plot:
        with plt.style.context(hb.plots.STYLE):
            plot(example, study_dir)


if __name__ == "__main__":
    main()
