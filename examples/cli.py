"""The command line of an example: run its methods as a study, or plot a saved study.

An example script ends with a call that passes everything it defines:

    if __name__ == "__main__":
        cli.main("illustrative_example", problem, experiment, methods, study_settings,
                 optimum=optimum, true_curves=True)

and the pixi tasks run the script as a module from this folder:

    pixi run example illustrative_example [--method hybrid_bo ...] [--n-runs 3]
        [--seed 0] [--n-iterations 5] [--name lcb-vs-ei] [--plot]
    pixi run plot illustrative_example [STUDY]

Running saves the study with `hb.save` to `<results root>/<example>/<study id>/` (the
results root defaults to `results/` at the repo root), where the study id is `--name`
or the UTC time at the start, with `-2`, `-3`, ... appended if taken when the study is
saved, and prints `hb.summary` of it and saves it as `summary.csv` there. `plot` loads
a study (default: the newest saved) with `hb.load` and saves its figures to its
`figures/` folder.
"""

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

import hybrid_bo as hb
from hybrid_bo.studies import Method

RESULTS_ROOT = Path(__file__).parent.parent / "results"


def main(
    name: str,
    problem: hb.Problem,
    experiment: hb.Experiment,
    methods: Mapping[str, Method],
    study_settings: hb.StudySettings,
    *,
    optimum: float | None = None,
    true_curves: bool = False,
    argv: Sequence[str] | None = None,
) -> None:
    """Runs the example's methods as a study and saves it, or plots a saved study,
    as the command line (`argv`, default `sys.argv`) says.

    `name` names the study and its folder. `study_settings` are the defaults the
    flags override. `optimum` is the stated oracle optimum of a test case, which
    regret is measured from. `true_curves` draws the experiment's values over the
    grid in the iteration figures, which only a cheap experiment (an oracle) can
    afford.
    """
    # The command: `plot [STUDY]`, else a run with every argument
    args = list(sys.argv[1:] if argv is None else argv)
    command, rest = ("plot", args[1:]) if args[:1] == ["plot"] else ("run", args)
    parsers = _parsers(name, methods)
    parsed = parsers[command].parse_args(rest)
    example_dir = _from_init_cwd(parsed.results_root) / name

    # Run: the example's settings, overridden by the flags given
    if command == "run":
        given = {
            "n_runs": parsed.n_runs,
            "seed": parsed.seed,
            "n_iterations": parsed.n_iterations,
        }
        settings = replace(
            study_settings,
            **{key: value for key, value in given.items() if value is not None},
        )
        chosen = {label: methods[label] for label in parsed.methods or methods}
        study = hb.study(
            chosen, problem, experiment, settings, name=name, optimum=optimum
        )
        study_id = parsed.name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        study_dir = save_new(study, example_dir, study_id)
        for label in chosen:
            n_saved = sum(run.method == label for run in study.runs)
            print(f"{label}: saved {n_saved} of {settings.n_runs} runs to {study_dir}")
        save_summary(study, study_dir)

    # Plot: the study given, else the one saved last
    else:
        saved = list(example_dir.glob("*/study.json"))
        if parsed.study is not None:
            study_dir = example_dir / parsed.study
        elif saved:
            study_dir = max(saved, key=lambda path: path.stat().st_mtime).parent
        else:
            parsers["plot"].error(f"{example_dir} holds no study")
        if not (study_dir / "study.json").exists():
            parsers["plot"].error(f"{study_dir} holds no study")

    # Plot, also straight after a run with --plot
    if command == "plot" or parsed.plot:
        with plt.style.context(hb.plots.STYLE):
            plot(name, problem, experiment if true_curves else None, study_dir)


def _parsers(
    name: str, methods: Mapping[str, Method]
) -> dict[str, argparse.ArgumentParser]:
    """The parsers of the two commands, by name: `run` (the default) and `plot`."""
    # Running, `pixi run example <name>`
    run_parser = argparse.ArgumentParser(
        prog=f"pixi run example {name}",
        description=f"Run the methods of {name} as a study and save it.",
        epilog=f"To plot a saved study instead: pixi run plot {name} [STUDY]",
    )
    run_parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=list(methods),
        metavar="METHOD",
        help=f"a method to run (repeatable): {', '.join(methods)}; default every one",
    )
    run_parser.add_argument("--n-runs", type=int, help="default: the example's")
    run_parser.add_argument("--seed", type=int, help="default: the example's")
    run_parser.add_argument("--n-iterations", type=int, help="default: the example's")
    run_parser.add_argument("--name", help="the study id; default: the UTC time")
    run_parser.add_argument("--plot", action="store_true", help="plot after running")
    run_parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)

    # Plotting, `pixi run plot <name>`
    plot_parser = argparse.ArgumentParser(
        prog=f"pixi run plot {name}",
        description=f"Plot a saved study of {name} into its figures/ folder.",
    )
    plot_parser.add_argument("study", nargs="?", help="the study id; default: newest")
    plot_parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return {"run": run_parser, "plot": plot_parser}


def _from_init_cwd(path: Path) -> Path:
    """`path`, relative to the folder pixi was started from (pixi runs the example
    from `examples/` and keeps the starting folder in `INIT_CWD`)."""
    return Path(os.environ.get("INIT_CWD", ".")) / path


def save_new(study: hb.Study, example_dir: Path, study_id: str) -> Path:
    """Saves `study` to the folder `study_id` in `example_dir`, with `-2`, `-3`, ...
    appended if it is taken, and returns the folder. `hb.save` creates the folder and
    refuses an existing one, so studies saved at once never share a folder."""
    path, suffix = example_dir / study_id, 2
    while True:
        try:
            return hb.save(study, path)
        except FileExistsError:
            path, suffix = example_dir / f"{study_id}-{suffix}", suffix + 1


def save_summary(study: hb.Study, study_dir: Path) -> None:
    """Prints `hb.summary(study)` and saves it to `summary.csv` in `study_dir`."""
    table = hb.summary(study)
    table.write_csv(study_dir / "summary.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(table)


def plot(
    name: str,
    problem: hb.Problem,
    experiment: hb.Experiment | None,
    study_dir: Path,
) -> None:
    """Saves the figures of the study in `study_dir` to its `figures/` folder:
    `convergence.png`, `regret.png` (when the study has an oracle optimum) and
    `<method>/run-000/iteration-00.png` for every iteration of every run, with the
    true curves from `experiment` if given. A figure that can't be drawn is skipped
    with a message. Saves `summary.csv` too if the study has none (a study saved
    before the CLI wrote summaries).

    Raises `ValueError` if the study was run on another example.
    """
    study = hb.load(study_dir)
    if study.name != name:
        msg = f"{study_dir} holds a study of {study.name}, not {name}"
        raise ValueError(msg)
    if not (study_dir / "summary.csv").exists():
        save_summary(study, study_dir)
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

    # Every iteration of every run
    for run in study.runs:
        run_dir = figures_dir / run.method / f"run-{run.index:03d}"
        for i in range(hb.history.n_iterations(run.history)):
            try:
                figure = hb.plots.iteration(run, problem, i, experiment=experiment)
            except (ValueError, TypeError, hb.HybridBOError) as error:
                print(f"{run.method} run {run.index}: no iteration figures: {error}")
                break
            run_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(run_dir / f"iteration-{i:02d}.png")
            plt.close(figure)
    print(f"Saved the figures to {figures_dir}")
