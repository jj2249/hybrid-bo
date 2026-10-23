"""The command line of an experiment: run its methods as a study, or plot a saved study.

An experiment ends with a call that passes everything it defines:

    if __name__ == "__main__":
        cli.main("illustrative_example", problem, measure, configurations,
                 study_settings, optimum=optimum, true_curves=True)

and the pixi tasks run the script as a module from this folder:

    pixi run experiment illustrative_example [--method hybrid_bo ...] [--n-runs 3]
        [--seed 0] [--n-workers 4] [--name lcb-vs-ei] [--plot] [--plot-runs 0 3]
        [--require-clean]
    pixi run plot illustrative_example [STUDY]
    pixi run experiment illustrative_example plot [STUDY] [--plot-runs all]

Running saves the study with `hb.save` to `<results root>/<experiment>/<study id>/` (the
results root defaults to `results/` at the repo root), running each configuration
with its own budget (there is no flag for it) in `--n-workers` processes, where the study id is `--name`
or the UTC time at the start, and prints `hb.summary` of it and saves it as
`summary.csv` there. Before anything runs, the study id is reserved by creating the
study log, `<study id>.log`, beside the study folders; an id is taken if that log or
its folder exists. A taken `--name` is refused, and a taken UTC time gets `-2`, `-3`,
... appended. The study log records the library's progress (DEBUG and above; the
console shows INFO and above) and every warning, a line logged by a run starting
with the run's tag (`hybrid_bo/run-003`); each run's lines also go to its per-run log in
`<study id>.logs/`. Once the study is saved, the log moves into its folder as
`study.log`, and the per-run logs to `logs/<label>/run-003.log`. A study that doesn't
finish leaves them where they were, which keeps its id taken. Running from a tree that isn't clean
(uncommitted or untracked files) warns that the study won't be reproducible;
`--require-clean` refuses to run instead. `plot` loads a study (default: the newest
saved) with `hb.load` and saves its figures to its `figures/` folder: convergence and
regret, and the iteration figures of the runs `--plot-runs` names (indices, or `all`;
default run 0 of each method). On the run command, `--plot-runs` implies `--plot`.
"""

import argparse
import logging
import os
import sys
import warnings
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

import hybrid_bo as hb
from hybrid_bo.run import code_version, is_clean
from hybrid_bo.saving import ARTIFACT_FILE, OLD_ARTIFACT_FILE
from hybrid_bo.studies import run_of

RESULTS_ROOT = Path(__file__).parent.parent / "results"
ARTIFACTS = (ARTIFACT_FILE, OLD_ARTIFACT_FILE)
STUDY_LOG = "study.log"
PER_RUN_LOGS = "logs"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
TAGGED_FORMAT = "%(asctime)s %(run_tag)s%(levelname)s %(name)s: %(message)s"
CONSOLE_FORMAT = "%(run_tag)s%(levelname)s %(name)s: %(message)s"

_logger = logging.getLogger(__name__)


def main(
    name: str,
    problem: hb.Problem,
    measure: hb.Measure,
    configurations: Mapping[str, hb.Configuration],
    study_settings: hb.StudySettings,
    *,
    optimum: float | None = None,
    true_curves: bool = False,
    argv: Sequence[str] | None = None,
) -> None:
    """Runs the experiment's configurations as a study and saves it, or plots a saved
    study, as the command line (`argv`, default `sys.argv`) says.

    `name` names the study and its folder. `--method` chooses configurations by their
    labels, each run with its own budget. `study_settings` are the defaults the
    flags override. `optimum` is the stated oracle optimum of a benchmark experiment,
    which regret is measured from. `true_curves` draws the measurement's values over the
    grid in the iteration figures, which only a cheap measurement (an oracle) can
    afford.
    """
    # The command: `plot [STUDY]`, else a run with every argument
    args = list(sys.argv[1:] if argv is None else argv)
    command, rest = ("plot", args[1:]) if args[:1] == ["plot"] else ("run", args)
    parsers = _parsers(name, configurations)
    parsed = parsers[command].parse_args(rest)
    experiment_dir = _from_init_cwd(parsed.results_root) / name

    # Run: a tree that isn't clean gives a study that isn't reproducible
    if command == "run":
        version = code_version(Path(hb.__file__).parent)
        clean = is_clean(version)
        if not clean and parsed.require_clean:
            parsers["run"].error(f"the code version {version} isn't clean")

        # The experiment's settings, overridden by the flags given
        given = {"n_runs": parsed.n_runs, "seed": parsed.seed}
        settings = replace(
            study_settings,
            **{key: value for key, value in given.items() if value is not None},
        )
        chosen = {
            label: configurations[label] for label in parsed.methods or configurations
        }

        # Reserve the study id: a taken name is refused, a taken UTC time suffixed
        default_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        try:
            log_path = reserve(
                experiment_dir,
                parsed.name or default_id,
                suffix_if_taken=not parsed.name,
            )
        except FileExistsError as error:
            parsers["run"].error(str(error))

        # Run and save the study, logging to its study log and run logs; a study
        # that doesn't finish leaves them where they are
        per_run_logs = log_path.with_suffix(".logs")
        try:
            with study_log(log_path, per_run_logs):
                if not clean:
                    _logger.warning(
                        "The code version %s isn't clean (uncommitted or untracked "
                        "files), so this study won't be reproducible",
                        version,
                    )
                study = hb.study(
                    chosen,
                    problem,
                    measure,
                    settings,
                    name=name,
                    optimum=optimum,
                    n_workers=parsed.n_workers,
                )
                study_dir = hb.save(study, log_path.with_suffix(""))
                for label in chosen:
                    n_saved = sum(run.method == label for run in study.runs)
                    _logger.info(
                        "%s: saved %d of %d runs to %s",
                        label,
                        n_saved,
                        settings.n_runs,
                        study_dir,
                    )
        except BaseException:
            print(f"The study didn't finish; its log is {log_path}", file=sys.stderr)
            if per_run_logs.exists():
                print(f"and its per-run logs are in {per_run_logs}", file=sys.stderr)
            raise
        log_path.replace(study_dir / STUDY_LOG)
        if per_run_logs.exists():
            per_run_logs.replace(study_dir / PER_RUN_LOGS)
        save_summary(study, study_dir)

    # Plot: the study given, else the one saved last
    else:
        saved = [
            path for name in ARTIFACTS for path in experiment_dir.glob(f"*/{name}")
        ]
        if parsed.study is not None:
            study_dir = experiment_dir / parsed.study
        elif saved:
            study_dir = max(saved, key=lambda path: path.stat().st_mtime).parent
        else:
            parsers["plot"].error(f"{experiment_dir} holds no study")
        if not any((study_dir / name).exists() for name in ARTIFACTS):
            parsers["plot"].error(f"{study_dir} holds no study")

    # Plot, also straight after a run with --plot or --plot-runs
    if command == "plot" or parsed.plot or parsed.plot_runs is not None:
        runs = _plot_runs(parsers[command], parsed.plot_runs)
        with plt.style.context(hb.plots.STYLE):
            plot(name, problem, measure if true_curves else None, study_dir, runs=runs)


def _parsers(
    name: str, configurations: Mapping[str, hb.Configuration]
) -> dict[str, argparse.ArgumentParser]:
    """The parsers of the two commands, by name: `run` (the default) and `plot`."""
    # Running, `pixi run experiment <name>`
    run_parser = argparse.ArgumentParser(
        prog=f"pixi run experiment {name}",
        description=f"Run the configurations of {name} as a study and save it.",
        epilog=f"To plot a saved study instead: pixi run plot {name} [STUDY]",
    )
    run_parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=list(configurations),
        metavar="METHOD",
        help=(
            f"a configuration to run, by its label (repeatable): "
            f"{', '.join(configurations)}; default every one"
        ),
    )
    run_parser.add_argument("--n-runs", type=int, help="default: the experiment's")
    run_parser.add_argument("--seed", type=int, help="default: the experiment's")
    run_parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="worker processes to run the runs on (default 1: in this process)",
    )
    run_parser.add_argument("--name", help="the study id; default: the UTC time")
    run_parser.add_argument("--plot", action="store_true", help="plot after running")
    _add_plot_runs(run_parser, " (implies --plot)")
    run_parser.add_argument(
        "--require-clean",
        action="store_true",
        help="refuse to run with uncommitted or untracked files",
    )
    run_parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)

    # Plotting, `pixi run plot <name>`
    plot_parser = argparse.ArgumentParser(
        prog=f"pixi run plot {name}",
        description=f"Plot a saved study of {name} into its figures/ folder.",
    )
    plot_parser.add_argument("study", nargs="?", help="the study id; default: newest")
    _add_plot_runs(plot_parser)
    plot_parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return {"run": run_parser, "plot": plot_parser}


def _add_plot_runs(parser: argparse.ArgumentParser, note: str = "") -> None:
    """Adds `--plot-runs` to the parser: run indices or `all`, None when not given."""
    parser.add_argument(
        "--plot-runs",
        nargs="+",
        type=_plot_run,
        metavar="RUN",
        help=f"the runs with iteration figures: indices or all; default 0{note}",
    )


def _plot_run(text: str) -> int | str:
    """One `--plot-runs` value: a run index (from 0) or `all`.

    Raises `argparse.ArgumentTypeError` otherwise.
    """
    if text == "all":
        return text
    if not text.isdigit():
        msg = f"not a run index or all: {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return int(text)


def _plot_runs(
    parser: argparse.ArgumentParser, values: list[int | str] | None
) -> list[int] | None:
    """The runs `--plot-runs` chose: its indices, run 0 if it wasn't given, or None
    (every run) for `all`. `all` among indices is a usage error (`parser.error`)."""
    if values is None:
        return [0]
    indices = [value for value in values if isinstance(value, int)]
    if len(indices) == len(values):
        return indices
    if len(values) > 1:
        parser.error("--plot-runs takes run indices or all, not both")
    return None


def _from_init_cwd(path: Path) -> Path:
    """`path`, relative to the folder pixi was started from (pixi runs the experiment
    from `experiments/` and keeps the starting folder in `INIT_CWD`)."""
    return Path(os.environ.get("INIT_CWD", ".")) / path


def reserve(experiment_dir: Path, study_id: str, *, suffix_if_taken: bool) -> Path:
    """Reserves a study id in `experiment_dir` by creating its study log,
    `<study id>.log`, and returns the log's path (the study folder is the path
    without `.log`). An id is taken if its log, its per-run logs (`<study id>.logs`)
    or its folder exists; creating the log fails if it exists, so studies started at
    once never share an id.

    A taken id gets `-2`, `-3`, ... appended if `suffix_if_taken`; otherwise raises
    `FileExistsError` saying what holds it.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    candidate, n = study_id, 2
    while True:
        folder, log = experiment_dir / candidate, experiment_dir / f"{candidate}.log"
        per_run_logs = experiment_dir / f"{candidate}.logs"
        if folder.exists():
            holder = f"the saved study {folder}"
        elif per_run_logs.exists():
            holder = f"the per-run logs of a study that didn't finish, {per_run_logs}"
        else:
            try:
                log.open("x").close()
            except FileExistsError:
                holder = f"the log of a study that didn't finish, {log}"
            else:
                return log
        if not suffix_if_taken:
            msg = f"the study id {candidate} is taken by {holder}"
            raise FileExistsError(msg)
        candidate, n = f"{study_id}-{n}", n + 1


@contextmanager
def study_log(path: Path, per_run_logs: Path) -> Generator[None]:
    """Logs to the study log at `path` (DEBUG and above from the library, INFO and
    above from the rest) and to the console (INFO and above), with warnings logged
    too, until the block ends; then puts logging back as it was.

    A line logged by a run of the study starts with the run's tag,
    `<label>/run-<index>`, and also goes to the run's per-run log,
    `<per_run_logs>/<label>/run-<index>.log`, created when the run first logs.
    """
    # The handlers, on the root logger so the experiment's own records are kept
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(logging.Formatter(TAGGED_FORMAT))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    run_handler = PerRunLogs(per_run_logs)
    run_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    for handler in (file_handler, console):
        handler.addFilter(tag_run)

    # Attach them, saving what to put back: the levels, and whether warnings were
    # logged already (logging's `showwarning` replaces the warnings module's)
    root, library = logging.getLogger(), logging.getLogger("hybrid_bo")
    root_level, library_level = root.level, library.level
    captured = getattr(warnings.showwarning, "__module__", None) == "logging"
    root.addHandler(file_handler)
    root.addHandler(console)
    root.addHandler(run_handler)
    root.setLevel(logging.INFO)
    library.setLevel(logging.DEBUG)
    logging.captureWarnings(True)

    try:
        yield
    finally:
        logging.captureWarnings(captured)
        root.setLevel(root_level)
        library.setLevel(library_level)
        root.removeHandler(run_handler)
        root.removeHandler(console)
        root.removeHandler(file_handler)
        file_handler.close()


def run_tag(label: str, index: int) -> str:
    """The tag of run `index` of the configuration `label`, `label/run-003`, as its
    lines in the study log start and its log is named."""
    return f"{label}/run-{index:03d}"


def tag_run(record: logging.LogRecord) -> bool:
    """Sets `record.run_tag` to the tag of the run that logged it and a space, or to
    nothing for a record from outside the runs; keeps every record (a filter)."""
    run = run_of(record)
    record.run_tag = "" if run is None else f"{run_tag(*run)} "
    return True


class PerRunLogs(logging.Handler):
    """Writes each record a run of the study logs to the run's per-run log,
    `<folder>/<label>/run-<index>.log`, and ignores the others."""

    def __init__(self, folder: Path) -> None:
        """Writes the per-run logs under `folder`, created with the first of them."""
        super().__init__()
        self.folder = folder

    def emit(self, record: logging.LogRecord) -> None:
        """Appends the record to its run's per-run log, if a run logged it."""
        run = run_of(record)
        if run is None:
            return
        try:
            path = self.folder / f"{run_tag(*run)}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as log:
                log.write(self.format(record) + "\n")
        except Exception:  # noqa: BLE001  (logging's contract: report, don't raise)
            self.handleError(record)


def save_summary(study: hb.Study, study_dir: Path) -> None:
    """Prints `hb.summary(study)` and saves it to `summary.csv` in `study_dir`."""
    table = hb.summary(study)
    table.write_csv(study_dir / "summary.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(table)


def plot(
    name: str,
    problem: hb.Problem,
    measure: hb.Measure | None,
    study_dir: Path,
    *,
    runs: Sequence[int] | None,
) -> None:
    """Saves the figures of the study in `study_dir` to its `figures/` folder:
    `convergence.png`, `regret.png` (when the study has an oracle optimum) and
    `<method>/run-000/iteration-00.png` for every iteration of each method's runs
    with these indices (None: every run), with the oracle values from `measure` if
    given; hybrid BO's are drawn without the GP panels when it has more than one
    unknown state. A figure that can't be drawn is skipped with a message. Saves
    `summary.csv` too if the study has none (a study saved before the CLI wrote
    summaries).

    Raises `ValueError` if the study was run on another experiment.
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

    # Every iteration of the runs chosen
    for run in study.runs:
        if runs is not None and run.index not in runs:
            continue
        run_dir = figures_dir / run.method / f"run-{run.index:03d}"
        # Hybrid BO's GP panels draw one unknown state, so with more it goes without
        hybrid_with_several_unknown_states = (
            isinstance(run.settings, hb.HybridBOSettings) and problem.system.n_y > 1
        )
        for i in range(hb.history.n_iterations(run.history)):
            try:
                figure = hb.plots.iteration(
                    run,
                    problem,
                    i,
                    measure=measure,
                    gp=not hybrid_with_several_unknown_states,
                )
            except (ValueError, hb.HybridBOError) as error:
                print(f"{run.method} run {run.index}: no iteration figures: {error}")
                break
            run_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(run_dir / f"iteration-{i:02d}.png")
            plt.close(figure)
    print(f"Saved the figures to {figures_dir}")
