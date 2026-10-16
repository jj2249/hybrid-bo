"""A run of a method, as pure data (ADR 0003), and the record of how it was produced."""

import functools
import importlib.metadata
import inspect
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hybrid_bo.acquisition import AcquisitionSolution, Scenarios
from hybrid_bo.gp import FittedGP
from hybrid_bo.history import History
from hybrid_bo.problem import Problem, _as_text


@dataclass(frozen=True)
class Run:
    """One run of a method: the evaluated points, the fitted surrogate models, the
    acquisition problems' solutions and a record of how they were produced.

    Holds only arrays, numbers and strings, never the problem or any user callable,
    so it saves as JSON (`hb.save`) and loads without the code that produced it
    (ADR 0003, ADR 0004).
    `method` is the label the run was run under: the key in the study's methods, or
    the method's name when run on its own. `index` is its position among that
    method's runs in a study, so run i of every method starts from the same initial
    points. `surrogates` has one fitted surrogate model per iteration, the one behind
    that iteration's trial point, and `acquisitions` the solution of that iteration's
    acquisition problem; both are empty for the baselines. `scenarios` are hybrid
    BO's, the same in every iteration, and None for the other methods. So every saved
    run has the same fields.
    """

    history: History
    record: dict[str, str]
    method: str
    index: int = 0
    surrogates: list[FittedGP] = field(default_factory=list)
    acquisitions: list[AcquisitionSolution] = field(default_factory=list)
    scenarios: Scenarios | None = None


def _git(path: Path, *args: str) -> str:
    """What `git *args` prints, run in the folder `path`.

    Raises `OSError` without git and `CalledProcessError` outside a git checkout.
    """
    result = subprocess.run(
        ["git", *args], cwd=path, capture_output=True, text=True, check=True
    )
    return result.stdout


def code_version(path: Path) -> str:
    """The git commit of the checkout holding the folder `path`, marked `-dirty` if a
    committed file has changed and `-untracked` if a file that isn't ignored was never
    added, or "unknown" outside a git checkout. Only a version with neither mark is
    clean (`is_clean`), so a study recording it can be reproduced."""
    try:
        described = _git(path, "describe", "--always", "--dirty").strip()
        status = _git(path, "status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    has_untracked = any(line.startswith("??") for line in status.splitlines())
    return f"{described}-untracked" if has_untracked else described


def is_clean(version: str) -> bool:
    """Whether the code version `version` (from `code_version`) pins the code: a
    commit, with nothing changed or untracked."""
    return version != "unknown" and not version.endswith(("-dirty", "-untracked"))


@functools.cache
def _code_version() -> str:
    """The code version of the package source (`code_version`), found once."""
    return code_version(Path(__file__).parent)


def _version(distribution: str) -> str:
    """The installed version of a package, or "unknown" when it isn't installed."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _problem_record(problem: Problem, measure: object) -> dict[str, str]:
    """The problem's variables, structure and equations, and the measurement, as text.

    The measurement is described by its docstring when it has one (the oracle
    simulation does), else by its name.
    """
    system = problem.system
    blocks = (system.x, system.y, system.u)
    name = getattr(measure, "__qualname__", type(measure).__qualname__)
    return {
        "u": "; ".join(repr(var) for var in system.u),
        "x": "; ".join(repr(var) for var in system.x),
        "y": "; ".join(repr(var) for var in system.y),
        "h_inputs": ", ".join(var.name for var in system.h_inputs),
        "measured": ", ".join(var.name for var in system.measured),
        "g": _as_text(system.g, *blocks),
        "f": _as_text(problem.f, *blocks),
        "measure": inspect.getdoc(measure) or name,
    }


def _generator_record(rng: np.random.Generator) -> dict[str, str]:
    """The generator's seed, spawn key and state, as text: rebuild it with
    `SeedSequence(seed, spawn_key=spawn_key)`."""
    seed_sequence = rng.bit_generator.seed_seq
    return {
        "seed": str(getattr(seed_sequence, "entropy", None)),
        "spawn_key": str(getattr(seed_sequence, "spawn_key", None)),
        "generator_state": (
            f"{getattr(seed_sequence, 'n_children_spawned', None)} children spawned, "
            f"bit generator state {rng.bit_generator.state}"
        ),
    }


def _versions_record() -> dict[str, str]:
    """The versions of hybrid_bo (with its git commit), Python and the numerical
    packages."""
    return {
        "hybrid_bo": _version("hybrid-bo"),
        "git": _code_version(),
        "python": platform.python_version(),
        "casadi": _version("casadi"),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
    }


def run_record(
    problem: Problem,
    measure: object,
    rng: np.random.Generator,
    method: str,
    settings: str,
) -> dict[str, str]:
    """How a run is produced, as text: the problem's variables, structure and
    equations, the measurement, the method and its settings, the generator and the
    package and code versions.

    Call it before the method draws anything, so the generator's state is the one
    the run starts from. The measurement is described by its docstring when it has
    one (the oracle simulation does), else by its name.
    """
    return (
        _problem_record(problem, measure)
        | {"method": method, "settings": settings}
        | _generator_record(rng)
        | _versions_record()
    )
