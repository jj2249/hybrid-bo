"""A run of a method, as pure data (ADR 0003), and the record of how it was produced."""

import functools
import importlib.metadata
import inspect
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hybrid_bo.history import History
from hybrid_bo.problem import Problem, _as_text


@dataclass(frozen=True)
class Run:
    """One run of a method: the evaluated points, the fitted surrogate models and a
    record of how they were produced.

    Holds only arrays, numbers and strings, never the problem or any user callable,
    so it pickles cleanly and loads without the code that produced it (ADR 0003).
    `surrogates` has one fitted surrogate model per iteration and is empty for the
    baselines, so every saved run has the same fields.
    """

    history: History
    record: dict[str, str]
    surrogates: list[object] = field(default_factory=list)  # FittedGP from #27


@functools.cache
def _code_version() -> str:
    """The git commit of the package source (`git describe --always --dirty`), or
    "unknown" outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _version(distribution: str) -> str:
    """The installed version of a package, or "unknown" when it isn't installed."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run_record(
    problem: Problem,
    experiment: object,
    rng: np.random.Generator,
    method: str,
    settings: str,
) -> dict[str, str]:
    """How a run is produced, as text: the problem's variables, structure and
    equations, the experiment, the method and its settings, the generator and the
    package and code versions.

    Call it before the method draws anything, so the generator's state is the one
    the run starts from. The experiment is described by its docstring when it has
    one (the oracle simulation does), else by its name.
    """
    system = problem.system
    seed_sequence = rng.bit_generator.seed_seq
    blocks = (system.x, system.y, system.u)
    name = getattr(experiment, "__qualname__", type(experiment).__qualname__)
    return {
        # The problem
        "u": "; ".join(repr(var) for var in system.u),
        "x": "; ".join(repr(var) for var in system.x),
        "y": "; ".join(repr(var) for var in system.y),
        "h_inputs": ", ".join(var.name for var in system.h_inputs),
        "measured": ", ".join(var.name for var in system.measured),
        "g": _as_text(system.g, *blocks),
        "f": _as_text(problem.f, *blocks),
        # The experiment and the method
        "experiment": inspect.getdoc(experiment) or name,
        "method": method,
        "settings": settings,
        # The generator: rebuild it with SeedSequence(seed, spawn_key=spawn_key)
        "seed": str(getattr(seed_sequence, "entropy", None)),
        "spawn_key": str(getattr(seed_sequence, "spawn_key", None)),
        "generator_state": (
            f"{getattr(seed_sequence, 'n_children_spawned', None)} children spawned, "
            f"bit generator state {rng.bit_generator.state}"
        ),
        # The versions
        "hybrid_bo": _version("hybrid-bo"),
        "git": _code_version(),
        "python": platform.python_version(),
        "casadi": _version("casadi"),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
    }
