"""A run of a method, as pure data (ADR 0003), and its artifact: how it was produced."""

import importlib.metadata
import inspect
import platform
from dataclasses import dataclass, field, is_dataclass

import numpy as np

from hybrid_bo.acquisition import AcquisitionSolution, Scenarios
from hybrid_bo.gp import FittedGP
from hybrid_bo.history import History
from hybrid_bo.problem import Problem, _as_text
from hybrid_bo.settings import MethodSettings


@dataclass(frozen=True)
class Run:
    """One run of a method: the evaluated points, the fitted surrogate models, the
    acquisition problems' solutions, the method's settings and the run's artifact,
    which says, as text, how they were produced.

    Holds only data (arrays, numbers, strings and frozen dataclasses of them), never
    the problem or any user callable, so it saves as JSON (`hb.save_run`) and loads
    without the code that produced it (ADR 0003).
    `method` is the method's name. `surrogates` has one fitted surrogate model per
    iteration, the one behind that iteration's trial point: a tuple of one fitted GP
    per unknown state, in the order of the system's y, for hybrid BO, and a 1-tuple of
    the GP of f for standard BO. `acquisitions` has the solution of that iteration's
    acquisition problem; both are empty for the baselines. `scenarios` are hybrid BO's,
    the same in every iteration, and None for the other methods. `settings` are the BO
    method's settings in full, which the iteration plots replay, and None for the
    baselines; the artifact holds them as text, for people. So every saved run has the
    same fields.
    """

    history: History
    artifact: dict[str, str]
    method: str
    surrogates: list[tuple[FittedGP, ...]] = field(default_factory=list)
    acquisitions: list[AcquisitionSolution] = field(default_factory=list)
    scenarios: Scenarios | None = None
    settings: MethodSettings | None = None


def _version(distribution: str) -> str:
    """The installed version of a package, or "unknown" when it isn't installed."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def problem_artifact(problem: Problem, measure: object) -> dict[str, str]:
    """The problem's variables, structure and equations, and the measurement, as text.

    A measurement that is a dataclass (the oracle simulation is) is described by its
    repr, any other by its docstring when it has one, else by its name.
    """
    system = problem.system
    blocks = (system.x, system.y, system.u)
    name = getattr(measure, "__qualname__", type(measure).__qualname__)
    description = repr(measure) if is_dataclass(measure) else inspect.getdoc(measure)
    return {
        "u": "; ".join(repr(var) for var in system.u),
        "x": "; ".join(repr(var) for var in system.x),
        "y": "; ".join(repr(var) for var in system.y),
        "h_inputs": ", ".join(var.name for var in system.h_inputs),
        "measured": ", ".join(var.name for var in system.measured),
        "g": _as_text(system.g, *blocks),
        "f": _as_text(problem.f, *blocks),
        "measure": description or name,
    }


def _generator_artifact(rng: np.random.Generator) -> dict[str, str]:
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


def _versions_artifact() -> dict[str, str]:
    """The versions of hybrid_bo, Python and the numerical packages."""
    return {
        "hybrid_bo": _version("hybrid-bo"),
        "python": platform.python_version(),
        "casadi": _version("casadi"),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
    }


def run_artifact(
    problem: Problem,
    measure: object,
    rng: np.random.Generator,
    method: str,
    settings: str,
) -> dict[str, str]:
    """How a run is produced, as text: the problem's variables, structure and
    equations, the measurement, the method and its settings, the generator and the
    package versions.

    Call it before the method draws anything, so the generator's state is the one
    the run starts from. The measurement is described as `problem_artifact` says.
    """
    return (
        problem_artifact(problem, measure)
        | {"method": method, "settings": settings}
        | _generator_artifact(rng)
        | _versions_artifact()
    )
