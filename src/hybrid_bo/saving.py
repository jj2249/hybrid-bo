"""Saving and loading a study as a folder of JSON files (ADR 0004).

`save(study, path)` writes `path/artifact.json` (the study's name, settings and
artifact, and the list of run files) and one file per run,
`path/runs/<method>/run-000.json`, holding every field of the run. `save_run` and
`load_run` write and read one run file, so a study can be saved run by run as its runs
finish, then completed with `save_artifact`. `load` also reads
a study saved before the artifact was named: `study.json` with a `record` key, but not
one saved before GPs were data (ADR 0008), before runs held their settings or before
runs had an artifact (a run's JSON with `record`, not `artifact`); a run saved with one
fitted GP per iteration, before surrogates were tuples, loads with each as a 1-tuple,
and a study saved before configurations, with one budget in its settings, loads with it
as each run's budget.
Arrays are nested lists; float64 values round-trip exactly. A GP's means, kernels and
hyperparameters, and the methods' settings and their parts, are saved with their type's
name (`"type"`), so a field that holds one of several types (kernels, approximations)
loads as the right one. The library never calls `save` itself.
"""

import dataclasses
import json
import types
import typing
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_bo import gp, settings
from hybrid_bo.errors import StudyFormatError
from hybrid_bo.run import Run
from hybrid_bo.studies import Study, StudySettings

# The artifact's file, and its name in studies saved before the artifact was named
ARTIFACT_FILE = "artifact.json"
OLD_ARTIFACT_FILE = "study.json"

# The types saved with their name, by name: the closed sets of a GP's parts and of the
# methods' settings and their parts
_TAGGED = {cls.__name__: cls for cls in (*gp.TYPES, *settings.TYPES)}


def _encode(value: Any) -> Any:
    """`value` as JSON data: a dataclass as a dict of its fields (with its type's
    name if it is tagged), an array as nested lists (an empty one as its
    shape, which nested lists would lose)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        tagged = _TAGGED.get(type(value).__name__) is type(value)
        tag = {"type": type(value).__name__} if tagged else {}
        return tag | {
            field.name: _encode(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape)} if value.size == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list | tuple):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value: Any, annotation: Any) -> Any:
    """The JSON data `value` as the type `annotation`, the inverse of `_encode`.

    Handles what runs and studies hold: dataclasses, arrays of floats, lists,
    tuples, dicts with string keys, optional values, unions and plain values. Where a
    tagged type is expected, a tagged dict decodes as the type it names; a union's other
    members are told apart by the data. Raises `StudyFormatError` for a tag that
    names no expected type, or data that fits no member of a union.
    """
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if value is None:
        return None

    # A tagged type, by its tag, where the annotation is one or a union holding one
    expected = {annotation, *args}
    if isinstance(value, dict) and "type" in value and expected & {*_TAGGED.values()}:
        tagged = _TAGGED.get(value["type"])
        if tagged not in expected:
            msg = f"A saved {value['type']!r} where {annotation} was expected"
            raise StudyFormatError(msg)
        fields = {key: item for key, item in value.items() if key != "type"}
        return _decode(fields, tagged)
    if origin in (types.UnionType, typing.Union):
        return _decode(value, _union_member(value, args))
    if annotation is np.ndarray:
        if isinstance(value, dict):
            return np.zeros(value["shape"])
        return np.asarray(value, dtype=float)
    if dataclasses.is_dataclass(annotation):
        hints = typing.get_type_hints(annotation)
        return annotation(
            **{key: _decode(item, hints[key]) for key, item in value.items()}
        )
    if origin is list:
        return [_decode(item, args[0]) for item in value]
    if origin is tuple:
        return tuple(_decode(item, args[0]) for item in value)
    if origin is dict:
        return {key: _decode(item, args[1]) for key, item in value.items()}
    return value


def _union_member(value: Any, members: tuple[Any, ...]) -> Any:
    """The member of a union that untagged JSON data `value` decodes as: the
    sequence type for a list, an array for an empty array's `{"shape": ...}`, the
    one dataclass for another dict, else the plain type.

    Raises `StudyFormatError` if no member fits.
    """
    members = tuple(member for member in members if member is not type(None))
    if isinstance(value, list):
        kinds = (list, tuple, np.ndarray)
        fits = [m for m in members if (typing.get_origin(m) or m) in kinds]
    elif isinstance(value, dict) and value.keys() == {"shape"}:
        fits = [m for m in members if m is np.ndarray]
    elif isinstance(value, dict):
        fits = [m for m in members if dataclasses.is_dataclass(m)]
    else:
        fits = [m for m in members if isinstance(value, typing.get_origin(m) or m)]
    if not fits:
        msg = f"The saved value {value!r} is none of {members}"
        raise StudyFormatError(msg)
    return fits[0]


def _with_surrogate_tuples(run_data: dict[str, Any]) -> dict[str, Any]:
    """The saved run `run_data` with each iteration's surrogate a list of fitted GPs:
    a run saved before more than one unknown state has one fitted GP (a dict) per
    iteration, which becomes a list of one."""
    surrogates = [
        [surrogate] if isinstance(surrogate, dict) else surrogate
        for surrogate in run_data.get("surrogates", [])
    ]
    return run_data | {"surrogates": surrogates}


def _check_format(run_data: dict[str, Any], run_path: Path) -> None:
    """Raises `StudyFormatError` if a saved run predates GPs as data (its fitted GPs
    hold named hyperparameters and no GP, or, with no iterations, its `record` holds
    the GP's kernel as text), is a BO run (its `record` has the prior GP) saved
    before runs held their settings, or was saved before runs had an artifact (it
    has `record`, the artifact's old name). Takes the run with its surrogates as
    lists (`_with_surrogate_tuples`)."""
    # The older formats first, for their specific messages: their runs had `record` too
    fitted_gps = [gp for surrogate in run_data["surrogates"] for gp in surrogate]
    has_old_surrogates = any("gp" not in fitted for fitted in fitted_gps)
    if has_old_surrogates or "kernel" in run_data.get("record", {}):
        msg = (
            f"{run_path} was saved before GPs were data (its fitted GPs hold named "
            "hyperparameters, not their GP), so it can't be loaded: rerun the study, "
            "or load it with the version that saved it"
        )
        raise StudyFormatError(msg)
    if "gp" in run_data.get("record", {}) and "settings" not in run_data:
        msg = (
            f"{run_path} was saved before runs held their settings (its record holds "
            "them as text only), so it can't be loaded: rerun the study, or load it "
            "with the version that saved it"
        )
        raise StudyFormatError(msg)
    if "record" in run_data:
        msg = (
            f"{run_path} was saved before runs had an artifact (it has `record`), so "
            "it can't be loaded: rerun the study, or load it with the version that "
            "saved it"
        )
        raise StudyFormatError(msg)


def _run_path(run: Run) -> Path:
    """Where a run is saved, relative to the study's folder.

    Raises `ValueError` if the run's method label can't be a folder name.
    """
    has_separator = "/" in run.method or "\\" in run.method
    if has_separator or run.method in ("", ".", ".."):
        msg = f"The method label {run.method!r} must be a folder name to be saved"
        raise ValueError(msg)
    return Path("runs", run.method, f"run-{run.index:03d}.json")


def _run_text(run: Run) -> str:
    """The run as its run file holds it."""
    return json.dumps(_encode(run), indent=1) + "\n"


def save_run(run: Run, path: str | Path) -> Path:
    """Writes `run` to a new file `path`, as a study's run file, and returns the
    path. Write to a temporary name and rename it, if a run file must appear only when
    complete.

    Raises `FileExistsError` if `path` exists.
    """
    path = Path(path)
    with path.open("x") as file:
        file.write(_run_text(run))
    return path


def load_run(path: str | Path) -> Run:
    """Reads the run saved by `save_run` (or `save`, as a run file of a study) at
    `path`.

    Raises `StudyFormatError` for a run saved before GPs were data, before runs held
    their settings or before runs had an artifact.
    """
    path = Path(path)
    run_data = _with_surrogate_tuples(json.loads(path.read_text()))
    _check_format(run_data, path)
    return _decode(run_data, Run)


def save(study: Study, path: str | Path) -> Path:
    """Writes `study` to a new folder `path` (see the module docs) and returns it.

    Raises `FileExistsError` if `path` exists, and `ValueError` if a method label
    can't be a folder name or two runs share a method and an index; nothing is
    written then.
    """
    # Every run's file, checked before anything is written
    path = Path(path)
    run_paths = _run_paths(study)
    path.mkdir(parents=True)

    # One file per run, then the artifact, which lists them in order
    for run, run_path in zip(study.runs, run_paths, strict=True):
        (path / run_path).parent.mkdir(parents=True, exist_ok=True)
        save_run(run, path / run_path)
    _write_artifact(study, path, run_paths)
    return path


def save_artifact(study: Study, path: str | Path) -> Path:
    """Writes the artifact of `study` (see the module docs) to the folder `path`,
    which already holds every run file `save` would write (e.g. by `save_run`), and
    returns the folder. The folder then loads with `load`, as if `save` had written
    it.

    Raises `FileExistsError` if the folder holds an artifact, `ValueError` if a
    method label can't be a folder name, two runs share a method and an index, or a
    run file is missing or doesn't hold its run; nothing is written then.
    """
    # Every run file in place, holding its run
    path = Path(path)
    run_paths = _run_paths(study)
    if (path / ARTIFACT_FILE).exists():
        msg = f"{path} already holds a study's artifact"
        raise FileExistsError(msg)
    for run, run_path in zip(study.runs, run_paths, strict=True):
        saved = path / run_path
        holds_run = saved.exists() and saved.read_text() == _run_text(run)
        if not holds_run:
            msg = f"{saved} doesn't hold run {run.index} of {run.method}"
            raise ValueError(msg)

    _write_artifact(study, path, run_paths)
    return path


def _run_paths(study: Study) -> list[Path]:
    """Where each run of the study is saved, relative to the study's folder.

    Raises `ValueError` if a method label can't be a folder name or two runs share a
    method and an index.
    """
    run_paths = [_run_path(run) for run in study.runs]
    if len(set(run_paths)) < len(run_paths):
        msg = "Two runs share a method and an index, so they would share a file"
        raise ValueError(msg)
    return run_paths


def _write_artifact(study: Study, path: Path, run_paths: list[Path]) -> None:
    """Writes the study's artifact file to its folder `path`, listing `run_paths`,
    so it appears only when complete."""
    study_data = {
        "name": study.name,
        "settings": _encode(study.settings),
        "artifact": study.artifact,
        "runs": [run_path.as_posix() for run_path in run_paths],
    }
    # To a temporary file first, so a folder holds an artifact only when it is
    # complete: an artifact says the study is finished
    temporary = path / f"{ARTIFACT_FILE}.tmp"
    temporary.write_text(json.dumps(study_data, indent=1) + "\n")
    temporary.replace(path / ARTIFACT_FILE)


def load(path: str | Path) -> Study:
    """Reads the study saved in the folder `path` by `save`.

    Raises `StudyFormatError` for a study saved before GPs were data, before runs
    held their settings or before runs had an artifact.
    """
    # The artifact, else the older `study.json`, which named it `record`
    path = Path(path)
    if (path / ARTIFACT_FILE).exists():
        study_data = json.loads((path / ARTIFACT_FILE).read_text())
    else:
        study_data = json.loads((path / OLD_ARTIFACT_FILE).read_text())
        study_data["artifact"] = study_data.pop("record")

    # The runs, in the order the artifact lists them
    runs = [load_run(path / run_path) for run_path in study_data["runs"]]

    # A study saved before configurations: its settings' one budget is every run's
    settings = dict(study_data["settings"])
    if "n_iterations" in settings:
        n_initial, n_iterations = (
            settings.pop("n_initial"),
            settings.pop("n_iterations"),
        )
        n_evaluations = n_initial + n_iterations
        runs = [dataclasses.replace(run, n_evaluations=n_evaluations) for run in runs]
    return Study(
        tuple(runs),
        _decode(settings, StudySettings),
        study_data["name"],
        study_data["artifact"],
    )
