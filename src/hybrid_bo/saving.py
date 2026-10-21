"""Saving and loading a study as a folder of JSON files (ADR 0004).

`save(study, path)` writes `path/artifact.json` (the study's name, settings and
artifact, and the list of run files) and one file per run,
`path/runs/<method>/run-000.json`, holding every field of the run. `load` also reads
a study saved before the artifact was named: `study.json` with a `record` key, but not
one saved before GPs were data (ADR 0008) or before runs held their settings. Arrays are
nested lists; float64 values round-trip exactly. A GP's means, kernels and
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
    sequence type for a list, the one dataclass for a dict, else the plain type.

    Raises `StudyFormatError` if no member fits.
    """
    members = tuple(member for member in members if member is not type(None))
    if isinstance(value, list):
        kinds = (list, tuple, np.ndarray)
        fits = [m for m in members if (typing.get_origin(m) or m) in kinds]
    elif isinstance(value, dict):
        fits = [m for m in members if dataclasses.is_dataclass(m)]
    else:
        fits = [m for m in members if isinstance(value, typing.get_origin(m) or m)]
    if not fits:
        msg = f"The saved value {value!r} is none of {members}"
        raise StudyFormatError(msg)
    return fits[0]


def _check_format(run_data: dict[str, Any], run_path: Path) -> None:
    """Raises `StudyFormatError` if a saved run predates GPs as data (its fitted GPs
    hold named hyperparameters and no GP, or, with no iterations, its record holds
    the GP's kernel as text), or is a BO run (its record has the prior GP) saved
    before runs held their settings."""
    surrogates = run_data.get("surrogates", [])
    has_old_surrogates = any("gp" not in surrogate for surrogate in surrogates)
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


def _run_path(run: Run) -> Path:
    """Where a run is saved, relative to the study's folder.

    Raises `ValueError` if the run's method label can't be a folder name.
    """
    has_separator = "/" in run.method or "\\" in run.method
    if has_separator or run.method in ("", ".", ".."):
        msg = f"The method label {run.method!r} must be a folder name to be saved"
        raise ValueError(msg)
    return Path("runs", run.method, f"run-{run.index:03d}.json")


def save(study: Study, path: str | Path) -> Path:
    """Writes `study` to a new folder `path` (see the module docs) and returns it.

    Raises `FileExistsError` if `path` exists, and `ValueError` if a method label
    can't be a folder name or two runs share a method and an index; nothing is
    written then.
    """
    # Every run's file, checked before anything is written
    path = Path(path)
    run_paths = [_run_path(run) for run in study.runs]
    if len(set(run_paths)) < len(run_paths):
        msg = "Two runs share a method and an index, so they would share a file"
        raise ValueError(msg)
    path.mkdir(parents=True)

    # One file per run, then the artifact, which lists them in order
    for run, run_path in zip(study.runs, run_paths, strict=True):
        (path / run_path).parent.mkdir(parents=True, exist_ok=True)
        (path / run_path).write_text(json.dumps(_encode(run), indent=1) + "\n")
    study_data = {
        "name": study.name,
        "settings": _encode(study.settings),
        "artifact": study.artifact,
        "runs": [run_path.as_posix() for run_path in run_paths],
    }
    (path / ARTIFACT_FILE).write_text(json.dumps(study_data, indent=1) + "\n")
    return path


def load(path: str | Path) -> Study:
    """Reads the study saved in the folder `path` by `save`.

    Raises `StudyFormatError` for a study saved before GPs were data or before runs
    held their settings.
    """
    # The artifact, else the older `study.json`, which named it `record`
    path = Path(path)
    if (path / ARTIFACT_FILE).exists():
        study_data = json.loads((path / ARTIFACT_FILE).read_text())
    else:
        study_data = json.loads((path / OLD_ARTIFACT_FILE).read_text())
        study_data["artifact"] = study_data.pop("record")

    # The runs, in the order the artifact lists them
    runs = []
    for run_path in study_data["runs"]:
        run_data = json.loads((path / run_path).read_text())
        _check_format(run_data, path / run_path)
        runs.append(_decode(run_data, Run))
    return Study(
        tuple(runs),
        _decode(study_data["settings"], StudySettings),
        study_data["name"],
        study_data["artifact"],
    )
