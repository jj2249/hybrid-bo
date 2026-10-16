"""Saving and loading a study as a folder of JSON files (ADR 0004).

`save(study, path)` writes `path/artifact.json` (the study's name, settings and
artifact, and the list of run files) and one file per run,
`path/runs/<method>/run-000.json`, holding every field of the run. `load` also reads
a study saved before the artifact was named: `study.json` with a `record` key. Arrays
are nested lists; float64 values round-trip exactly. The library never calls `save`
itself.
"""

import dataclasses
import json
import types
import typing
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_bo.run import Run
from hybrid_bo.studies import Study, StudySettings

# The artifact's file, and its name in studies saved before the artifact was named
ARTIFACT_FILE = "artifact.json"
OLD_ARTIFACT_FILE = "study.json"


def _encode(value: Any) -> Any:
    """`value` as JSON data: a dataclass as a dict of its fields, an array as nested
    lists (an empty one as its shape, which nested lists would lose)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
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
    tuples, dicts with string keys, optional values and plain values.
    """
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if value is None:
        return None
    if origin in (types.UnionType, typing.Union):
        (inner,) = [arg for arg in args if arg is not type(None)]
        return _decode(value, inner)
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
    """Reads the study saved in the folder `path` by `save`."""
    # The artifact, else the older `study.json`, which named it `record`
    path = Path(path)
    if (path / ARTIFACT_FILE).exists():
        study_data = json.loads((path / ARTIFACT_FILE).read_text())
    else:
        study_data = json.loads((path / OLD_ARTIFACT_FILE).read_text())
        study_data["artifact"] = study_data.pop("record")

    # The runs, in the order the artifact lists them
    runs = tuple(
        _decode(json.loads((path / run_path).read_text()), Run)
        for run_path in study_data["runs"]
    )
    return Study(
        runs,
        _decode(study_data["settings"], StudySettings),
        study_data["name"],
        study_data["artifact"],
    )
