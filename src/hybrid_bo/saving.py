"""Saving and loading a run as a JSON file (ADR 0003).

`save_run(run, path)` writes every field of the run, and `load_run(path)` reads it
back without the code that produced it; `save_runs` and `load_runs` do the same for a
list of runs per label, as a folder of files. Arrays are nested lists; float64 values
round-trip exactly. A GP's means, kernels and hyperparameters, and the methods'
settings and their parts, are saved with their type's name (`"type"`), so a field that
holds one of several types (kernels, approximations) loads as the right one. The
library never calls them itself.
"""

import dataclasses
import json
import types
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_bo import gp, settings
from hybrid_bo.errors import RunFormatError
from hybrid_bo.run import Run

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

    Handles what runs hold: dataclasses, arrays of floats, lists,
    tuples, dicts with string keys, optional values, unions and plain values. Where a
    tagged type is expected, a tagged dict decodes as the type it names; a union's other
    members are told apart by the data. Raises `RunFormatError` for a tag that
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
            raise RunFormatError(msg)
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

    Raises `RunFormatError` if no member fits.
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
        raise RunFormatError(msg)
    return fits[0]


def save_run(run: Run, path: str | Path) -> Path:
    """Writes `run` to a new file `path` and returns the path. Write to a temporary
    name and rename it, if the file must appear only when complete.

    Raises `FileExistsError` if `path` exists.
    """
    path = Path(path)
    with path.open("x") as file:
        file.write(json.dumps(_encode(run), indent=1) + "\n")
    return path


def load_run(path: str | Path) -> Run:
    """Reads the run saved by `save_run` at `path`.

    Raises `RunFormatError` if the file doesn't hold a run of this version.
    """
    try:
        return _decode(json.loads(Path(path).read_text()), Run)
    except (KeyError, TypeError) as error:
        msg = f"{path} doesn't hold a run of this version: {error!r}"
        raise RunFormatError(msg) from error


def save_runs(runs: Mapping[str, Sequence[Run]], folder: str | Path) -> Path:
    """Writes `runs`, a list of runs per label, to `folder`, as
    `<label>/run-000.json`, and returns it. The folder may exist if it is empty, so a
    script can claim it before its runs start.

    Raises `FileExistsError` if `folder` holds anything.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        msg = f"{folder} isn't empty, so it may hold another study"
        raise FileExistsError(msg)
    for label, label_runs in runs.items():
        (folder / label).mkdir()
        for i, run in enumerate(label_runs):
            save_run(run, folder / label / f"run-{i:03d}.json")
    return folder


def load_runs(folder: str | Path) -> dict[str, list[Run]]:
    """Reads the runs saved by `save_runs` in `folder`, a list of runs per label.

    Raises `RunFormatError` if a file doesn't hold a run of this version.
    """
    labels = sorted(path for path in Path(folder).iterdir() if path.is_dir())
    return {
        label.name: [load_run(path) for path in sorted(label.glob("run-*.json"))]
        for label in labels
    }
