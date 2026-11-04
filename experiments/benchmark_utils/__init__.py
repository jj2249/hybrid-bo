"""What the Benchopt classes in this folder share: the experiments and their code."""

import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType

import hybrid_bo

FOLDER = Path(__file__).parents[1]
# Benchopt's own file in this folder, which sits beside the experiments
NOT_EXPERIMENTS = {"__init__", "objective"}
MODULES = sorted(
    path.stem for path in FOLDER.glob("*.py") if path.stem not in NOT_EXPERIMENTS
)

# Benchopt imports its classes by path, so the experiments need this to be importable
sys.path.insert(0, str(FOLDER))


def experiment(module: str) -> ModuleType:
    """The experiment in the file `module`.py of this folder."""
    return importlib.import_module(module)


def fingerprint(folder: Path) -> str:
    """A short hash of the Python files in `folder` and the library's source.

    Benchopt's cache only sees its own classes, so it would return old results for a
    changed experiment or library. The hash, as a parameter of the dataset, is part
    of the cache's key, and it records in the results which code produced them. It
    covers every experiment: Benchopt pairs a selected module with each value of
    another parameter, so `code` can have only one value.
    """
    library = Path(hybrid_bo.__file__).parent
    files = sorted(folder.glob("*.py")) + sorted(library.glob("*.py"))
    code = b"".join(file.read_bytes() for file in files)
    return hashlib.sha256(code).hexdigest()[:12]
