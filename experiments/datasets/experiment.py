"""Every experiment in this folder, as a Benchopt dataset."""

from typing import Any

from benchmark_utils import FOLDER, MODULES, experiment, fingerprint
from benchopt import BaseDataset


class Dataset(BaseDataset):
    """One experiment. `code` is the hash that keeps Benchopt's cache honest
    (`fingerprint`)."""

    name = "experiment"
    parameters = {"module": MODULES, "code": [fingerprint(FOLDER)]}
    module: str
    code: str

    def get_data(self) -> dict[str, Any]:
        """The experiment, and the code it runs with."""
        return {"experiment": experiment(self.module), "code": self.code}
