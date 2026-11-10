"""Prior means: frozen dataclasses that hold their own math."""

from dataclasses import dataclass
from typing import Any

import casadi as cas


@dataclass(frozen=True)
class Zero:
    """The zero prior mean."""

    def at(self, X: Any) -> Any:
        """The prior mean at the rows of the CasADi matrix X (n, d), (n, 1)."""
        return cas.DM.zeros(X.shape[0], 1)
