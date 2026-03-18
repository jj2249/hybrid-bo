from typing import TypeAlias

from casadi import DM, MX, SX
from numpy import ndarray

# Options: SX, MX
SymbolicType: TypeAlias = SX

CasadiType: TypeAlias = SX | MX | DM

NumericType: TypeAlias = ndarray | float | int
