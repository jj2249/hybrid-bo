from typing import TypeAlias

from casadi import DM, MX, SX
from numpy import ndarray

CasadiType: TypeAlias = SX | MX | DM
SymbolicType: TypeAlias = SX | MX
NumericType: TypeAlias = ndarray | float | int
MathArray: TypeAlias = SX | ndarray
