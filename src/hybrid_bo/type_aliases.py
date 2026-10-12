from casadi import DM, MX, SX
from numpy import ndarray

# Options: SX, MX
SymbolicType = SX

CasadiType = SX | MX | DM

NumericType = ndarray | float | int
