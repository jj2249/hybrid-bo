"""The library before the restructure. New code never imports from here (DESIGN.md)."""

from hybrid_bo.legacy import (
    affine_transformers,
    array_operations,
    kernels,
    means,
    optimizers,
    parameterization,
    routines,
    type_aliases,
)
from hybrid_bo.legacy.config import Config
from hybrid_bo.legacy.gp import GP
from hybrid_bo.legacy.problem import Problem
