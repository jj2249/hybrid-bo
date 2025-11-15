import numpy as np
from sklearn.gaussian_process.kernels import Matern

from hybrid_bo import GP

gp: GP = GP(1, Matern(nu=2.5))
gp.model.n_restarts_optimizer = 10

pass
