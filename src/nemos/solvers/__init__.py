from ._svrg import SVRG, ProxSVRG
from ._svrg_defaults import (
    glm_softplus_poisson_l_max_and_l,
    svrg_optimal_batch_and_stepsize,
)
from ._optimistix_solvers import (
    BFGS,
    NonlinearCG,
    # GradientDescent
)
from ._optax_based_solvers import (
    LBFGS,
    # ProximalGradient,
    GradientDescent,
)

# from ._optimistix_prox_grad import ProximalGradient

from ._prox_grad_learning_rate_from_optax import ProximalGradient
