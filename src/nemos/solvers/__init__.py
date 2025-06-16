from ._optimistix_solvers import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_MAX_STEPS

from ._svrg import SVRG, ProxSVRG
from ._svrg_defaults import (
    glm_softplus_poisson_l_max_and_l,
    svrg_optimal_batch_and_stepsize,
)

# Purely Optimistix-based
from ._optimistix_solvers import (
    BFGS,
    NonlinearCG,
    # GradientDescent,
    OptimistixLBFGS,
)
# from ._optimistix_prox_grad import ProximalGradient

# Optimistix using Optax solvers
from ._optax_based_solvers import (
    LBFGS,
    GradientDescent,
)
from ._prox_grad_learning_rate_from_optax import ProximalGradient

# from ._prox_grad_learning_rate_within_optax import ProximalGradient


# JAXopt using Optax
from ._jaxopt_optax_solvers import JaxoptOptaxLBFGS

# Purely JAXopt
from ._jaxopt_solvers import (
    JaxoptProximalGradient,
    JaxoptGradientDescent,
    JaxoptLBFGS,
    JaxoptBFGS,
)

# Purely Optax
from ._optax_lbfgs import OptaxLBFGS
from ._optax_prox_grad import OptaxProximalGradient
