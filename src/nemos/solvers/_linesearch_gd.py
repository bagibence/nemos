from collections.abc import Callable

import optimistix as optx


class LSGradientDescent(optx.AbstractGradientDescent):
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    descent: optx.AbstractDescent = optx.SteepestDescent()
    search: optx.AbstractSearch = optx.BacktrackingArmijo()
