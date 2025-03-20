import optax
from optax.tree_utils import tree_add, tree_sub
import optax._src.base as optax_base


def prox_lasso(regularizer_strength: float):
    def _prox_lasso(regularizer_strength: float, scaling):
        del scaling

        def init_fn(params):
            del params
            return optax.EmptyState()

        def update_fn(updates, state, params, *, scaling):
            if params is None:
                raise ValueError(optax_base.NO_PARAMS_MSG)

            def soft_threshold(x, threshold):
                return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)

            W_prev, b_prev = params
            dW, db = updates

            W_new = tree_add(W_prev, dW)
            b_new = tree_add(b_prev, db)

            # apply soft-threshold on the weights only
            threshold = regularizer_strength / b_prev.shape[0] * scaling
            W_new = jax.tree.map(
                lambda p: soft_threshold(p, threshold),
                W_new,
                is_leaf=lambda x: x is None,
            )

            # updates are new params - old params
            W_update = tree_sub(W_new, W_prev)
            b_update = tree_sub(b_new, b_prev)

            new_updates = (W_update, b_update)

            return new_updates, state

        return optax.GradientTransformation(init_fn, update_fn)

    return _prox_lasso(regularizer_strength, None)


def prox_ridge(regularizer_strength: float):
    def _prox_ridge(regularizer_strength: float, scaling):
        del scaling

        def init_fn(params):
            del params
            return optax.EmptyState()

        def update_fn(updates, state, params, *, scaling):
            if params is None:
                raise ValueError(optax_base.NO_PARAMS_MSG)

            def ridge_shrink(x, alpha_reg):
                # alpha_reg = alpha * lambda
                scale = 1.0 / (1.0 + alpha_reg)
                return scale * x

            W_prev, b_prev = params
            dW, db = updates

            W_new = tree_add(W_prev, dW)
            b_new = tree_add(b_prev, db)

            # apply soft-threshold on the weights only
            l2_reg = regularizer_strength / b_prev.shape[0] * scaling
            W_new = jax.tree.map(
                lambda p: ridge_shrink(p, l2_reg),
                W_new,
                is_leaf=lambda x: x is None,
            )

            # updates are new params - old params
            W_update = tree_sub(W_new, W_prev)
            b_update = tree_sub(b_new, b_prev)

            new_updates = (W_update, b_update)

            return new_updates, state

        return optax.GradientTransformation(init_fn, update_fn)

    return _prox_ridge(regularizer_strength, None)


def prox_chain(
    *args: optax.GradientTransformation,
) -> optax.GradientTransformationExtraArgs:
    # transforms = [optax_base.with_extra_args_support(t) for t in args]
    transforms = args
    init_fns, update_fns = zip(*transforms)

    def init_fn(params):
        return tuple(fn(params) for fn in init_fns)

    def update_fn(updates, state, params=None, **extra_args):
        new_state = []
        for s, fn in zip(state[:-1], update_fns[:-1]):
            updates, new_s = fn(updates, s, params, **extra_args)
            new_state.append(new_s)

        # the last transformation is the proximal operator
        updates, new_s = update_fns[-1](
            updates,
            state[-1],
            params,
            # scaling=new_state[-1],
            scaling=new_state[-1].learning_rate,
        )
        new_state.append(new_s)

        return updates, tuple(new_state)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def scale_by_learning_rate(
    stepsize: float, flip_sign: bool = True
) -> optax.GradientTransformation:
    m = -1 if flip_sign else 1

    def init_fn(params):
        del params
        return ScaleByLearningRateState(jnp.array(stepsize))

    def update_fn(updates, state, params=None, **extra_args):
        del params
        updates = jax.tree.map(lambda g: m * stepsize * g, updates)

        return updates, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


class ProximalGradient(optx.OptaxMinimiser, OptimistixSolverMixin):
    fun: Callable
    fun_with_aux: Callable

    stats: dict

    def __init__(
        self,
        fun: Callable,
        rtol: float,
        atol: float,
        prox: optax.GradientTransformation,
        stepsize: Optional[float] = None,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        verbose: frozenset[str] = frozenset(),
        linesearch_kwargs: Optional[dict] = None,
    ):
        self.fun = fun
        self.fun_with_aux = lambda params, args: (fun(params, args), None)

        self.stats = {}

        if prox.__name__ == "prox_lasso":
            prox = prox_lasso
        elif prox.__name__ == "prox_ridge":
            prox = prox_ridge

        _optax_proxgrad = prox_chain(
            optax.sgd(1.0, nesterov=True),
            self._make_rate_scaler(stepsize, linesearch_kwargs),
            prox,
        )

        super().__init__(
            optim=_optax_proxgrad,
            rtol=rtol,
            atol=atol,
            norm=norm,
            verbose=verbose,
        )

    @staticmethod
    def _make_rate_scaler(
        stepsize: float | None,
        linesearch_kwargs: dict[str, Any] | None,
    ):
        if stepsize is None:
            if linesearch_kwargs is None:
                linesearch_kwargs = {
                    # "approx_dec_rtol": None,  # setting this to none might be useful
                }

            if "max_linesearch_steps" not in linesearch_kwargs:
                linesearch_kwargs["max_linesearch_steps"] = 15

            return optax.scale_by_zoom_linesearch(**linesearch_kwargs)
        else:
            return scale_by_learning_rate(stepsize)

    def run(
        self,
        init_params,
        hyperparams_prox,
        *args,
        options: Optional[dict[str, Any]] = None,
        has_aux: bool = False,
        max_steps: Optional[int] = 100_000,
        adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint(),
        throw: bool = True,
        tags: frozenset[object] = frozenset(),
    ):
        del hyperparams_prox
        return super().run(
            init_params,
            *args,
            options=options,
            has_aux=has_aux,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=tags,
        )

    def update(
        self,
        params,
        state,
        hyperparams_prox,
        *xy_args,
        options: Optional[dict[str, Any]] = None,
        tags: frozenset[object] = frozenset(),
    ):
        del hyperparams_prox
        return super().update(params, state, *xy_args, options=options, tags=tags)
