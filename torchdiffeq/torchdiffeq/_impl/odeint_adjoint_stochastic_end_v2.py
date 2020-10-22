from .tsit5 import Tsit5Solver
from .dopri5 import Dopri5Solver
from .bosh3 import Bosh3Solver
from .adaptive_heun import AdaptiveHeunSolver
from .fixed_grid import Euler, Midpoint, RK4
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .adams import VariableCoefficientAdamsBashforth
from .misc import _check_inputs
import numpy as np
import torch
#from .odeint import odeint
from .adjoint import odeint_adjoint
from torch.distributions import normal
from torch.distributions import uniform
SOLVERS = {
    'explicit_adams': AdamsBashforth,
    'fixed_adams': AdamsBashforthMoulton,
    'adams': VariableCoefficientAdamsBashforth,
    'tsit5': Tsit5Solver,
    'dopri5': Dopri5Solver,
    'bosh3': Bosh3Solver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
    'adaptive_heun': AdaptiveHeunSolver,
}

def odeint_adjoint_stochastic_end_v2(func, y0, actual_t, rtol=1e-6, atol=1e-12, method=None, options=None, shrink_proportion = 0.5, shrink_std = 0.02 , mode='train', min_length=0.01 ):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a Tensor holding the state `y` and a scalar Tensor
            `t` into a Tensor of state derivatives with respect to time.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. May
            have any floating point or complex dtype.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`. The initial time point should be the first element of this sequence,
            and each time must be larger than the previous time. May have any floating
            point dtype. Converted to a Tensor with float64 dtype.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        name: Optional name for this operation.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
        TypeError: if `options` is supplied without `method`, or if `t` or `y0` has
            an invalid dtype.
    """

    t = actual_t.clone()
    if isinstance(y0, tuple):
        integration_time = t.type_as(y0[0])#integration_time.type_as(x)
    else:
        integration_time = t.type_as(y0)

    rev = False

    if t[1]<t[0]:
        t = reverse_time(t)
        rev = True
    range_time = (t[1]-t[0]) * shrink_proportion

    #m = normal.Normal(t[0] + range_time - shrink_std , t[0] + range_time + shrink_std)
    m = uniform.Uniform(t[0] + range_time - shrink_std , t[0] + range_time + shrink_std)

    integration_time[0]=t[0]
    if mode=='train':
        integration_time[1]=max(m.sample(), t[0] + min_length)
    else:
        integration_time[1]= t[0] + range_time

    #print("actual_t")
    #print(actual_t)
    #print("integration_time")
    #print(integration_time)
    #print("=================")


    if rev:
        integration_time = reverse_time(integration_time)


    out = odeint_adjoint( func, y0, integration_time,rtol=rtol,atol=atol,method=method,options=options)
    return out

def reverse_time(t):
    temp1 = t[1].item()
    temp0 = t[0].item()
    t[1] = temp0
    t[0] = temp1
    return t

