from . import common

import numpy
from scipy.optimize import minimize


def compsens1d_scipy(t, signal, dw, w_max, method="BP", noise=1e-8):
    # prepare input
    t, signal = common.prepare_signal(t, signal)
    method = common.get_method(method)

    # compute number of times/frequencies
    N_t = len(t)
    N_w = int(round(w_max / dw + 1.))
    dt = common.compute_dt(t)
    common.check_underdetermined(N_t, N_w)

    # build the A matrix
    A = common.compute_A_matrix(N_t, N_w, dt, dw)

    # compute f/||f||_1
    signal_norm = numpy.sum(numpy.abs(signal))
    signal_tilde = signal / signal_norm

    # set up optimization problem
    if method == "BP":

        def c(x):
            return noise - numpy.linalg.norm(A @ x - signal_tilde, 2)

        constraints = [{"type": "ineq", "fun": c}]

        def fun(x):
            return numpy.linalg.norm(x, 1)
    elif method == "QP":
        raise NotImplementedError
    elif method == "LS":
        raise NotImplementedError

    # solve the optimization problem
    x0 = numpy.zeros(2 * N_w)
    result = minimize(fun, x0, constraints=constraints)
