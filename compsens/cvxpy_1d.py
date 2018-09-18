from . import common

import numpy
import cvxpy

ECOS_DEFAULTS = {
    "abstol": 1e-7,
    "abstol_inacc": 5e-5,
    "feastol": 1e-6,
    "feastol_inacc": 1e-4,
    "max_iters": 1000,
    "reltol": 1e-5,
    "reltol_inacc": 5e-5
}


def compsens1d_cvxpy(t,
                     signal,
                     dw,
                     w_max,
                     method="BP",
                     noise=1e-8,
                     verbose=True,
                     solver="ECOS",
                     **solver_args):
    # prepare input
    t, signal = common.prepare_signal(t, signal)
    method = common.get_method(method)

    # set default options for solver
    if solver == "ECOS":
        for key in ECOS_DEFAULTS:
            solver_args[key] = solver_args.get(key, ECOS_DEFAULTS[key])

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

    # set up the optimization problem
    variable = cvxpy.Variable(2 * N_w)

    if method == "BP":
        constraints = [cvxpy.norm(A * variable - signal_tilde, 2) <= noise]
        objective = cvxpy.Minimize(cvxpy.norm(variable, 1))
    elif method == "QP":
        raise NotImplementedError
        constraints = None
        objective = cvxpy.Minimize(
            cvxpy.norm(A * variable - signal_tilde, 2)**2 +
            noise * cvxpy.norm(variable, 1))
    elif method == "LS":
        raise NotImplementedError
        constraints = [cvxpy.norm(variable) <= noise]
        objective = cvxpy.Minimize(cvxpy.norm(A * variable - signal_tilde, 2))

    problem = cvxpy.Problem(objective, constraints)

    # solve the optimization problem
    if solver is None:
        problem.solve(verbose=verbose, **solver_args)
    else:
        solver = getattr(cvxpy, solver.upper())
        problem.solve(verbose=verbose, solver=solver, **solver_args)

    # check status of problem
    if problem.status != "optimal":
        raise RuntimeError("optimization problem is {}".format(problem.status))

    # transform the results to the desired format
    coefficients = numpy.sqrt(numpy.pi / 2.) * signal_norm / dw * numpy.array(
        variable.value)
    coefficients = coefficients[0:N_w] + 1j * coefficients[N_w:]
    frequencies = numpy.arange(0, N_w) * dw

    # return results
    return frequencies, coefficients, problem.solver_stats
