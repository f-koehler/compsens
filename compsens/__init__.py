import numpy
import cvxpy


def compute_dt(t):
    dt = numpy.unique(numpy.diff(t))
    if len(numpy.unique(dt)) > 1:
        raise RuntimeError("time points are not equidistant: " + str(dt))
    return dt[0]


def sanity_check(t, w_min, w_max):
    dt = compute_dt(t)
    t_final = max(t)

    dt_required = 2 * numpy.pi / (3 * w_max)
    t_final_required = 2 * numpy.pi / w_min

    # check temporal resolution
    if dt >= dt_required:
        raise RuntimeError("temporal spacing too large: dt={}≮{}".format(
            dt, dt_required))

    # check signal length
    if t_final < t_final_required:
        raise RuntimeError("final time too small: t_final={}≯{}".format(
            t_final, t_final_required))


def compsens1d(t, signal, w_max, dw, method="bp", noise=1e-10, verbose=True):
    # check for valid method
    method = method.lower()
    if method not in ["bp", "qp", "ls"]:
        raise RuntimeError(
            "invalid method {}, choose from: bp, qp, and ls".format(method))

    # create working copies of input
    t = numpy.copy(t)
    signal = numpy.copy(signal)

    # sort by time
    permutation = t.argsort()
    t = t[permutation]
    signal = signal[permutation]

    # compute number of times/frequencies
    N_t = len(t)
    N_w = int(round(w_max / dw + 1.))
    dt = compute_dt(t)

    # check if linear system is under-determined
    if N_t >= 2 * N_w:
        raise RuntimeError(
            "only N_t={} time points allowed for these parameters".format(
                2 * N_w - 1))

    # build the A matrix
    eta = dt * dw
    index_products = numpy.outer(numpy.arange(0, N_t), numpy.arange(0, N_w))
    A = numpy.block(
        [numpy.cos(eta * index_products),
         numpy.sin(eta * index_products)])

    # compute f/||f||_1
    signal_norm = numpy.sum(numpy.abs(signal))
    signal_tilde = signal / signal_norm

    # set up the optimization problem
    variable = cvxpy.Variable(2 * N_w)

    if method == "bp":
        constraints = [cvxpy.norm(A * variable - signal_tilde, 2) <= noise]
        objective = cvxpy.Minimize(cvxpy.norm(variable, 1))
    elif method == "qp":
        constraints = None
        objective = cvxpy.Minimize(
            cvxpy.norm(A * variable - signal_tilde, 2)**2 +
            noise * cvxpy.norm(variable, 1))
    elif method == "sp":
        constraints = [cvxpy.norm(variable) <= noise]
        objective = cvxpy.Minimize(cvxpy.norm(A * variable - signal_tilde, 2))

    problem = cvxpy.Problem(objective, constraints)

    # solve the optimization problem
    problem.solve(verbose=verbose)

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
