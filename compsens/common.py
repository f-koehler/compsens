import numpy


def check_sanity(t, w_min, w_max):
    dt = compute_dt(t)
    t_final = max(t)

    dt_required = 2 * numpy.pi / (3 * w_max)
    t_final_required = 2 * numpy.pi / w_min

    # check temporal resolution
    if dt >= dt_required:
        raise ValueError("temporal spacing too large: dt={}≮{}".format(
            dt, dt_required))

    # check signal length
    if t_final <= t_final_required:
        raise ValueError("final time too small: t_final={}≯{}".format(
            t_final, t_final_required))


def check_underdetermined(N_t, N_w):
    if N_t >= 2 * N_w:
        raise RuntimeError(
            "only N_t={} time points allowed for these parameters".format(
                2 * N_w - 1))


def compute_A_matrix(N_t, N_w, dt, dw):
    eta = dt * dw
    index_products = numpy.outer(numpy.arange(0, N_t), numpy.arange(0, N_w))
    return numpy.block(
        [numpy.cos(eta * index_products),
         numpy.sin(eta * index_products)])


def compute_dt(t):
    dt = numpy.unique(numpy.around(numpy.diff(t), 10))
    if len(numpy.unique(dt)) > 1:
        raise ValueError("time points are not equidistant: " + str(dt))
    return dt[0]


def get_method(method):
    # check for valid method
    method = method.upper()
    if method not in ["BP", "QP", "LS"]:
        raise ValueError(
            "invalid method {}, choose from: BP, QP, and LS".format(method))

    return method


def prepare_signal(t, signal):
    # create working copies
    t = numpy.copy(t)
    signal = numpy.copy(signal)

    # sort by time
    permutation = t.argsort()
    t = t[permutation]
    signal = signal[permutation]

    return t, signal
