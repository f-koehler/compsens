from pytest import approx, raises
import compsens
import numpy
from scipy.signal import find_peaks


def test_clean_signal():
    t = compsens.signal_generation.generate_time_points(0.2, 7.)
    s = compsens.signal_generation.clean_signal()(t)

    def compute_peaks(w, a):
        a = numpy.abs(a)
        a = a / a.max()
        return w[find_peaks(a, height=0.01)[0]]

    # BP
    w, a, _ = compsens.compsens1d_cvxpy(
        t, s, 0.1, 2., verbose=False, method="BP")
    peaks = compute_peaks(w, a)
    assert len(peaks) == 1
    assert peaks[0] == approx(1.)


    # QP
    with raises(NotImplementedError):
        compsens.compsens1d_cvxpy(
            t, s, 0.1, 2., verbose=False, method="QP")


    # LS
    with raises(NotImplementedError):
        compsens.compsens1d_cvxpy(
            t, s, 0.1, 2., verbose=False, method="LS")


def test_multiple_clean_signals():
    t = compsens.signal_generation.generate_time_points(0.2, 12.)
    s = compsens.signal_generation.multiple_clean_signals(
        [1., 0.5, 0.25], [1., 1.25, 1.5], [0., 0.1, 0.3])(t)

    def compute_peaks(w, a):
        a = numpy.abs(a)
        a = a / a.max()
        peaks = w[find_peaks(a, height=0.01)[0]]
        peaks.sort()
        return peaks

    # BP
    w, a, _ = compsens.compsens1d_cvxpy(
        t,
        s,
        0.05,
        3.0,
        verbose=True,
        method="bp")
    peaks = compute_peaks(w, a)
    assert len(peaks) == 3
    assert peaks[0] == approx(1.)
    assert peaks[1] == approx(1.25)
    assert peaks[2] == approx(1.5)
