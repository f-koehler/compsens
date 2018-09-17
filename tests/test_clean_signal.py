from pytest import approx, raises
import compsens
import numpy
from scipy.signal import find_peaks


def test_compsens1d_errors():
    t = compsens.signal_generation.generate_time_points(0.1, 10.)
    s = compsens.signal_generation.clean_signal()(t)

    # try to use an invalid method
    with raises(ValueError):
        compsens.compsens1d(t, s, 0.1, 1., method="xx")

    # try to create an non-underdetermined equation system
    with raises(RuntimeError):
        compsens.compsens1d(t, s, 0.1, 1.)


def test_clean_signal():
    t = compsens.signal_generation.generate_time_points(0.2, 7.)
    s = compsens.signal_generation.clean_signal()(t)

    def compute_peaks(w, a):
        a = numpy.abs(a)
        a = a / a.max()
        return w[find_peaks(a, height=0.01)[0]]

    # BP
    w, a, _ = compsens.compsens1d(t, s, 0.1, 2., verbose=False, method="bp")
    peaks = compute_peaks(w, a)
    assert len(peaks) == 1
    assert peaks[0] == approx(1.)

    # # QP
    # w, a, _ = compsens.compsens1d(t, s, 0.1, 2., verbose=False, method="qp")
    # peaks = compute_peaks(w, a)
    # assert len(peaks) == 1
    # assert peaks[0] == approx(1.)

    # # LS
    # w, a, _ = compsens.compsens1d(t, s, 0.1, 2., verbose=False, method="ls")
    # peaks = compute_peaks(w, a)
    # assert len(peaks) == 1
    # assert peaks[0] == approx(1.)
