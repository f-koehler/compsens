from compsens.signal_generation import generate_time_points
from pytest import approx, raises
from compsens import common
import numpy


def test_check_sanity():
    t = generate_time_points(1., 10.)

    # w_min and w_max ok
    common.check_sanity(t, 0.75, 1.5)

    # w_max not ok
    with raises(ValueError):
        common.check_sanity(t, 0.75, 2.5)

    # w_min not ok
    with raises(ValueError):
        common.check_sanity(t, 0.25, 1.5)


def test_check_underdetermined():
    with raises(RuntimeError):
        common.check_underdetermined(20, 10)

    with raises(RuntimeError):
        common.check_underdetermined(20, 8)

    common.check_underdetermined(20, 12)


def test_compute_A_matrix():
    # test shape
    N_t = 29
    N_w = 13
    A = common.compute_A_matrix(N_t, N_w, 0.01, 0.1)
    assert A.shape[0] == N_t
    assert A.shape[1] == 2 * N_w

    # test values
    A = common.compute_A_matrix(3, 3, 0.5 * numpy.pi, 1.)
    assert A[:, 0] == approx(1.)
    assert A[0, :3] == approx(1.)
    assert A[1, 1] == approx(0.)
    assert A[1, 2] == approx(-1.)
    assert A[2, 1] == approx(-1.)
    assert A[2, 2] == approx(1.)

    assert A[:, 3] == approx(0.)
    assert A[0, 3:] == approx(0.)
    assert A[1, 4] == approx(1.)
    assert A[1, 5] == approx(0.)
    assert A[2, 4] == approx(0.)
    assert A[2, 5] == approx(0.)


def test_compute_dt():
    assert common.compute_dt([0, 1., 2., 3.]) == approx(1.)
    assert common.compute_dt(numpy.linspace(0., 1., 201)) == approx(0.005)

    with raises(ValueError):
        common.compute_dt([1., 2., 4., 5.])


def test_get_method():
    assert common.get_method("bp") == "BP"
    assert common.get_method("Qp") == "QP"
    assert common.get_method("LS") == "LS"
    with raises(ValueError):
        common.get_method("XXX")


def test_prepare_signal():
    t = [0., 0.5, -0.5]
    signal = [2, 3, 1]
    t, signal = common.prepare_signal(t, signal)

    assert numpy.all(t == [-0.5, 0., 0.5])
    assert numpy.all(signal == [1, 2, 3])
