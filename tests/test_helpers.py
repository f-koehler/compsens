from compsens.signal_generation import generate_time_points
from pytest import approx, raises
import compsens
import numpy


def test_check_sanity():
    t = generate_time_points(1., 10.)

    # w_min and w_max ok
    compsens.check_sanity(t, 0.75, 1.5)

    # w_max not ok
    with raises(ValueError):
        compsens.check_sanity(t, 0.75, 2.5)

    # w_min not ok
    with raises(ValueError):
        compsens.check_sanity(t, 0.25, 1.5)


def test_compute_dt():
    assert compsens.compute_dt([0, 1., 2., 3.]) == approx(1.)
    assert compsens.compute_dt(numpy.linspace(0., 1., 201)) == approx(0.005)

    with raises(ValueError):
        compsens.compute_dt([1., 2., 4., 5.])
