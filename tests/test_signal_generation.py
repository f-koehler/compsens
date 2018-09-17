from pytest import approx
from compsens.signal_generation import generate_time_points
from compsens.signal_generation import clean_signal
import numpy


def test_generate_time_points():
    t = generate_time_points(0.5, 10.)
    assert t[0] == 0.
    assert len(t) == 21
    assert t[-1] == 10.

    dts = numpy.unique(numpy.diff(t))
    assert len(dts) == 1
    assert dts[0] == 0.5


def test_clean_signal():
    # test cos(x)
    s = clean_signal()
    assert s(0.) == approx(1.)
    assert s(numpy.pi / 2.) == approx(0.)
    assert s(numpy.pi) == approx(-1.)
    assert s(3 * numpy.pi / 2.) == approx(0.)
    assert s(2 * numpy.pi) == approx(1.)

    # test 2*sin(x)
    s = clean_signal(2., 1., -numpy.pi / 2.)
    assert s(0.) == approx(0.)
    assert s(numpy.pi / 2.) == approx(2.)
    assert s(numpy.pi) == approx(0.)
    assert s(3 * numpy.pi / 2.) == approx(-2.)
    assert s(2 * numpy.pi) == approx(0.)

    # test cos(2x)
    s = clean_signal(1., 2., 0.)
    assert s(0.) == approx(1.)
    assert s(numpy.pi / 4.) == approx(0.)
    assert s(numpy.pi / 2.) == approx(-1.)
    assert s(3 * numpy.pi / 4.) == approx(0.)
    assert s(numpy.pi) == approx(1.)
