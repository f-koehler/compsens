import numpy


def generate_time_points(dt, t_final):
    return numpy.linspace(0., t_final, int(round(t_final / dt + 1)))


def clean_signal(amplitude=1., frequency=1., phase=0.):
    return lambda t: amplitude * numpy.cos(frequency * t + phase)
