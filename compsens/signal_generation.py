import numpy


def generate_time_points(dt, t_final):
    return numpy.linspace(0., t_final, int(round(t_final / dt + 1)))


def clean_signal(amplitude=1., frequency=1., phase=0.):
    return lambda t: amplitude * numpy.cos(frequency * t + phase)


def multiple_clean_signals(amplitudes=[1.], frequencies=[1.], phases=[0.]):
    def signal(t):
        s = numpy.zeros_like(t)
        for a, w, p in zip(amplitudes, frequencies, phases):
            s += a * numpy.cos(w * t + p)
        return s

    return signal
