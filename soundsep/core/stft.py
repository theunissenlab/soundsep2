import numpy as np
from scipy.fft import fft, fftfreq, next_fast_len


def create_gaussian_window(half_window: int, nstd: float):
    """Create a gaussian window

    Returns gaussian window with array size 2*half_window.

    [ half_window | half_window ]
    [             _             ]
    [          .'   '.          ]
    [         .       .         ]
    [......-'           '-......]

                  --- 1 std
                  ------ 2 std
                  --------- 3 std


    The steepness of the decline on each side is determined
    by nstd

    :param half_window: Half width of the gaussian window
    :type half_window: int
    :param nstd: Number of standard deviations of the gaussian the window should
        cover. Larger means that the edges will fall more quickly, weighting points
        closer to the center. Smaller values of nstd means the fall-off will be
        more gradual and edge points will be weighted more cloesly to the center.
    :type nstd: float


    """
    t = np.arange(-half_window, half_window + 1)
    std = 2.0 * float(half_window) / float(nstd)
    return (
        np.exp(-t**2 / (2.0 * std**2))
        / (std * np.sqrt(2 * np.pi))
    )


def spectral_derivative(spec):
    dy, dx = np.gradient(np.abs(spec))
    theta = np.arctan2(dy, dx)
    dspec = np.abs(dy) * np.sin(theta) + np.abs(dx) * np.cos(theta)

    return dspec
