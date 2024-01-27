import numpy as np


def preempashis(signal: np.ndarray, coeff: float = 0.95) -> np.ndarray:
    """Apply preempashis to the input signal

    Parameters
    ----------
    signal : np.ndarray
        The signal to filter
    coeff : float
        The preemphasis coefficient. 0 is no filter, default is 0.95.

    Returns
    -------
    np.ndarray
        the filtered signal
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])
