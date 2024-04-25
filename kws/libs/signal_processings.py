import numpy as np
import torch


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


def signal2spectrogram(
    signal: torch.Tensor, n_fft: int = 400, hop_length: int = 160, power: float = 2.0
) -> torch.Tensor:
    """Convert a signal to a spectrogram

    Parameters
    ----------
    signal : torch.Tensor
        The input signal
    n_fft : int
        The number of fft to apply, default is 400
    hop_length : int
        The hop length, default is 160
    power : float
        The power of the spectrogram, default is 2.0

    Returns
    -------
    torch.Tensor
        The spectrogram
    """
    spec = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, return_complex=False)
    spec = torch.norm(spec, p=power, dim=-1)
    return spec


def signal2mfcc(
    signal: torch.Tensor, n_fft: int = 400, hop_length: int = 160, n_mfcc: int = 13
) -> torch.Tensor:
    """Convert a signal to a Mel-frequency cepstral coefficients (MFCC)

    Parameters
    ----------
    signal : torch.Tensor
        The input signal
    n_fft : int
        The number of fft to apply, default is 400
    hop_length : int
        The hop length, default is 160
    n_mfcc : int
        The number of mfcc to return, default is 13

    Returns
    -------
    torch.Tensor
        The mfcc
    """
    spec = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, return_complex=False)
    spec = torch.norm(spec, p=2, dim=-1)
    mel_basis = torch.nn.functional.linear(
        spec, torch.nn.functional.hann_window(n_fft), n_mfcc
    )
    return mel_basis
