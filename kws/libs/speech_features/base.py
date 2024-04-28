from typing import Optional

import torch

from . import sigproc


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


def mfcc(
    signal: torch.Tensor,
    samplerate: int = 16000,
    winlen: float = 0.025,
    winstep: float = 0.01,
    numcep: int = 13,
    nfilt: int = 26,
    nfft: Optional[bool] = None,
    lowfreq: int = 0,
    highfreq: Optional[int] = None,
    preemph: float = 0.97,
    ceplifter: int = 22,
    appendEnergy: bool = True,
):

    pass


def fbank(
    signal: torch.Tensor,
    samplerate: int = 16000,
    winlen: float = 0.025,
    winstep: float = 0.01,
    nfilt: int = 26,
    nfft: int = 512,
    lowfreq: int = 0,
    highfreq: Optional[int] = None,
    preemph: float = 0.97,
):
    def winfunc(x):
        return torch.ones((x,))

    highfreq = highfreq or samplerate // 2
    signal = sigproc.preempashis(signal, preemph)
    frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate)
