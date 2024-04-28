import decimal
import math

import torch


def round_half_up(number, decimals=0) -> float:
    context = decimal.Context(rounding=decimal.ROUND_HALF_UP)
    return float(
        context.create_decimal(number).quantize(decimal.Decimal(10) ** -decimals)
    )


def rolling_window(a, window, step=1) -> torch.Tensor:
    shape = a.size()[:-1] + (a.size(-1) - window, window)
    strides = a.stride()[:-1] + (a.stride(-1) * step, a.stride(-1))
    return a.as_strided(size=shape, stride=strides)


def framesig(
    sig: torch.Tensor,
    frame_len: float,
    frame_step: float,
    stride_trick: bool = True,
) -> torch.Tensor:
    def winfunc(x):
        return torch.ones((int(x),))

    slen = sig.size(0)
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = torch.zeros((padlen - slen,))
    padsignal = torch.cat((sig, zeros))

    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=int(frame_len), step=int(frame_step))
    else:
        indices = torch.arange(0, frame_len).repeat(numframes, 1) + torch.arange(
            0, int(numframes * frame_step), int(frame_step)
        ).unsqueeze(1).repeat(1, int(frame_len))
        frames = padsignal[indices]
        win = winfunc(frame_len).repeat(numframes, 1)

    return frames * win


def magspec(frames, NFFT) -> torch.Tensor:
    # check if the frame length is greater than the NFFT
    if frames.shape[1] > NFFT:
        print(
            f"Warning: frame length ({frames.shape[1]}) is greater than FFT size ({NFFT}), frame will be truncated. Increase NFFT to avoid."
        )
    # compuse the fft along the last dimension and keep only the non-negative frequencies
    complex_spec = torch.fft.rfft(frames, n=NFFT)
    return torch.abs(complex_spec)


def powspec(frames, NFFT) -> torch.Tensor:
    magnitude_spectrum = magspec(frames, NFFT)
    return (magnitude_spectrum**2) / NFFT


def preempashis(signal: torch.Tensor, coeff: float = 0.95) -> torch.Tensor:
    return torch.cat([signal[0], signal[1:] - coeff * signal[:-1]])
