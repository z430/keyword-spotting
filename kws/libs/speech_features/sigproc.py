"""Signal processing functions for speech feature extraction.

This module provides PyTorch implementations of common signal processing
functions used for speech feature extraction, originally from the
python_speech_features package.
"""

import torch
import math
from typing import Tuple, Union
import numpy as np


def framesig(
    signal: torch.Tensor,
    frame_len: int,
    frame_step: int,
    winfunc: callable = torch.hann_window,
) -> torch.Tensor:
    """Frame a signal into overlapping frames.

    Args:
        signal: The audio signal to frame of shape (N,).
        frame_len: The length of each frame in samples.
        frame_step: The step between successive frames in samples.
        winfunc: The window function to apply to each frame.

    Returns:
        torch.Tensor: Windowed frames of shape (num_frames, frame_len).
    """
    signal_len = signal.shape[0]
    if signal_len <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + math.ceil((signal_len - frame_len) / frame_step)

    # Pad the signal to ensure we get the right number of frames
    pad_len = (num_frames - 1) * frame_step + frame_len
    pad_signal = torch.zeros(pad_len, device=signal.device, dtype=signal.dtype)
    pad_signal[:signal_len] = signal

    # Extract the frames
    indices = torch.arange(0, frame_len, device=signal.device).unsqueeze(0)
    frame_indices = torch.arange(
        0, num_frames * frame_step, frame_step, device=signal.device
    ).unsqueeze(1)
    indices_frames = indices + frame_indices
    frames = pad_signal[indices_frames]

    # Apply window function
    win = winfunc(frame_len, device=signal.device)
    return frames * win


def magspec(frames: torch.Tensor, NFFT: int) -> torch.Tensor:
    """Compute the magnitude spectrum of each frame.

    Args:
        frames: The audio frames of shape (num_frames, frame_len).
        NFFT: The FFT size to use.

    Returns:
        torch.Tensor: Magnitude spectrum of shape (num_frames, NFFT//2 + 1).
    """
    if frames.shape[1] > NFFT:
        frames = frames[:, :NFFT]
    elif frames.shape[1] < NFFT:
        pad = torch.zeros(
            frames.shape[0],
            NFFT - frames.shape[1],
            device=frames.device,
            dtype=frames.dtype,
        )
        frames = torch.cat([frames, pad], dim=1)

    complex_spec = torch.fft.rfft(frames, n=NFFT, dim=1)
    return torch.abs(complex_spec)


def powspec(frames: torch.Tensor, NFFT: int) -> torch.Tensor:
    """Compute the power spectrum of each frame.

    Args:
        frames: The audio frames of shape (num_frames, frame_len).
        NFFT: The FFT size to use.

    Returns:
        torch.Tensor: Power spectrum of shape (num_frames, NFFT//2 + 1).
    """
    return 1.0 / NFFT * torch.square(magspec(frames, NFFT))


def preemphasis(signal: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """Apply preemphasis filter to the signal.

    Args:
        signal: The audio signal to filter.
        coeff: The preemphasis coefficient.

    Returns:
        torch.Tensor: The filtered signal.
    """
    return torch.cat([signal[:1], signal[1:] - coeff * signal[:-1]])
