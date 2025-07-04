"""
This module is maintained for backward compatibility.
New code should use kws.libs.audio_processor instead.
"""

from pathlib import Path
from loguru import logger

import numpy as np
import random

from kws.libs.audio_processor import (
    BACKGROUND_NOISE_DIR,
    SILENCE_INDEX,
    AudioConfig,
    AudioProcessor,
)


class AudioProcessor:
    def __init__(self, dataset_path: Path, config: AudioConfig):
        self.config = config
        self.dataset_path = dataset_path
        self.background_data = self._load_background_data()

    def _load_background_data(self) -> list[np.ndarray]:
        """Load background noise data."""
        background_path = self.dataset_path / BACKGROUND_NOISE_DIR
        if not background_path.exists():
            logger.warning(f"Background noise directory not found: {background_path}")
            return []

        background_data = []
        for wav_path in background_path.glob("*.wav"):
            try:
                audio, _ = librosa.load(wav_path, sr=self.config.sample_rate)
                background_data.append(audio)
            except Exception as e:
                logger.error(f"Error loading background noise file {wav_path}: {e}")

        if not background_data:
            logger.warning(f"No background noise files found in {background_path}")

        return background_data

    def transform(self, filepath: str, label: int) -> np.ndarray:
        """Apply audio transformations including time shifting and background noise."""
        audio, _ = librosa.load(filepath, sr=self.config.sample_rate)
        # Fix audio length
        audio = librosa.util.fix_length(audio, size=self.config.desired_samples)
        # audio = psf.base.sigproc.preemphasis(audio)

        # Handle silence
        if label == SILENCE_INDEX:
            audio = np.zeros_like(audio)

        # Apply time shifting
        audio = self._apply_time_shift(audio)

        # Add background noise
        if self.config.use_background_noise or label == SILENCE_INDEX:
            audio = self._add_background_noise(audio, label, self.background_data)

        # transform to feature
        audio = self.to_mfcc(
            audio, winlen=self.config.frame_length, winstep=self.config.frame_step
        )
        return audio

    def _apply_time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply random time shifting to audio."""
        time_shift_amount = np.random.randint(
            -self.config.time_shift, self.config.time_shift
        )
        padding = [max(0, time_shift_amount), max(0, -time_shift_amount)]
        offset = max(0, -time_shift_amount)

        padded_audio = np.pad(audio, padding, "constant")
        return padded_audio[offset : offset + self.config.desired_samples]

    def _add_background_noise(
        self, audio: np.ndarray, label: int, background_data: list[np.ndarray]
    ) -> np.ndarray:
        """Add background noise to audio."""
        if not background_data:
            return audio

        background_samples = random.choice(background_data)
        background_clip = background_samples[
            np.random.randint(
                0, len(background_samples) - self.config.desired_samples
            ) :
        ][: self.config.desired_samples]

        # background_offset = np.random.randint(
        #     0, len(background_samples) - self.config.desired_samples
        # )
        # background_clip = background_samples[
        #     background_offset : background_offset + self.config.desired_samples
        # ]

        # Determine background volume
        if label == SILENCE_INDEX:
            background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < self.config.background_frequency:
            background_volume = np.random.uniform(0, self.config.background_volume)
        else:
            background_volume = 0

        return audio + (background_clip * background_volume)

    def to_mfcc(
        self,
        signal,
        samplerate=16000,
        numcep=10,
        nfft=512,
        winlen=0.025,
        winstep=0.01,
        nfilt=26,
    ):
        nfft = max(512, int(winlen * samplerate))
        return psf.mfcc(
            signal,
            samplerate=samplerate,
            numcep=numcep,
            nfft=nfft,
            winlen=winlen,
            winstep=winstep,
            nfilt=nfilt,
        )
