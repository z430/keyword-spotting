"""Audio processing module for feature extraction and augmentation."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import librosa
import numpy as np
import python_speech_features as psf
from loguru import logger

from kws.common.errors import AudioProcessingError, handle_error


BACKGROUND_NOISE_DIR = "_background_noise_"
SILENCE_INDEX = 0


@dataclass
class AudioConfig:
    """Configuration for audio processing.

    This class defines parameters for audio processing including:
    - Background noise addition
    - Time shifting
    - Sample rate and duration
    - Feature extraction parameters
    """

    # Background noise parameters
    background_volume: float = 0.1
    background_frequency: float = 0.8
    use_background_noise: bool = True

    # Basic audio parameters
    sample_rate: int = 16000
    clip_duration_ms: int = 1000
    time_shift_ms: float = 100

    # Feature extraction parameters
    frame_length: float = 0.025
    frame_step: float = 0.01
    num_cepstral_coeffs: int = 10
    num_mel_filters: int = 26
    fft_size: int = 512

    @property
    def time_shift(self) -> int:
        """Calculate time shift in samples."""
        return int((self.time_shift_ms * self.sample_rate) / 1000)

    @property
    def desired_samples(self) -> int:
        """Calculate desired number of samples based on duration."""
        return int(self.sample_rate * self.clip_duration_ms / 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "background_volume": self.background_volume,
            "background_frequency": self.background_frequency,
            "use_background_noise": self.use_background_noise,
            "sample_rate": self.sample_rate,
            "clip_duration_ms": self.clip_duration_ms,
            "time_shift_ms": self.time_shift_ms,
            "frame_length": self.frame_length,
            "frame_step": self.frame_step,
            "num_cepstral_coeffs": self.num_cepstral_coeffs,
            "num_mel_filters": self.num_mel_filters,
            "fft_size": self.fft_size,
        }


class AudioProcessor:
    """Audio processing class for feature extraction and augmentation.

    This class handles:
    - Loading and preprocessing audio files
    - Applying augmentation (time shifting, background noise)
    - Feature extraction (MFCC)
    """

    def __init__(self, dataset_path: Path, config: AudioConfig):
        """Initialize the audio processor.

        Args:
            dataset_path: Path to the dataset containing audio files
            config: Audio processing configuration

        Raises:
            AudioProcessingError: If there's an error initializing the processor
        """
        self.config = config
        self.dataset_path = dataset_path
        try:
            self.background_data = self._load_background_data()
        except Exception as e:
            handle_error(
                e, AudioProcessingError, "Failed to initialize audio processor"
            )

    def _load_background_data(self) -> List[np.ndarray]:
        """Load background noise data.

        Returns:
            List of background noise audio arrays

        Raises:
            AudioProcessingError: If there's an error loading background noise
        """
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
                logger.warning(f"Error loading background noise file {wav_path}: {e}")

        if not background_data:
            logger.warning(f"No background noise files found in {background_path}")

        return background_data

    def transform(self, filepath: str, label: int) -> np.ndarray:
        """Apply audio transformations and extract features.

        Args:
            filepath: Path to the audio file
            label: Label index for the audio file

        Returns:
            Feature array (MFCC)

        Raises:
            AudioProcessingError: If there's an error processing the audio
        """
        try:
            # Load audio file
            audio, _ = librosa.load(filepath, sr=self.config.sample_rate)

            # Fix audio length
            audio = librosa.util.fix_length(audio, size=self.config.desired_samples)

            # Handle silence
            if label == SILENCE_INDEX:
                audio = np.zeros_like(audio)

            # Apply time shifting
            audio = self._apply_time_shift(audio)

            # Add background noise
            if self.config.use_background_noise or label == SILENCE_INDEX:
                audio = self._add_background_noise(audio, label, self.background_data)

            # Transform to feature
            audio_features = self.extract_features(
                audio, winlen=self.config.frame_length, winstep=self.config.frame_step
            )
            return audio_features

        except Exception as e:
            raise AudioProcessingError(
                f"Error processing audio file {filepath}: {str(e)}"
            ) from e

    def _apply_time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply random time shifting to audio.

        Args:
            audio: Audio signal array

        Returns:
            Time-shifted audio array
        """
        time_shift_amount = np.random.randint(
            -self.config.time_shift, self.config.time_shift
        )
        padding = [max(0, time_shift_amount), max(0, -time_shift_amount)]
        offset = max(0, -time_shift_amount)

        padded_audio = np.pad(audio, padding, "constant")
        return padded_audio[offset : offset + self.config.desired_samples]

    def _add_background_noise(
        self, audio: np.ndarray, label: int, background_data: List[np.ndarray]
    ) -> np.ndarray:
        """Add background noise to audio.

        Args:
            audio: Audio signal array
            label: Label index
            background_data: List of background noise arrays

        Returns:
            Audio with added background noise
        """
        if not background_data:
            return audio

        # Select random background noise
        background_samples = random.choice(background_data)

        # Ensure we have enough samples for the clip
        if len(background_samples) <= self.config.desired_samples:
            # If the background is too short, repeat it
            repetitions = int(
                np.ceil(self.config.desired_samples / len(background_samples))
            )
            background_samples = np.tile(background_samples, repetitions)

        # Select random segment from background
        offset = np.random.randint(
            0, len(background_samples) - self.config.desired_samples
        )
        background_clip = background_samples[
            offset : offset + self.config.desired_samples
        ]

        # Determine background volume
        if label == SILENCE_INDEX:
            background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < self.config.background_frequency:
            background_volume = np.random.uniform(0, self.config.background_volume)
        else:
            background_volume = 0

        return audio + (background_clip * background_volume)

    def extract_features(
        self,
        signal: np.ndarray,
        samplerate: int = None,
        numcep: int = None,
        nfft: int = None,
        winlen: float = None,
        winstep: float = None,
        nfilt: int = None,
    ) -> np.ndarray:
        """Extract MFCC features from audio.

        Args:
            signal: Audio signal array
            samplerate: Sample rate (defaults to config value)
            numcep: Number of cepstral coefficients (defaults to config value)
            nfft: FFT size (defaults to config value)
            winlen: Window length in seconds (defaults to config value)
            winstep: Window step in seconds (defaults to config value)
            nfilt: Number of mel filters (defaults to config value)

        Returns:
            MFCC features
        """
        # Use config values if not specified
        samplerate = samplerate or self.config.sample_rate
        numcep = numcep or self.config.num_cepstral_coeffs
        nfft = nfft or self.config.fft_size
        winlen = winlen or self.config.frame_length
        winstep = winstep or self.config.frame_step
        nfilt = nfilt or self.config.num_mel_filters

        # Adjust FFT size if needed
        nfft = max(self.config.fft_size, int(winlen * samplerate))

        return psf.mfcc(
            signal,
            samplerate=samplerate,
            numcep=numcep,
            nfft=nfft,
            winlen=winlen,
            winstep=winstep,
            nfilt=nfilt,
        )
