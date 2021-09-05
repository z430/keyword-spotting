from typing import Tuple
import tensorflow as tf


def decode_audio(audio_name: str) -> Tuple[tf.Tensor, tf.Tensor]:

    """Decode audio from tf
    The audio samples already normalized [-1, 1]

    Args:
        audio_name (str): path of audio

    Returns:
        Tuple[tf.float32, tf.int32]: the audio samples and the sample rate
    """
    audio_binary = tf.io.read_file(audio_name)
    waveform, sample_rate = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(waveform, axis=-1)
    return waveform, sample_rate


def audio_transform(audio_name: str, audio_label: str) -> tf.float32:
    """Read audio and load the audio file for training

    Args:
        audio_name (str): path of audio
        audio_label (str): the label of the audio

    Returns:
        tf.float32: transformed audio with fixed length
    """
    # read audio with tf
    waveform, sr = decode_audio(audio_name=audio_name)


if __name__ == "__main__":
    audio_path = "../../data/train/backward/0165e0e8_nohash_0.wav"
    label = "backward"
    audio_transform(audio_path, label)
