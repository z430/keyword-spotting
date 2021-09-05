from typing import List
import numpy as np
import tensorflow as tf
import pandas as pd

from input_data import GetData


class TFDataLoader:
    def __init__(self, data: GetData) -> None:
        self.data = data
        print(self.data)

    def read_audio(self, filename: str, label: int) -> List[tf.float32, str]:
        wave = tf.py_function(
            self.data.audio_transform, [filename, label], [tf.float32]
        )
        wave = tf.convert_to_tensor(wave)
        wave = tf.squeeze(wave, axis=0)
        return wave, label

    def load_dataset(self, dataset: pd.DataFrame) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((dataset["file"], dataset["label"]))
        ds = ds.map(self.read_audio, num_parallel_calls=AUTOTUNE)
        return ds

    def get_data(self, mode: str):
        """[summary]

        Args:
            mode (str): [description]

        Returns:
            [type]: [description]
        """
        if mode not in ["training", "validation", "testing"]:
            print(f"ERROR: {mode} not fount")
            return

        files = self.data.get_datafiles(mode)
        files = pd.DataFrame(files)
        processed_ds = self.load_dataset(files)
        return processed_ds
