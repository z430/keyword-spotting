""" Training """
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.python.keras.utils.conv_utils import conv_input_length

from utils.config import ConfigParser

AUTOTUNE = tf.data.experimental.AUTOTUNE


def main() -> None:
    config = ConfigParser("./data/config.yaml")
    hyperparameter = conv_input_length("./data/hyp.yaml")

    print(f"Input Shape: {input_shape}, len labels: {num_labels}")

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ]
    )

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
        ],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)

    history = model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=config.epochs,
        callbacks=[early_stopping, WandbCallback()],
    )

    # test the model
    az = model.evaluate(testing_ds)
    print(az)


if __name__ == "__main__":
    main()
