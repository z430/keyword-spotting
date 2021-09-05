""" Training """
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import wandb
from wandb.keras import WandbCallback

AUTOTUNE = tf.data.experimental.AUTOTUNE


def main() -> None:
    wandb.init(
        project="tf-keywork-spotting",
        config={
            "learning_rate": 0.005,
            "batch_size": 16,
            "epochs": 1,
            "loss_function": "sparse_categorical_crossentropy",
            "architecture": "simpleCNN",
            "wanted_words": wanted_words,
        },
    )

    config = wandb.config

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
