""" Training """
import tensorflow as tf

from utils import dataloaders
from utils.config import ConfigParser
from utils.input_data import GetData
from models import models

AUTOTUNE = tf.data.experimental.AUTOTUNE


def main() -> None:
    config = ConfigParser("./data/config.yaml")
    hyperparameter = ConfigParser("./data/hyp.yaml")

    # dataset
    data = GetData(wanted_words=config.wanted_words)
    dataloader = dataloaders.SpeechCommandLoader(
        config.sample_rate, config.sample_size, batch_size=1, autotune=AUTOTUNE
    )

    training_dataset = data.transform_df(data.training)
    validation_dataset = data.transform_df(data.validation)

    training_loader = dataloader.to_tf_dataset(training_dataset)
    validation_loader = dataloader.to_tf_dataset(validation_dataset)

    input_shape = dataloader.get_input_shape(training_loader)
    num_labels = len(data.words_list)

    print(f"Input Shape: {input_shape}, len labels: {num_labels}")

    model = models.select_model(input_shape, num_labels, "cnn_trad_fpool3")

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", "categorical_accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)

    history = model.fit(
        training_loader,
        validation_data=validation_loader,
        epochs=config.epochs,
        callbacks=[early_stopping],
    )


if __name__ == "__main__":
    main()
