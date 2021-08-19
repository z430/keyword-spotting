""" Training """
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from models import models
from utils.dataset import LitKeywordSpotting


def main():

    pl.seed_everything(52, workers=True)

    datadir = "../data/train"
    wanted_words = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]

    # define the model
    lit_data = LitKeywordSpotting(
        root_dir=datadir, wanted_words=wanted_words, batch_size=64
    )
    len_classes = lit_data.total_clases
    input_size = lit_data.input_size

    net = models.LinearModel(input_size=input_size[1:], len_classes=len_classes)

    wandb_logger = WandbLogger(project="Keyword Spotting", log_model="all")

    early_stopping = EarlyStopping("val_loss")
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="my/path/",
        filename="sample-mnist-epoch{epoch:02d}-val_loss{val/loss:.2f}",
        auto_insert_metric_name=False,
        save_top_k=2,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[early_stopping, checkpoint_callback],
        deterministic=True,
        weights_save_path="checkpoints/",
        gpus=-1,
        max_epochs=100,
    )
    wandb_logger.watch(net)
    trainer.fit(net, lit_data)


if __name__ == "__main__":
    main()
