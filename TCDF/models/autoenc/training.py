import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from . import autoenc
# import autoenc

def setup_autoenc_training(
    encoder,
    decoder,
    model_dir
):
    autoencoder = autoenc.AutoencoderKL(encoder, decoder)

    gpu_ids = [2]
    accelerator = "gpu" if (len(gpu_ids) > 0) else "cpu"
    devices = gpu_ids if (accelerator == "gpu") else 1
    early_stopping = pl.callbacks.EarlyStopping(
        "val_rec_loss", patience=50, verbose=True
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_rec_loss:.4f}",
        monitor="val_rec_loss",
        every_n_epochs=1,
        save_top_k=3
    )
    callbacks = [early_stopping, checkpoint]
    wandb_logger = WandbLogger(project="autoenc", name="11-26_1515")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1000,
        strategy='ddp' if (len(gpu_ids) > 1) else "auto",
        callbacks=callbacks,
        # logger=wandb_logger
    )

    return (autoencoder, trainer)
