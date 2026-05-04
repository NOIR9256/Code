import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

class ForecastModel(pl.LightningModule):
    def __init__(self, model, criterion, lr):
        super(ForecastModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


from ..diffusion import diffusion


def setup_genforecast_training(
    model,
    autoencoder,
    context_encoder,
    model_dir,
    lr=1e-4
):
    ldm = diffusion.LatentDiffusion(model, autoencoder, 
        context_encoder=context_encoder, lr=lr)

    # num_gpus = torch.cuda.device_count() - 2
    # accelerator = "gpu" if (num_gpus > 0) else "cpu"
    # # devices = torch.cuda.device_count() if (accelerator == "gpu") else 1
    # devices = num_gpus if (accelerator == "gpu") else 1
    gpu_ids = [0]
    accelerator = "gpu" if (len(gpu_ids) > 0) else "cpu"
    devices = gpu_ids if (accelerator == "gpu") else 1

    early_stopping = pl.callbacks.EarlyStopping(
        "val_loss_ema", patience=50, verbose=True, check_finite=False
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_loss_ema:.4f}",
        monitor="val_loss_ema",
        every_n_epochs=1,
        save_top_k=3
    )
    callbacks = [early_stopping, checkpoint]
    wandb_logger = WandbLogger(project="ldcast",name="11-19_1546")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=500,
        strategy='ddp' if (len(gpu_ids) > 1) else "auto",
        callbacks=callbacks,
        # logger=wandb_logger,
        #precision=16
    )

    return (ldm, trainer)
