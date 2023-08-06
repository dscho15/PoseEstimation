from dataset import ICBINDataset
from model import UNet
from pldatamodule import ICBINDataModule
from plmodel import PlSegmentModel
import albumentations as A

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

if __name__ == "__main__":

    n_classes = 2
    n_pts = 12

    pl_datamodule = ICBINDataModule(
        transform=A.Compose([A.Normalize()]),
        path_to_scenes="datasets/icbin")
    
    train_dataloader = pl_datamodule.train_dataloader()
    validation_dataloader = pl_datamodule.val_dataloader()

    model = PlSegmentModel(
        n_classes=n_classes,
        n_pts=n_pts
    )

    earlystopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
        callbacks=[earlystopping]
    )

    trainer.fit(model, train_dataloader, validation_dataloader)