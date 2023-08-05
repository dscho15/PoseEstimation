from pytorch_lightning import Trainer
from pldatamodule import ICBINDataModule
from plmodel import SegmentModel
from model import UNet
from dataset import ICBINDataset

import albumentations as A

if __name__ == "__main__":

    n_classes = 2
    n_pts = 12

    pl_datamodule = ICBINDataModule(
        transform=A.Compose([A.Normalize()]),
        path_to_scenes="datasets/icbin")
        
    train_dataloader = pl_datamodule.train_dataloader()
    validation_dataloader = pl_datamodule.val_dataloader()

    SegmentModel(
        n_classes=n_classes,
        n_pts=n_pts
    )