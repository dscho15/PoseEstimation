from pytorch_lightning import Trainer
from pldatamodule import ICBINDataModule
from model import UNet
from dataset import ICBINDataset

import model
import dataset

import albumentations as A

if __name__ == "__main__":

    n_classes = 2
    n_pts = 12

    pl_datamodule = ICBINDataModule(data_dir="datasets/icbin")
    
    seg_model = UNet(
        n_pts = n_pts,
        n_classes = n_classes
    ).to("cuda")

    trainer = Trainer(
        
    )

    model(crops.to("cuda"))
