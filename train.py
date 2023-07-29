from pytorch_lightning import Trainer
from model import UNet
from dataset import ICBINDataset

import model
import dataset

import albumentations as A

if __name__ == "__main__":

    n_classes = 2
    n_pts = 12

    transform = A.Compose([
        A.Normalize(),
    ])

    dataset = ICBINDataset(
        path="datasets/icbin",
        transform=transform
    )

    crops, masks = dataset[0]

    print(crops.shape)
    print(masks.shape)

    model = UNet(
        n_pts = dataset.N_PTS,
        n_classes = dataset.N_CLASSES
    )

    trainer = Trainer()
    trainer.fit(model)