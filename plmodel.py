from typing import Any
import torch
from model import UNet
from pytorch_lightning import LightningModule


class SegmentModel(LightningModule):

    def __init__(self, 
                 n_classes : int, 
                 n_pts : int):

        super().__init__()
        
        self.model = UNet(n_pts, n_classes)

    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y_hat = self.model(x)

        loss = torch.nn.functional.cross_entropy(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        y_hat = self.model(x)

        loss = torch.nn.functional.cross_entropy(y_hat, y)

        return loss

    def configure_optimizers(self):

        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        return [opt], [scheduler]
