import torch
from model import UNet
from pytorch_lightning import LightningModule


class PlSegmentModel(LightningModule):

    def __init__(self, 
                 n_classes : int, 
                 n_pts : int):

        super().__init__()
        
        self.model = UNet(n_pts, n_classes)
        self.save_hyperparameters()

    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        imgs, gt_confidence = batch
        pred_confidence = self.model(imgs)

        loss = torch.nn.functional.cross_entropy(pred_confidence, gt_confidence)

        return loss

    def validation_step(self, batch, batch_idx):
        
        imgs, gt_confidence = batch
        pred_confidence = self.model(imgs)

        loss = torch.nn.functional.cross_entropy(pred_confidence, gt_confidence)

        return loss

    def configure_optimizers(self):

        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        return [opt], [scheduler]
