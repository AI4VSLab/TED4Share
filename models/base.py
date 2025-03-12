import lightning.pytorch as pl
import torch
from torch import optim
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassPrecision, MulticlassF1Score, MulticlassAUROC, MulticlassPrecisionRecallCurve
)
from torchmetrics.classification.accuracy import MulticlassAccuracy
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)
        targets_prob = prob.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        focal_loss = -self.alpha * ((1 - targets_prob) ** self.gamma) * \
                     log_prob.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BaseNet(pl.LightningModule):
    def __init__(self, classes, lr, encoder, wd=0, momentum=0, optimizer="AdamW", **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.lr = lr
        self.wd = wd
        self.optimizer = optimizer
        self.classes = classes
        self.encoder = encoder

        self.train_ac = MulticlassAccuracy(num_classes=len(classes))
        self.val_ac = MulticlassAccuracy(num_classes=len(classes))
        self.test_ac = MulticlassAccuracy(num_classes=len(classes))

        self.train_p = MulticlassPrecision(num_classes=len(classes))
        self.val_p = MulticlassPrecision(num_classes=len(classes))
        self.test_p = MulticlassPrecision(num_classes=len(classes))

        self.train_f1 = MulticlassF1Score(num_classes=len(classes))
        self.val_f1 = MulticlassF1Score(num_classes=len(classes))
        self.test_f1 = MulticlassF1Score(num_classes=len(classes))

        self.train_auc = MulticlassAUROC(num_classes=len(classes))
        self.val_auc = MulticlassAUROC(num_classes=len(classes))
        self.test_auc = MulticlassAUROC(num_classes=len(classes))

        self.train_pr = MulticlassPrecisionRecallCurve(num_classes=len(classes))
        self.val_pr = MulticlassPrecisionRecallCurve(num_classes=len(classes))
        self.test_pr = MulticlassPrecisionRecallCurve(num_classes=len(classes))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.encoder(x)

    def _calculate_loss(self, batch):
        imgs, labels = batch
        preds = self(imgs)
        loss = F.cross_entropy(preds, labels)
        return {"loss": loss, "preds": preds, "labels": labels}

    def training_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.log("train_loss", output["loss"], on_epoch=True, prog_bar=True)
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.log("val_loss", output["loss"], on_epoch=True, prog_bar=True)
        return output["loss"]

    def test_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.log("test_loss", output["loss"], on_epoch=True, prog_bar=True)
        return output["loss"]
