import lightning.pytorch as pl
import torch
from torch import optim
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryPrecision, BinaryF1Score, MulticlassAUROC, MulticlassPrecisionRecallCurve, MulticlassConfusionMatrix, BinaryRecall, BinarySpecificity
)
from torchmetrics.classification.accuracy import MulticlassAccuracy, BinaryAccuracy
import torch.nn as nn
import torchvision

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
    def __init__(self, classes, lr, encoder = None, wd=0, momentum=0, optimizer="AdamW", **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.lr = lr
        self.wd = wd
        self.optimizer = optimizer
        self.classes = classes
        self.encoder = encoder
        self.warmup_epochs = kwargs.get('warmup_epochs', 0)
        self.epochs = kwargs.get('epochs', 20)
        self.scheduler = kwargs.get('scheduler', 'step')

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.train_p = BinaryPrecision()
        self.val_p = BinaryPrecision()
        self.test_p = BinaryPrecision()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        self.train_auc = MulticlassAUROC(num_classes=len(classes))
        self.val_auc = MulticlassAUROC(num_classes=len(classes))
        self.test_auc = MulticlassAUROC(num_classes=len(classes))

        self.train_pr = MulticlassPrecisionRecallCurve(num_classes=len(classes))
        self.val_pr = MulticlassPrecisionRecallCurve(num_classes=len(classes))
        self.test_pr = MulticlassPrecisionRecallCurve(num_classes=len(classes))

        # recall/sensitivity
        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.test_recall = BinaryRecall()

        # specificity
        self.train_spec = BinarySpecificity()
        self.val_spec = BinarySpecificity()
        self.test_spec = BinarySpecificity()

        self.conf_mat = MulticlassConfusionMatrix(num_classes=len(classes))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        if self.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs-self.warmup_epochs, eta_min=1e-5)
        elif self.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        
        if self.warmup_epochs > 0 :
            # if warmup_epochs is 0, it would simply not do warmups. warmup starts with lr*start_factor and ends with lr
            scheduler2use = [optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=self.warmup_epochs), scheduler]
        
            scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                            schedulers=scheduler2use,
                            milestones=[self.warmup_epochs])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.encoder(x)

    def _calculate_loss(self, batch):
        imgs, labels = batch['img'], batch['label']
        preds = self(imgs)
        loss = F.cross_entropy(preds, labels)

        
        return {"loss": loss, "preds": preds, "labels": labels}

    def training_step(self, batch, batch_idx):

        output = self._calculate_loss(batch)

        
        if self.current_epoch % 10 == 0 and hasattr(self.logger, "experiment"):
            if self.loss_type == 'simclr': imgs_to_log = batch['img'][0,:8]  # log up to 4 images
            else: imgs_to_log = batch['img'][:8]
            grid = torchvision.utils.make_grid(imgs_to_log, nrow=2, normalize=True, scale_each=True)
            self.logger.experiment.add_image("images", grid, self.global_step)

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
