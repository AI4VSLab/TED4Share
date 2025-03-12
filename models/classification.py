import torch
import torch.nn as nn
from models.base import BaseNet, FocalLoss

class MLP(nn.Module):
    def __init__(self, in_features, num_classes, dropout_rate=0.2):
        super().__init__()
        self.dense1 = nn.Linear(in_features, in_features * 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(in_features * 2, in_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.dense1(x)))
        x = self.dropout2(self.relu2(self.dense2(x)))
        return self.output_layer(x)

class ClassificationNet(BaseNet):
    def __init__(self, feature_dim, step_size=1000, gamma=0.5, loss_type="focal", **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.step_size = step_size
        self.gamma = gamma
        self.loss_type = loss_type
        self.num_classes = len(self.classes)

        # If feature_dim != 0, attach an MLP
        if feature_dim != 0:
            self.encoder = nn.Sequential(
                self.encoder,          # ResNet50 with fc=Identity -> 2048 features
                MLP(feature_dim, self.num_classes)
            )

        # Choose loss
        if self.loss_type == "focal":
            self.criterion = FocalLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def reset_mlp(self):
        """
        Reset the MLP head if you want to fine-tune from scratch or freeze backbone.
        """
        if isinstance(self.encoder, nn.Sequential):
            self.encoder[-1] = MLP(self.feature_dim, self.num_classes)

    def _calculate_loss(self, batch):
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = self.criterion(preds, labels)
        return {"loss": loss, "preds": preds, "labels": labels}
