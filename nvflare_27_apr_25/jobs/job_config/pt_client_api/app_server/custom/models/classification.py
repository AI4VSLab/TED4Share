import torch
import torch.nn as nn
from models.base import BaseNet, FocalLoss
from .nt_xnet_loss import NT_Xnet_loss, nt_xnet_loss
from util.get_models import get_baseline_model



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
    def __init__(self, feature_dim=2048, step_size=1000, gamma=0.5, loss_type="focal", **kwargs):

        print('----'*20)
        print(kwargs.keys())
        print('----'*20)
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.step_size = step_size
        self.gamma = gamma
        self.loss_type = loss_type
        self.num_classes = len(self.classes)

        # ---------------------------- setup model ----------------------------
        if 'encoder' not in kwargs: 
            self.encoder = get_baseline_model(  **kwargs)
        

        # If feature_dim != 0, attach an MLP
        if feature_dim != 0:
            self.encoder = nn.Sequential(
                self.encoder,          # ResNet50 with fc=Identity -> 2048 features
                MLP(feature_dim, self.num_classes)
            )

        # ---------------------------- setup hook ----------------------------
        self.hook_outputs = []
        def hook_fn(module, input, output):
            self.hook_outputs.append(output)

        # ---------------------------- configure loss function ----------------------------
        if self.loss_type == "focal": self.criterion = FocalLoss()
        elif self.loss_type == "simclr": 
            # NOTE: assume we are using the pretrained resnet50 here
            self.avgpool_hook = self.encoder[0].pooler.register_forward_hook(hook_fn)
            self.criterion = nt_xnet_loss #NT_Xnet_loss()
        else: self.criterion = nn.CrossEntropyLoss()



    def reset_mlp(self):
        """
        Reset the MLP head if you want to fine-tune from scratch or freeze backbone.
        """
        if isinstance(self.encoder, nn.Sequential):
            self.encoder[-1] = MLP(self.feature_dim, self.num_classes)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # clear hook outputs
        self.hook_outputs.clear()

    def _compute_nt_xnet_loss(self, batch):
        
        # batch['img'] shape [32, 2, 3, 224, 224]
        imgs_1, imgs_2 = batch['img'][:,0], batch['img'][:,1]

        _ = self.forward(imgs_1)
        _ = self.forward(imgs_2)

        z_1 = self.hook_outputs[0].squeeze(-1).squeeze(-1)
        z_2 = self.hook_outputs[1].squeeze(-1).squeeze(-1)
        loss = self.criterion(z_1, z_2, 0.1, flattened = True)

        return {"loss": loss, "preds": -1, "labels": batch['label']}


    def _calculate_loss(self, batch):
        if self.loss_type == "simclr":
            return self._compute_nt_xnet_loss(batch)
        if self.loss_type == 'mae':
            return {'loss': self.encoder.forward(batch['img']).loss}

        imgs, labels = batch['img'], batch['label']
        preds = self.forward(imgs)
        loss = self.criterion(preds, labels)
        return {"loss": loss, "preds": preds, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        if self.loss_type == "simclr": return 0
        else: output = self._calculate_loss(batch)

        self.log("val_loss", output["loss"], on_epoch=True, prog_bar=True)
        return output["loss"]
