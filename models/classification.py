import torch
import torch.nn as nn
from models.base import BaseNet, FocalLoss
from .nt_xnet_loss import NT_Xnet_loss, nt_xnet_loss
from util.get_models import get_baseline_model
import os
import pandas as pd
from datetime import datetime



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
    '''
    @params:
        loss_type: focal, simclr, mae

    '''
    def __init__(self, feature_dim=2048, step_size=1000, gamma=0.5, loss_type="focal", **kwargs):

        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.step_size = step_size
        self.gamma = gamma
        self.loss_type = loss_type
        self.num_classes = len(self.classes)
        self.log_dir = kwargs.get('log_dir', '')

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
            #self.avgpool_hook = self.encoder[0].pooler.register_forward_hook(hook_fn)
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

        z_1 = self.forward(imgs_1)
        z_2 = self.forward(imgs_2)

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
        #if self.loss_type == "simclr": return self._calculate_loss(batch)
        #else: output = self._calculate_loss(batch)


        output = self._calculate_loss(batch)

        self.log("val_loss", output["loss"], on_epoch=True, prog_bar=True)
        return output["loss"]

    def test_step(self, batch, batch_idx):
        if self.loss_type == "simclr": return 0
        else: output = self._calculate_loss(batch)

        probs = torch.nn.functional.softmax(output["preds"], dim=1)
        pred_prob, preds = torch.max(probs, dim=1)
        self.pred_probs.extend(pred_prob.cpu().numpy())
        self.labels_list.extend(output["labels"].cpu().numpy())
        self.pred_class.extend(preds.cpu().numpy())


        if self.loss_type == 'focal' or self.loss_type == 'ce':
            prob_cls_1 = torch.softmax(output["preds"], dim=1)[:, 1]
            self.test_acc(prob_cls_1, output["labels"])
            self.test_p(prob_cls_1, output["labels"])
            self.test_f1(prob_cls_1, output["labels"])
            self.conf_mat(output["preds"], output["labels"])
            self.test_recall(prob_cls_1, output["labels"])
            self.test_spec(prob_cls_1, output["labels"]) # needs prob [0,1] like log reg, hence pass prob for class 1, since 0 is for healthy
        self.log("test_loss", output["loss"], on_epoch=True, prog_bar=True)
        return {"loss":output["loss"]}
    
    def on_test_epoch_start(self):
        self.labels_list = []
        self.pred_probs = []
        self.pred_class = []

    def on_test_epoch_end(self):
        acc = self.test_acc.compute()
        p = self.test_p.compute()
        f1 = self.test_f1.compute()
        conf_mat = self.conf_mat.compute()
        recall = self.test_recall.compute()
        spec = self.test_spec.compute()


        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        self.log("test_p", p, on_epoch=True, prog_bar=True)
        self.log("test_f1", f1, on_epoch=True, prog_bar=True)
        
        print(f"Confusion Matrix: \n {conf_mat}")
        

        # ----------- Save test metrics to CSV -----------

        # Create metrics dictionary
        metrics = {
            'accuracy': acc.item(),
            'precision': p.item(),
            'f1_score': f1.item(),
            'recall': recall.item(),
            'specificity': spec.item(),
        }

        # Convert confusion matrix to flat format for saving
        conf_mat_np = conf_mat.cpu().numpy()
        for i in range(conf_mat_np.shape[0]):
            for j in range(conf_mat_np.shape[1]):
                metrics[f'conf_mat_{i}_{j}'] = conf_mat_np[i, j]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # save test summary
        df = pd.DataFrame([metrics])
        csv_path = os.path.join(self.log_dir, f"test_results_summary_{timestamp}.csv")
        df.to_csv(csv_path, index=False)


        # save all test probs
        df_probs = pd.DataFrame({
            'labels': self.labels_list,
            'pred_class': self.pred_class,
            'pred_probs': self.pred_probs,
        })
        csv_path = os.path.join(self.log_dir, f"test_results_{timestamp}.csv")
        df_probs.to_csv(csv_path, index=False)
        
        # clean
        self.test_acc.reset()
        self.test_p.reset()
        self.test_f1.reset()
        self.conf_mat.reset()
        self.test_recall.reset()
        self.test_spec.reset()