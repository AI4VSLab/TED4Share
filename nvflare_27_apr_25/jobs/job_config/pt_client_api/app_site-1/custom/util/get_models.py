import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from transformers import ResNetForImageClassification, ViTForImageClassification, ResNetModel, ResNetConfig, AutoImageProcessor, ViTMAEForPreTraining


MODELS_CLASSIFICATION = {
        'microsoft/resnet-18': ResNetForImageClassification.from_pretrained("microsoft/resnet-18"),
        'microsoft/resnet-50': ResNetForImageClassification.from_pretrained("microsoft/resnet-50"),
        'google/vit-base-patch16-224': ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'),
        'facebook/deit-tiny-patch16-224': ViTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224'),
        'WinKawaks/vit-small-patch16-224': ViTForImageClassification.from_pretrained('WinKawaks/vit-small-patch16-224')
}

MODELS_BACKBONE = {
        'microsoft/resnet-18': ResNetModel.from_pretrained("microsoft/resnet-18"),
        'microsoft/resnet-50': ResNetModel.from_pretrained("microsoft/resnet-50")
}


class get_pretrained_model_hugging_face:
    def __new__(cls, pretrained_model_name = ''):
        # Assuming MODELS_CLASSIFICATION is defined elsewhere
        return MODELS_BACKBONE[pretrained_model_name]


class HFModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).pooler_output.squeeze(-1).squeeze(-1)

class HF_MAE_Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits



def get_baseline_model(pretrained=True, model_architecture="resnet50", **kwargs):
    """
    Returns a ResNet50 (or other model) with the final fc replaced by Identity
    so the output is (batch_size, 2048).
    """
    if model_architecture.lower() == "resnet50":
        if pretrained:
            # For torchvision >=0.13, use explicit weights:
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)

        # Remove the final classification layer (2048 -> 1000)
        model.fc = nn.Identity()
        
        # Now the output dimension is 2048, matching "feature_dim=2048".
        return model
    
    elif model_architecture == 'microsoft/resnet-50':
        if pretrained: model = get_pretrained_model_hugging_face(pretrained_model_name=model_architecture)
        else:
            config = ResNetConfig.from_pretrained("microsoft/resnet-50")
            # Initialize the model from config (random weights)
            model = ResNetModel(config)

        return HFModel(model)

    elif model_architecture == 'microsoft/resnet-18':
        if pretrained: 
            model = get_pretrained_model_hugging_face(pretrained_model_name=model_architecture)
        else:
            config = ResNetConfig.from_pretrained("microsoft/resnet-18")
            # Initialize the model from config (random weights)
            model = ResNetModel(config)
        return HFModel(model)
    
    elif model_architecture == 'facebook/vit-mae-base':
        mae_model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
        processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')

        if 'use_for_classification' in kwargs and kwargs['use_for_classification']: 
            classification_config = mae_model.config
            classification_config.num_labels = len(kwargs['classes'])  # Change this to match your number of classes
            classification_model = ViTForImageClassification(config=classification_config)

            # Step 4: Transfer encoder (ViT) weights from MAE to classification model
            classification_model.vit.load_state_dict(mae_model.vit.state_dict())
            mae_model = HF_MAE_Model(classification_model)
        return mae_model

    # Add more architectures if needed
    raise ValueError(f"Unsupported architecture {model_architecture}")
