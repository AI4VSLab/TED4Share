import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights

def get_baseline_model(pretrained=True, model_architecture="resnet50"):
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

    # Add more architectures if needed
    raise ValueError(f"Unsupported architecture {model_architecture}")
