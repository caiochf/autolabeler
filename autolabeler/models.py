from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models


class Backbone(nn.Module):
    """
    Backbone wrapper that outputs an embedding vector (features) instead of logits.

    Supported backbones:
      - resnet50
      - vgg16
      - efficientnet_v2_s
      - densenet121
      - convnext_tiny
    """

    def __init__(self, name: str, device: torch.device):
        super().__init__()
        self.name = name.lower()
        self.device = device

        if self.name == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = m.fc.in_features
            m.fc = nn.Identity()  # remove classifier
            self.net = m

        elif self.name == "vgg16":
            m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            # VGG16 has: features -> avgpool -> classifier
            # We remove the last Linear layer (4096 -> 1000), keeping a 4096-D embedding.
            self.net = nn.Sequential(
                m.features,
                m.avgpool,
                nn.Flatten(),
                *list(m.classifier.children())[:-1],
            )
            self.feature_dim = 4096
            
        elif self.name == "densenet121":
            m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            # DenseNet: classifier is Linear(num_features -> 1000)
            self.feature_dim = m.classifier.in_features
            m.classifier = nn.Identity()
            self.net = m

        elif self.name in ["convnext_tiny", "convnext-tiny"]:
            m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            # ConvNeXt: classifier is (LayerNorm, Flatten, Linear)
            # Easiest is to replace classifier with Identity and use model output as embedding.
            # In torchvision, convnext forward returns logits; removing classifier makes it return features.
            self.feature_dim = m.classifier[2].in_features  # Linear input dim
            m.classifier = nn.Identity()
            self.net = m


        elif self.name in ["efficientnetv2_s", "efficientnet_v2_s", "efficientnetv2-s"]:
            m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
            self.feature_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()  # remove classifier
            self.net = m

        else:
            raise ValueError(f"Unknown backbone name: {name}")

        self.net.eval().to(self.device)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, 3, H, W)
        returns: embeddings of shape (B, D)
        """
        emb = self.net(x)
        if emb.ndim > 2:
            emb = torch.flatten(emb, start_dim=1)
        return emb

