"""MobileNetV3-Small ImageNet transfer-learning baseline."""
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.config import NUM_CLASSES


class MobileNetV3Baseline(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.network = mobilenet_v3_small(weights=weights)
        in_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, x):
        return self.network(x)
