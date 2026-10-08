"""MobileNetV3-Small features followed by frequency-domain attention."""
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.config import NUM_CLASSES
from src.models.frequency_attention import FrequencyDomainAttention


class MobileNetV3FrequencyAttention(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.features = backbone.features
        channels = backbone.classifier[0].in_features
        self.attention = FrequencyDomainAttention(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(channels, 256),
                                        nn.Hardswish(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES))

    def forward(self, x):
        x = self.attention(self.features(x))
        return self.classifier(self.pool(x))
