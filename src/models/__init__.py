"""Model factory."""
from src.models.baseline import MobileNetV3Baseline
from src.models.proposed_model import MobileNetV3FrequencyAttention


def create_model(name, pretrained=False):
    if name == "baseline":
        return MobileNetV3Baseline(pretrained=pretrained)
    if name == "frequency_attention":
        return MobileNetV3FrequencyAttention(pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")
