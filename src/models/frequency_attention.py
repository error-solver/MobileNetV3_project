"""Lightweight differentiable frequency-guided channel attention."""
import torch
from torch import nn


class FrequencyDomainAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.gate = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, features):
        spectrum = torch.fft.rfft2(features.float(), dim=(-2, -1), norm="ortho")
        magnitude = torch.abs(spectrum)
        descriptor = torch.log1p(magnitude).mean(dim=(-2, -1))
        weights = self.gate(descriptor).to(dtype=features.dtype)
        return features * (1.0 + weights[:, :, None, None])
