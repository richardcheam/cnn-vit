from __future__ import annotations

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, int, int] = (64, 128, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)

        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(c3, num_classes)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        features = self.activation(self.bn3(self.conv3(x)))
        pooled = self.avgpool(features).flatten(1)
        return features, pooled

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features, pooled = self.forward_features(x)
        logits = self.classifier(self.dropout(pooled))
        if return_features:
            return logits, features
        return logits
