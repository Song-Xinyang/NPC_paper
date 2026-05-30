from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet3D(nn.Module):
    def __init__(self, block=Bottleneck3D, layers: Sequence[int] = (3, 4, 6, 3), in_channels: int = 3, feature_dim: int = 2048):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.feature_dim = feature_dim
        self._init_weights()

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class MultiEndpointCoxResNet(nn.Module):
    def __init__(self, endpoints: Sequence[str] = ("os", "pfs", "dmfs"), in_channels: int = 3):
        super().__init__()
        self.encoder = ResNet3D(in_channels=in_channels)
        self.endpoints = list(endpoints)
        self.heads = nn.ModuleDict({endpoint: nn.Linear(2048, 1) for endpoint in endpoints})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder.forward_features(x)
        return {endpoint: head(feats).squeeze(1) for endpoint, head in self.heads.items()}

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(x)


def cox_partial_likelihood_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood.

    time is assumed in increasing units; higher risk indicates higher hazard.
    """
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order].float()
    log_cumsum_hazard = torch.logcumsumexp(risk, dim=0)
    uncensored = event == 1
    if uncensored.sum() == 0:
        return torch.tensor(0.0, device=risk.device, requires_grad=True)
    loss = -torch.sum(risk[uncensored] - log_cumsum_hazard[uncensored]) / torch.sum(uncensored)
    return loss


def load_medicalnet_weights(model: nn.Module, checkpoint_path: str | None, strict: bool = False) -> None:
    if checkpoint_path is None:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        k2 = k.replace("module.", "")
        if k2.startswith("encoder."):
            k2 = k2[len("encoder."):]
        cleaned[k2] = v
    target = model.encoder if hasattr(model, "encoder") else model
    missing, unexpected = target.load_state_dict(cleaned, strict=strict)
    print("Loaded MedicalNet-style weights.")
