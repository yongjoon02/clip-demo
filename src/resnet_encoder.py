# src/resnet_encoder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from collections import OrderedDict

class Bottleneck(nn.Module):
    """
    ResNet Bottleneck 블록 (ResNet-D 변형)
    - Anti-aliasing을 위한 blur pooling 포함
    - 3x3 conv에서 stride=2를 2x2 avg pooling으로 대체
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # ResNet-D 개선사항: 1x1 conv에서 stride 제거
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # 3x3 conv
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # Anti-aliasing: stride=2일 때 blur pooling 사용
        self.avgpool = nn.AvgPool2d(2) if stride == 2 else nn.Identity()

        # 1x1 conv (expansion)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # Shortcut connection
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            # ResNet-D: shortcut에서도 anti-aliasing 적용
            self.downsample = nn.Sequential(OrderedDict([
                ("pool", nn.AvgPool2d(2) if stride == 2 else nn.Identity()),
                ("conv", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("bn", nn.BatchNorm2d(planes * self.expansion))
            ]))

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # Anti-aliasing pooling
        if self.stride == 2:
            out = self.avgpool(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AttentionPool2d(nn.Module):
    """
    CLIP의 Attention Pooling
    - Global Average Pooling 대신 attention 메커니즘 사용
    - 위치별 가중 평균으로 더 풍부한 표현 학습
    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class ModifiedResNet(nn.Module):
    """
    CLIP의 수정된 ResNet
    - ResNet-D 개선사항 적용
    - Attention Pooling 사용
    - Anti-aliasing을 위한 blur pooling
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # Stem (초기 conv 레이어들)
        # ResNet-D: 7x7 conv를 3개의 3x3 conv로 교체
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AvgPool2d(2)

        # ResNet 레이어들
        self._inplanes = width  # 내부 상태 추적
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        # Attention Pooling (Global Average Pooling 대신)
        embed_dim = width * 32  # width * 8 * Bottleneck.expansion
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

class ResNetEncoder(nn.Module):
    """
    CLIP ResNet 인코더 래퍼
    - ModifiedResNet + L2 정규화
    - 다양한 ResNet 크기 지원
    """
    
    def __init__(self, 
                 model_name: str = "RN50", 
                 embed_dim: int = 512,
                 input_resolution: int = 224):
        super().__init__()
        
        # ResNet 구성 정의
        resnet_configs = {
            "RN50": {
                "layers": [3, 4, 6, 3],
                "width": 64,
                "heads": embed_dim // 64,
            },
            "RN101": {
                "layers": [3, 4, 23, 3],
                "width": 64,
                "heads": embed_dim // 64,
            },
            "RN50x4": {
                "layers": [4, 6, 10, 6],
                "width": 80,
                "heads": embed_dim // 64,
            },
            "RN50x16": {
                "layers": [6, 8, 18, 8],
                "width": 96,
                "heads": embed_dim // 64,
            },
            "RN50x64": {
                "layers": [3, 15, 36, 10],
                "width": 128,
                "heads": embed_dim // 64,
            }
        }
        
        if model_name not in resnet_configs:
            raise ValueError(f"Unknown ResNet model: {model_name}")
            
        config = resnet_configs[model_name]
        
        self.model = ModifiedResNet(
            layers=config["layers"],
            output_dim=embed_dim,
            heads=config["heads"],
            input_resolution=input_resolution,
            width=config["width"]
        )
        
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 이미지 텐서
        Returns:
            (B, embed_dim) L2 정규화된 임베딩
        """
        features = self.model(x)
        # L2 정규화 (CLIP과 동일)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

# 편의를 위한 팩토리 함수들
def resnet50(embed_dim: int = 512, input_resolution: int = 224) -> ResNetEncoder:
    """ResNet-50 기반 CLIP 인코더"""
    return ResNetEncoder("RN50", embed_dim, input_resolution)

def resnet101(embed_dim: int = 512, input_resolution: int = 224) -> ResNetEncoder:
    """ResNet-101 기반 CLIP 인코더"""
    return ResNetEncoder("RN101", embed_dim, input_resolution)

def resnet50x4(embed_dim: int = 512, input_resolution: int = 224) -> ResNetEncoder:
    """ResNet-50x4 기반 CLIP 인코더 (더 넓은 채널)"""
    return ResNetEncoder("RN50x4", embed_dim, input_resolution) 