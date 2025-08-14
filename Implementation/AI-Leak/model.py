import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SelfAttentionBlock(nn.Module):
    """Channel SE + spatial conv attention."""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.se_fc1 = nn.Linear(in_channels, max(1, in_channels // reduction), bias=True)
        self.se_fc2 = nn.Linear(max(1, in_channels // reduction), in_channels, bias=True)
        self.spatial = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        z = F.adaptive_avg_pool2d(x, 1).view(b, c)
        z = F.relu(self.se_fc1(z))
        z = torch.sigmoid(self.se_fc2(z)).view(b, c, 1, 1)
        x = x * z
        s = torch.sigmoid(self.spatial(x))
        x = x * s
        return x

class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.attn = SelfAttentionBlock(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(2048, embed_dim)

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = self.attn(feats)
        feats = self.pool(feats).flatten(1)
        emb = self.proj(feats)
        emb = F.normalize(emb, dim=-1)
        return emb
