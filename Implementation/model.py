import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SelfAttentionPool(nn.Module):
    # Single-head dot-product attention over spatial locations.
    # Input: B x C x H x W -> Output: B x D
    def __init__(self, in_channels: int, out_dim: int = None):
        super().__init__()
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.out_proj = None
        if out_dim is not None and out_dim != in_channels:
            self.out_proj = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)  # B x HW x C
        k = self.k(x).flatten(2)                  # B x C x HW
        v = self.v(x).flatten(2).transpose(1, 2)  # B x HW x C
        attn = torch.matmul(q, k) / (C ** 0.5)    # B x HW x HW
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)               # B x HW x C
        out = out.mean(dim=1)                     # B x C (global mean over spatial)
        if self.out_proj is not None:
            out = self.out_proj(out)
        return out

class Backbone(nn.Module):
    def __init__(self, name="resnet50", pretrained=True, embedding_dim=512, use_attention=True):
        super().__init__()
        if name.lower() == "resnet50":
            try:
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            except Exception:
                weights = None
            net = models.resnet50(weights=weights)
            feat_dim = 2048
        elif name.lower() == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            except Exception:
                weights = None
            net = models.resnet18(weights=weights)
            feat_dim = 512
        else:
            raise ValueError("Unsupported backbone: " + name)

        self.encoder = nn.Sequential(*list(net.children())[:-2])  # keep conv features
        self.use_attention = use_attention
        if use_attention:
            self.pool = SelfAttentionPool(feat_dim, out_dim=embedding_dim)
            self.out_dim = embedding_dim
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = feat_dim

    def forward(self, x):
        feats = self.encoder(x)                 # B x C x H x W
        if self.use_attention:
            emb = self.pool(feats)             # B x D
        else:
            emb = self.pool(feats).flatten(1)  # B x C
        emb = F.normalize(emb, p=2, dim=1)     # L2 norm helps metric learning
        return emb

def pairwise_distance(a, b, metric="euclidean", inv_cov=None):
    # a: B x D (queries), b: M x D (prototypes)
    if metric == "euclidean":
        a2 = (a * a).sum(dim=1, keepdim=True)
        b2 = (b * b).sum(dim=1, keepdim=True).t()
        ab = a @ b.t()
        return a2 + b2 - 2 * ab
    elif metric == "cosine":
        a_n = F.normalize(a, dim=1)
        b_n = F.normalize(b, dim=1)
        return 1 - a_n @ b_n.t()
    elif metric == "mahalanobis":
        if inv_cov is None:
            raise ValueError("inv_cov required for mahalanobis")
        B, D = a.shape
        M = b.shape[0]
        dists = torch.zeros(B, M, device=a.device)
        if inv_cov.dim() == 2:
            Si = inv_cov  # shared D x D
            for m in range(M):
                diff = a - b[m:m+1, :]
                dists[:, m] = torch.einsum("bd,dd,bd->b", diff, Si, diff)
        else:
            for m in range(M):
                diff = a - b[m:m+1, :]
                dists[:, m] = torch.einsum("bd,dd,bd->b", diff, inv_cov[m], diff)
        return dists
    else:
        raise ValueError("Unknown metric: " + metric)
