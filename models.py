"""MNIST classifiers to attack (same architectures as ai-hw-fall-2026-nn).

Every model takes RAW pixels in [0, 1] and normalises internally, so the
attacks work directly in pixel space: epsilon = 0.1 means "each pixel may
change by at most 0.1 (about 25 grey levels out of 255)".
"""
import torch
import torch.nn as nn

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class Normalize(nn.Module):
    def forward(self, x):
        return (x - MNIST_MEAN) / MNIST_STD


class MLP(nn.Module):
    """Shallow MLP: 784 -> 256 -> 10."""

    def __init__(self, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Two conv blocks (conv-BN-ReLU x2 + max-pool), 28 -> 14 -> 7, then dense."""

    def __init__(self, c1=32, c2=64, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, c1), block(c1, c2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TransformerEncoderNet(nn.Module):
    """Tiny ViT: 16 patches of 7x7 -> 64-d tokens + [CLS] + position
    embeddings -> 4 Transformer encoder layers -> classify from [CLS]."""

    def __init__(self, patch=7, dim=64, depth=4, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.n_patches = (28 // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)      # (B, 16, dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


ARCHS = {"mlp": MLP, "cnn": CNN, "transformer": TransformerEncoderNet}


def build_model(arch):
    """Architecture wrapped with normalisation: input = raw pixels in [0, 1]."""
    if arch not in ARCHS:
        raise ValueError(f"Unknown model '{arch}'. Choose from {list(ARCHS)}")
    return nn.Sequential(Normalize(), ARCHS[arch]())


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
