import torch
import torch.nn as nn
from polarmae.layers.masking import MaskedLayerNorm


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MlpLN(nn.Module):
    # MLP + LayerNorm
    def __init__(
        self,
        embed_dim: int,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.ln = MaskedLayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.ln(x, mask)
        x = self.fc2(x)
        return x