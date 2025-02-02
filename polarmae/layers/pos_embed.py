import torch
import torch.nn as nn


@torch.no_grad()
def get_pos_embed(embed_dim, ipt_pos, scale=1024):
    """
    embed_dim: output dimension for each position
    ipt_pos: [B, G, 3], where 3 is (x, y, z)

    assumes that the points are in the range [-1, 1]
    """
    B, G, _ = ipt_pos.size()
    assert embed_dim % 6 == 0

    # scale ipt_pos from [-1,1] to [0, scale]
    min_val = ipt_pos.reshape(-1, 3).min(dim=0).values
    max_val = ipt_pos.reshape(-1, 3).max(dim=0).values
    ipt_pos = scale * (ipt_pos - min_val) / (max_val - min_val)

    omega = torch.arange(embed_dim // 6).float().to(ipt_pos.device) # NOTE
    omega /= embed_dim / 6.
    # (0-31) / 32
    omega = 1. / 10000**omega  # (D/6,)
    rpe = []
    for i in range(_):
        pos_i = ipt_pos[:, :, i]    # (B, G)
        out = torch.einsum('bg, d->bgd', pos_i, omega)  # (B, G, D/6), outer product
        emb_sin = torch.sin(out) # (M, D/6)
        emb_cos = torch.cos(out) # (M, D/6)
        rpe.append(emb_sin)
        rpe.append(emb_cos)
    return torch.cat(rpe, dim=-1)

class LearnedPositionalEncoder(nn.Module):
    def __init__(
            self,
            num_channels: int,
            embed_dim: int,
            use_relative_features: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_relative_features = use_relative_features

        self.pos_enc = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        self.feature_enc = None
        if self.use_relative_features:
            assert num_channels > 3, "num_channels must be greater than 3 to use relative features"
            self.feature_enc = nn.Sequential(
                nn.Linear(num_channels-3, 128),
                nn.GELU(),
                nn.Linear(128, embed_dim),
            )

    def reset_parameters(self):
        for p in self.parameters():
            if isinstance(p, nn.Linear):
                p.reset_parameters()

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        pos =  self.pos_enc(pos[...,:3])
        if self.feature_enc is not None:
            pos = pos + self.feature_enc(pos[...,3:])
        return pos