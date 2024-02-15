import torch


class MLPBlock(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.2):
        super(MLPBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, in_channels),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MixerBlock(torch.nn.Module):
    def __init__(self, num_patches: int, dim: int, token_dim: int, channel_dim: int, dropout: float):
        super(MixerBlock, self).__init__()
        self.token_mixing = MLPBlock(num_patches, token_dim, dropout)
        self.channels_mixing = MLPBlock(dim, channel_dim, dropout)
        self.pre_norm = torch.nn.LayerNorm(dim)
        self.post_norm = torch.nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.pre_norm(x)
        h = h.transpose(-1, -2)
        h = self.token_mixing(h)
        h = h.transpose(-1, -2)

        h = h + residual
        residual = h
        h = self.post_norm(h)
        h = self.channels_mixing(h)

        h = h + residual

        return h
