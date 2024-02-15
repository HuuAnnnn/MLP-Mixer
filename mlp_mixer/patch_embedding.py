import torch

class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, width: int, patch_size: int):
        super().__init__()
        self.width = width
        self.patch_size = patch_size
        self.per_patch_fc = torch.nn.Conv2d(in_channels, hidden_dim, (patch_size, patch_size), patch_size)

    def forward(self, x: torch.Tensor):
        patches = self.per_patch_fc(x)
        B, C, W, H = patches.shape
        return patches.reshape(B, W * H, C)
