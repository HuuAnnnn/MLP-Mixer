import torch
import torch.nn.functional as F
from mlp_mixer.mixer_block import MixerBlock
from mlp_mixer.patch_embedding import PatchEmbedding


class MLPMixer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: tuple[int, int],
        patch_size: int,
        hidden_dim: int,
        num_mixers: int,
        num_classes: int,
        token_dim: int,
        channels_dim: int,
        dropout: float,
    ) -> None:
        super(MLPMixer, self).__init__()
        W, H = image_size
        num_patches = (W * H) // (patch_size**2)
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            width=W,
            patch_size=patch_size,
        )
        self.mixers = torch.nn.ModuleList(
            MixerBlock(
                num_patches=num_patches,
                dim=hidden_dim,
                token_dim=token_dim,
                channel_dim=channels_dim,
                dropout=dropout,
            )
            for _ in range(num_mixers)
        )
        self.g_avg_pooling = torch.nn.LayerNorm(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.patch_embedding(x)
        for mixer in self.mixers:
            h = mixer(h)
        h = self.g_avg_pooling(h)
        h = h.mean(dim=1)
        h = self.fc(h)

        return F.softmax(h, dim=1)
