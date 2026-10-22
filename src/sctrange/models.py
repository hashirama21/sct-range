"""2.5D U-Net: a stack of neighbouring source slices and the body mask in, one CT slice out."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class UNet2d(nn.Module):
    """Encoder-decoder with skip connections; spatial size must be divisible by ``2 ** (depth - 1)``.

    Channel 0 is the encoded CT; further output channels (the range-risk head) share the decoder.
    """

    def __init__(self, in_channels: int, base_channels: int = 32, depth: int = 4, out_channels: int = 1) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        widths = [base_channels * 2**level for level in range(depth)]
        self.depth = depth
        self.encoders = nn.ModuleList([ConvBlock(in_channels, widths[0])])
        self.encoders.extend(ConvBlock(widths[i - 1], widths[i]) for i in range(1, depth))
        self.decoders = nn.ModuleList(ConvBlock(widths[i] + widths[i - 1], widths[i - 1]) for i in range(depth - 1, 0, -1))
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    @property
    def stride(self) -> int:
        return 2 ** (self.depth - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x if index == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)
