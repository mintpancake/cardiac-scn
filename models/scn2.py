import torch
import torch.nn as nn
from typing import Sequence


class SCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 128,
        factor: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()

        self.HLA = LocalAppearance(
            in_channels, num_classes, filters, dropout)

        self.down = nn.AvgPool3d(factor, factor, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=factor,
                              mode='trilinear', align_corners=True)

        self.HSC = nn.Sequential(
            nn.Conv3d(num_classes, filters, 7, 1, 3, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(filters, filters, 7, 1, 3, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(filters, filters, 7, 1, 3, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(filters, num_classes, 7, 1, 3, bias=True),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.trunc_normal_(self.HSC[-2].weight, 0, 1e-3)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        hla = self.HLA(x)
        hsc = self.up(self.HSC(self.down(hla)))
        heatmap = hla * hsc
        return heatmap, hla, hsc


class LocalAppearance(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.pool = nn.AvgPool3d(2, 2, ceil_mode=True)
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)

        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels, filters, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.enc1 = self.EncBlock(filters, filters, dropout)
        self.enc2 = self.EncBlock(filters, filters, dropout)
        self.enc3 = self.EncBlock(filters, filters, dropout)
        self.enc4 = self.EncBlock(filters, filters, dropout)

        self.dec4 = self.DecBlock(filters, filters)
        self.dec3 = self.DecBlock(filters, filters)
        self.dec2 = self.DecBlock(filters, filters)
        self.dec1 = self.DecBlock(filters, filters)

        self.out_conv = nn.Conv3d(filters, num_classes, 3, 1, 1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.trunc_normal_(self.out_conv.weight, 0, 1e-3)

    def EncBlock(self, in_channels, out_channels, dropout=0.5):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.Dropout3d(dropout, True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def DecBlock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)

        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d4 = self.dec4(e4)
        d3 = self.dec3(e3)+self.up(d4)
        d2 = self.dec2(e2)+self.up(d3)
        d1 = self.dec1(e1)+self.up(d2)

        out = self.out_conv(d1)
        return out
