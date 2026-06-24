"""
unet_resnet.py
U-Net architecture with a pretrained ResNet34 encoder for road segmentation.
Output: single-channel mask (road probability) at input resolution.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """Two conv layers + BN + ReLU, used in the decoder."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + concat skip connection + ConvBlock."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        # handle size mismatch from odd input dims
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    """
    U-Net with ResNet34 encoder (pretrained on ImageNet).
    Input:  (B, 3, H, W)
    Output: (B, 1, H, W) - raw logits, apply sigmoid for probability
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        # Encoder stages (extracting intermediate feature maps for skip connections)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # 64 channels, /2
        self.pool0 = resnet.maxpool                                        # /4
        self.enc1 = resnet.layer1   # 64 channels
        self.enc2 = resnet.layer2   # 128 channels, /8
        self.enc3 = resnet.layer3   # 256 channels, /16
        self.enc4 = resnet.layer4   # 512 channels, /32

        # Decoder
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.enc0(x)        # /2
        x0p = self.pool0(x0)     # /4
        x1 = self.enc1(x0p)      # /4
        x2 = self.enc2(x1)       # /8
        x3 = self.enc3(x2)       # /16
        x4 = self.enc4(x3)       # /32

        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)
        return out


if __name__ == "__main__":
    model = UNetResNet34(pretrained=True)
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")