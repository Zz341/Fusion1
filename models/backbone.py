import torch
import torch.nn as nn

class ResNetBackbone(nn.Module):
    def __init__(self, input_channels=1, output_channels=64):
        super(ResNetBackbone, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1, bias=False),
            nn.GroupNorm(4, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.GroupNorm(4, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, output_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.stem(x)