import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # TODO: conv1, bn1, conv2, bn2, shortcut (Identity or 1x1 projection)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(
              in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        # TODO: conv1 -> bn1 -> relu -> conv2 -> bn2 -> add shortcut -> relu
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out +=self.shortcut(x)
        return torch.relu(out)
