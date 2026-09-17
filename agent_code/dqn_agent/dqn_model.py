import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, channels)
        self.attn = SpatialAttention()

    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.attn(out)
        return F.relu(out + residual)

class DQNResNet(nn.Module):
    def __init__(self, input_channels=10, num_actions=6):
        super(DQNResNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU()
        )
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 17 * 17, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.res1(out)
        out = self.res2(out)
        return self.head(out)