import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )
        # GroupNorm replaced by BatchNorm 
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )
        # GroupNorm replaced by BatchNorm 
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        # ReLU replaced by Leaky ReLU 
        out = F.leaky_relu(out, 0.1)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Removed Attention
        
        out = out + residual
        # ReLU replaced by Leaky ReLU 
        return F.leaky_relu(out, 0.1)


class DQNResNet(nn.Module):
    def __init__(
        self,
        input_channels=12,
        num_actions=6
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_actions = num_actions
        
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=3,
                padding=1
            ),
            # GroupNorm replaced by BatchNorm2d
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        
        self.res1 = ResBlock(64)
        
        # Downsampling: reduces 17x17 to 8x8
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res2 = ResBlock(64)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            # Input-Dimension reduced: 64 * 8 * 8 instead of 64 * 17 * 17
            nn.Linear(
                64 * 8 * 8,
                512
            ),
            nn.LeakyReLU(0.1),
            nn.Linear(
                512,
                num_actions
            )
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.res1(out)
        out = self.pool(out) # Added Pooling 
        out = self.res2(out)
        return self.head(out)