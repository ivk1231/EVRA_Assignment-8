import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.relu(x)
        return x

class CustomNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # C1: Regular strided conv with more channels
        self.c1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.res1 = ResidualBlock(24)
        
        # C2: Depthwise Separable Conv with stride
        self.c2 = DepthwiseSeparableConv(24, 48, stride=2)
        self.res2 = ResidualBlock(48)
        
        # C3: Dilated Conv
        self.c3 = DilatedConv(48, 96)
        self.res3 = ResidualBlock(96)
        
        # C4: Regular conv with stride
        self.c4 = nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Final FC layer with more capacity
        self.fc = nn.Linear(192, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # C1 block
        x = self.relu(self.bn1(self.c1(x)))
        x = self.res1(x)
        
        # C2 block
        x = self.c2(x)
        x = self.res2(x)
        
        # C3 block
        x = self.c3(x)
        x = self.res3(x)
        
        # C4 block
        x = self.relu(self.bn4(self.c4(x)))
        
        # GAP and FC
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x