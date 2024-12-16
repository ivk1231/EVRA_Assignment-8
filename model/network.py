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
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation
        )
    
    def forward(self, x):
        return self.conv(x)

class CustomNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Reduced initial channels
        # C1: Regular strided conv
        self.c1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # C2: Depthwise Separable Conv with stride
        self.c2 = DepthwiseSeparableConv(16, 32, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        # C3: Dilated Conv
        self.c3 = DilatedConv(32, 64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # C4: Regular conv with stride
        self.c4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final FC layer
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.c1(x)))
        x = self.relu(self.bn2(self.c2(x)))
        x = self.relu(self.bn3(self.c3(x)))
        x = self.relu(self.bn4(self.c4(x)))
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x