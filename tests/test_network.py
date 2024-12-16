import torch
import pytest
from model.network import CustomNet, DepthwiseSeparableConv, DilatedConv

def test_network_output_shape():
    model = CustomNet(num_classes=10)
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)
    assert output.shape == (batch_size, 10)

def test_depthwise_separable_conv():
    conv = DepthwiseSeparableConv(in_channels=32, out_channels=64)
    x = torch.randn(1, 32, 16, 16)
    output = conv(x)
    assert output.shape == (1, 64, 16, 16)

def test_dilated_conv():
    conv = DilatedConv(in_channels=64, out_channels=128)
    x = torch.randn(1, 64, 16, 16)
    output = conv(x)
    assert output.shape == (1, 128, 16, 16)

def test_parameter_count():
    model = CustomNet()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 200000, f"Model has {total_params} parameters, should be less than 200000" 