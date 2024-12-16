import torch
from torchvision import transforms
import numpy as np
from PIL import Image

def get_test_transforms():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        normalize,
    ])

def test_augmentation_output_shape():
    # Create dummy input (PIL Image)
    input_image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    
    # Apply transforms
    transform = get_test_transforms()
    output = transform(input_image)
    
    # Check output shape
    assert output.shape == (3, 32, 32), f"Expected shape (3, 32, 32), got {output.shape}"

def test_augmentation_value_range():
    # Create dummy input (PIL Image)
    input_image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    
    # Apply transforms
    transform = get_test_transforms()
    output = transform(input_image)
    
    # Check value ranges after normalization
    assert output.min() >= -3 and output.max() <= 3, f"Values should be roughly in [-3, 3] range after normalization"