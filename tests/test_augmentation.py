import numpy as np
from utils.augmentation import get_training_augmentation

def test_augmentation_output_shape():
    # Create dummy image
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    dataset_mean = [128, 128, 128]
    
    # Get augmentation pipeline
    transform = get_training_augmentation(dataset_mean)
    
    # Apply augmentation
    augmented = transform(image=image)['image']
    
    # Check output shape
    assert augmented.shape == image.shape, "Augmented image should have same shape as input"

def test_augmentation_value_range():
    # Create dummy image
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    dataset_mean = [128, 128, 128]
    
    # Get augmentation pipeline
    transform = get_training_augmentation(dataset_mean)
    
    # Apply augmentation
    augmented = transform(image=image)['image']
    
    # Check value range
    assert augmented.min() >= 0 and augmented.max() <= 255, "Values should be in [0, 255] range"
    assert augmented.dtype == np.uint8, "Output should be uint8" 