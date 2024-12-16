from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, CoarseDropout
)
import numpy as np

def get_training_augmentation(dataset_mean):
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        CoarseDropout(
            max_holes=1,
            min_holes=1,
            hole_size_range=(16, 16),
            min_height=16,
            max_height=16,
            min_width=16,
            max_width=16,
            fill_value=dataset_mean,
            mask_fill_value=None,
            p=0.5
        )
    ]) 