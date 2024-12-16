from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, CoarseDropout,
    RandomBrightnessContrast, RandAugment
)
import numpy as np

def get_training_augmentation():
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        CoarseDropout(
            max_holes=1,
            min_holes=1,
            max_height=16,
            min_height=16,
            max_width=16,
            min_width=16,
            fill_value=tuple([x * 255 for x in [0.4914, 0.4822, 0.4465]]),
            p=0.5
        ),
        RandAugment(
            n=2,
            m=9,
            p=0.5
        )
    ]) 