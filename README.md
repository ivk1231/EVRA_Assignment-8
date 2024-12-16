# CIFAR-10 Neural Network Implementation

A custom CNN architecture for CIFAR-10 dataset with specific architectural constraints.

## Architecture Overview

The network follows a C1C2C3C4 structure with the following components:

### Network Blocks
1. **C1 Block**
   - Regular strided convolution (3→16 channels)
   - BatchNorm + ReLU
   - ResidualBlock(16 channels)

2. **C2 Block**
   - Depthwise Separable Conv (16→32 channels, stride=2)
   - ResidualBlock(32 channels)

3. **C3 Block**
   - Dilated Conv (32→64 channels, dilation=2)
   - ResidualBlock(64 channels)

4. **C4 Block**
   - Regular convolution (64→128 channels, stride=2)
   - BatchNorm + ReLU

5. **Output Block**
   - Global Average Pooling
   - Dropout(0.2)
   - Fully Connected (128→10)

### Training Strategy

The training configuration was carefully chosen for stable convergence:

1. **Extended Training Duration (200 epochs)**:
   - Allows gradual learning of complex features
   - Gives cosine annealing scheduler enough cycles
   - Prevents premature convergence
   - Results in more robust feature representations

2. **Learning Rate Schedule**:
   - Initial LR: 0.1 (relatively high)
   - Cosine Annealing scheduler
   - Smooth decay over 200 epochs
   - No restarts to ensure stability

3. **Regularization Balance**:
   - Weight decay: 5e-4 (moderate)
   - Dropout: 0.2 (light, only before FC)
   - Label smoothing: 0.1
   - Dual augmentation strategy

4. **Data Augmentation**:
   Two complementary augmentation pipelines:
   
   **Albumentations Pipeline** (preprocessing):
   ```python
   Compose([
       HorizontalFlip(p=0.5),
       ShiftScaleRotate(
           shift_limit=0.1,
           scale_limit=0.1,
           rotate_limit=15,
           p=0.5
       ),
       CoarseDropout(
           max_holes=1, min_holes=1,
           hole_size_range=(16, 16),
           min_height=16, max_height=16,
           min_width=16, max_width=16,
           fill_value=dataset_mean,
           p=0.5
       )
   ])
   ```

   **Torchvision Pipeline** (training):
   ```python
   transforms.Compose([
       transforms.RandomCrop(32, padding=4),
       transforms.RandomHorizontalFlip(),
       transforms.ColorJitter(brightness=0.2, contrast=0.2),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), 
                          (0.2023, 0.1994, 0.2010))
   ])
   ```

5. **Batch Size and Momentum**:
   - Batch size: 256 for stable gradients
   - Momentum: 0.9 with SGD
   - Provides good balance of speed and stability

## Project Structure
```
.
├── model/
│   ├── __init__.py
│   └── network.py          # Neural network architecture
├── utils/
│   ├── __init__.py
│   └── augmentation.py     # Albumentations augmentation utilities
├── tests/
│   ├── __init__.py
│   ├── test_network.py     # Network architecture tests
│   └── test_augmentation.py# Augmentation tests
├── train.py               # Training script with torchvision transforms
└── requirements.txt       # Dependencies
```

## Requirements
```
torch>=1.9.0
torchvision>=0.10.0
albumentations>=1.1.0
numpy>=1.19.5
pytest>=6.2.5
```

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
python -m pytest tests/
```

3. Run training:
```bash
python train.py
```

The training script will:
- Load and preprocess CIFAR-10 dataset
- Train for 200 epochs with cosine learning rate decay
- Save best model as 'best_model.pth'
- Print training/validation metrics per epoch