import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.network import CustomNet
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data normalization
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # Enhanced training transforms
    transform_train = transforms.Compose([
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
    
    # Validation transforms
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # CIFAR10 Dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    
    valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    val_loader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = CustomNet().to(device)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # SGD with momentum and increased weight decay
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3, nesterov=True)
    
    # OneCycleLR scheduler
    epochs = 100
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,  # 20% warmup
        div_factor=10,  # initial_lr = max_lr/10
        final_div_factor=100  # min_lr = initial_lr/100
    )
    
    # Training loop
    best_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}/{epochs}')
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model and early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best accuracy: {best_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered. Best accuracy: {best_acc:.2f}%')
                break

if __name__ == '__main__':
    main() 