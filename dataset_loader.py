from typing import Tuple
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import warnings


IMG_SIZE = 64  # Small input size to keep the CNN lightweight


class RobustImageFolder(datasets.ImageFolder):
    """ImageFolder that gracefully handles corrupted/truncated images."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable loading of truncated images
        Image.LOAD_TRUNCATED_IMAGES = True
        self.corrupted_files = []
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except (OSError, RuntimeError, ValueError) as e:
            # Log corrupted file but skip it
            self.corrupted_files.append((path, str(e)))
            # Return a random valid sample instead
            valid_idx = index
            while valid_idx in range(len(self.imgs)):
                try:
                    path, target = self.imgs[valid_idx]
                    sample = self.loader(path)
                    if self.transform is not None:
                        sample = self.transform(sample)
                    return sample, target
                except:
                    valid_idx = (valid_idx + 1) % len(self.imgs)
            # Fallback: return a blank tensor
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), target


def build_transforms(
    preset: str = "simplecnn",
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and test transforms for images.

    Presets:
    - simplecnn: 64x64 + mean/std=0.5 (current behavior)
    - efficientnet_b0: 224x224 + ImageNet normalization
    """

    preset = (preset or "simplecnn").lower()

    if preset in {"efficientnet", "efficientnet_b0", "effnet_b0"}:
        size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        size = IMG_SIZE
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    common = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    train_tfms = transforms.Compose(common)
    test_tfms = transforms.Compose(common)
    return train_tfms, test_tfms


def load_imagefolder_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    preset: str = "simplecnn",
) -> Tuple[DataLoader, DataLoader, int]:
    """Create ImageFolder-based train/test DataLoaders and return num_classes.

    Expects directory layout:
      train_dir/<class_name>/*.{jpg,png,...}
      test_dir/<class_name>/*.{jpg,png,...}
    
    Handles corrupted/truncated images gracefully.
    """
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_tfms, test_tfms = build_transforms(preset=preset)
    train_ds = RobustImageFolder(root=train_dir, transform=train_tfms)
    test_ds = RobustImageFolder(root=test_dir, transform=test_tfms)

    num_classes = len(train_ds.classes)
    if num_classes <= 1:
        raise ValueError(
            "Detected <=1 class. Ensure data is organized in subfolders per class."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Log corrupted files
    if train_ds.corrupted_files:
        warnings.warn(f"Found {len(train_ds.corrupted_files)} corrupted training images")
    if test_ds.corrupted_files:
        warnings.warn(f"Found {len(test_ds.corrupted_files)} corrupted test images")
    
    return train_loader, test_loader, num_classes


def load_global_test_loader(
    global_test_dir: str,
    batch_size: int = 32,
    preset: str = "simplecnn",
) -> Tuple[DataLoader, int]:
    """Create a DataLoader for the global held-out test set and return (loader, num_classes).
    
    Handles corrupted/truncated images gracefully.
    """
    if not os.path.isdir(global_test_dir):
        raise FileNotFoundError(f"Global test directory not found: {global_test_dir}")

    _, test_tfms = build_transforms(preset=preset)
    test_ds = RobustImageFolder(root=global_test_dir, transform=test_tfms)
    num_classes = len(test_ds.classes)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    if test_ds.corrupted_files:
        warnings.warn(f"Found {len(test_ds.corrupted_files)} corrupted global test images")
    
    return test_loader, num_classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, criterion=None) -> Tuple[float, float]:
    """Return (loss, accuracy) with optional criterion for loss."""
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total if total > 0 else 0.0
    avg_loss = (total_loss / total) if (criterion is not None and total > 0) else 0.0
    return avg_loss, acc
