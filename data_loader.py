import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_transforms(image_size=224, augment=True):
    """
    Get data transformation pipelines
    
    Args:
        image_size: Target image size (default: 224)
        augment: Whether to apply data augmentation for training
    
    Returns:
        train_transform, val_transform: Transformation pipelines
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_dir, batch_size=32, image_size=224, 
                        train_split=0.8, augment=True, num_workers=4):
    """
    Create data loaders for training and validation
    
    Args:
        data_dir: Path to the pictures directory containing subdirectories (labels)
        batch_size: Batch size for training
        image_size: Target image size
        train_split: Fraction of data to use for training (rest for validation)
        augment: Whether to apply data augmentation
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, class_names: Data loaders and class names
    """
    # Get transformations
    train_transform, val_transform = get_data_transforms(image_size, augment)
    
    # Load full dataset (without transform first to get class info)
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # Get class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"Found {num_classes} classes: {class_names}")
    print(f"Total images: {len(full_dataset)}")
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Create indices list and randomly shuffle
    indices = list(range(len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate datasets with appropriate transforms
    train_dataset_full = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    # Create subsets with the split indices
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, class_names, num_classes


def create_data_loaders_subset(data_dir, batch_size=32, image_size=224, 
                                train_split=0.8, data_subset=1.0, augment=True, num_workers=4):
    """
    Create data loaders for training and validation using a subset of data
    
    Args:
        data_dir: Path to the pictures directory containing subdirectories (labels)
        batch_size: Batch size for training
        image_size: Target image size
        train_split: Fraction of data to use for training (rest for validation)
        data_subset: Fraction of total data to use (e.g., 0.25 for 25%)
        augment: Whether to apply data augmentation
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, class_names, num_classes: Data loaders and class names
    """
    # Get transformations
    train_transform, val_transform = get_data_transforms(image_size, augment)
    
    # Load full dataset (without transform first to get class info)
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # Get class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    # Calculate subset size
    total_samples = len(full_dataset)
    subset_size = int(data_subset * total_samples)
    
    print(f"Found {num_classes} classes: {class_names}")
    print(f"Total images: {total_samples}, Using subset: {subset_size} ({data_subset*100:.1f}%)")
    
    # Create indices list and randomly shuffle
    indices = list(range(total_samples))
    np.random.seed(42)  # Fixed seed for reproducibility
    np.random.shuffle(indices)
    
    # Take subset
    subset_indices = indices[:subset_size]
    
    # Split subset into train/val
    train_size = int(train_split * len(subset_indices))
    train_indices = subset_indices[:train_size]
    val_indices = subset_indices[train_size:]
    
    # Create separate datasets with appropriate transforms
    train_dataset_full = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    # Create subsets with the split indices
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, class_names, num_classes


def get_class_distribution(data_dir):
    """
    Print class distribution in the dataset
    
    Args:
        data_dir: Path to the pictures directory
    """
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            class_counts[class_name] = count
    
    print("\nClass Distribution:")
    print("-" * 40)
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count} images")
    print("-" * 40)
    
    return class_counts

