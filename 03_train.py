import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import create_model
from data_loader import create_data_loaders, get_class_distribution


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, loss, acc, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Attention-based CNN for Image Classification')
    parser.add_argument('--data_dir', type=str, default='pictures', 
                       help='Path to directory containing labeled subdirectories')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224, 
                       help='Input image size')
    parser.add_argument('--train_split', type=float, default=0.8, 
                       help='Fraction of data for training')
    parser.add_argument('--save_dir', type=str, default='checkpoints', 
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print class distribution
    get_class_distribution(args.data_dir)
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, class_names, num_classes = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_split=args.train_split,
        augment=True
    )
    
    # Create model
    print(f"\nCreating model with {num_classes} classes...")
    model = create_model(num_classes, input_channels=3)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('accuracy', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)
            print(f"New best model saved! Validation Accuracy: {val_acc:.2f}%")
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs - 1, val_loss, val_acc, final_model_path)
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs, 
                         save_path=os.path.join(args.save_dir, 'training_history.png'))
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
