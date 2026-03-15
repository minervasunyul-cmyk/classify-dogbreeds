"""
Visualize attention maps from CBAM modules in the trained model.
Shows where the model focuses its attention when making predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

from model import create_model
from data_loader import get_data_transforms


class AttentionVisualizer:
    """Class to extract and visualize attention maps from CBAM modules"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attention_maps = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture attention maps"""
        def make_spatial_hook(name):
            def hook(module, input, output):
                # Capture the spatial attention map (output of spatial attention module)
                # Remove batch dimension for visualization
                self.attention_maps[name] = output[0].cpu().numpy()
            return hook
        
        # Register hooks on the spatial attention modules within each CBAM
        if hasattr(self.model, 'attention1') and hasattr(self.model.attention1, 'spatial_attention'):
            self.hooks.append(self.model.attention1.spatial_attention.register_forward_hook(
                make_spatial_hook('attention1')))
        if hasattr(self.model, 'attention2') and hasattr(self.model.attention2, 'spatial_attention'):
            self.hooks.append(self.model.attention2.spatial_attention.register_forward_hook(
                make_spatial_hook('attention2')))
        if hasattr(self.model, 'attention3') and hasattr(self.model.attention3, 'spatial_attention'):
            self.hooks.append(self.model.attention3.spatial_attention.register_forward_hook(
                make_spatial_hook('attention3')))
        if hasattr(self.model, 'attention4') and hasattr(self.model.attention4, 'spatial_attention'):
            self.hooks.append(self.model.attention4.spatial_attention.register_forward_hook(
                make_spatial_hook('attention4')))
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_maps(self, image_tensor):
        """Get attention maps for a given image"""
        self.attention_maps = {}
        with torch.no_grad():
            _ = self.model(image_tensor)
        return self.attention_maps


def load_model(checkpoint_path, num_classes, device):
    """Load model from checkpoint"""
    model = create_model(num_classes, input_channels=3)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def get_class_names_from_data_dir(data_dir):
    """Get class names from data directory"""
    if os.path.exists(data_dir):
        try:
            from torchvision import datasets
            dataset = datasets.ImageFolder(root=data_dir)
            return dataset.classes, len(dataset.classes)
        except:
            # Fallback: get from directory structure
            try:
                class_names = sorted([d for d in os.listdir(data_dir) 
                                    if os.path.isdir(os.path.join(data_dir, d))])
                return class_names, len(class_names)
            except:
                pass
    return None, None

def get_num_classes_from_checkpoint(checkpoint_path):
    """Try to infer number of classes from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try to get from checkpoint metadata
        if 'class_names' in checkpoint:
            return len(checkpoint['class_names'])
        
        # Try to infer from model state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Look for the last fc layer (fc.4.weight is the final classifier)
        for key in sorted(state_dict.keys(), reverse=True):
            if 'fc' in key and 'weight' in key:
                # Last fc layer weight shape is [num_classes, hidden_dim]
                num_classes = state_dict[key].shape[0]
                if num_classes > 0:  # Make sure it's valid
                    return num_classes
    except Exception as e:
        print(f"Warning: Could not infer num_classes from checkpoint: {e}")
    
    return None


def visualize_attention(image_path, model, class_names, device, image_size=224, 
                       output_dir='attention_visualizations', layer='all'):
    """Visualize attention maps for a single image"""
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load original image (without normalization for display)
    original_image = Image.open(image_path).convert('RGB')
    original_image_resized = original_image.resize((image_size, image_size))
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()] if class_names else f"Class {predicted.item()}"
    confidence_score = confidence.item()
    
    # Get attention maps
    visualizer = AttentionVisualizer(model, device)
    visualizer.register_hooks()
    attention_maps = visualizer.get_attention_maps(image_tensor)
    visualizer.remove_hooks()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image name for saving
    image_name = Path(image_path).stem
    
    # Visualize each attention layer
    layers_to_show = list(attention_maps.keys()) if layer == 'all' else [layer]
    
    fig, axes = plt.subplots(2, len(layers_to_show) + 1, figsize=(5 * (len(layers_to_show) + 1), 10))
    if len(layers_to_show) == 0:
        axes = axes.reshape(2, 1)
    
    # Show original image
    axes[0, 0].imshow(original_image_resized)
    axes[0, 0].set_title(f'Original Image\nPredicted: {predicted_class} ({confidence_score:.2%})', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_image_resized)
    axes[1, 0].set_title('Original Image', fontsize=12)
    axes[1, 0].axis('off')
    
    # Process each attention layer
    for idx, layer_name in enumerate(layers_to_show, 1):
        attn_map = attention_maps[layer_name]
        
        # Handle different shapes: [C, H, W] or [H, W] or [1, H, W]
        if len(attn_map.shape) == 3:
            # If multiple channels, average them (should be 1 channel for spatial attention)
            attn_map = attn_map.squeeze()
            if len(attn_map.shape) == 2:
                pass  # Already 2D
            else:
                attn_map = attn_map.mean(axis=0)
        elif len(attn_map.shape) == 2:
            pass  # Already 2D
        else:
            # Handle unexpected shapes
            attn_map = attn_map.squeeze()
            if len(attn_map.shape) != 2:
                print(f"Warning: Unexpected attention map shape {attn_map.shape} for {layer_name}")
                continue
        
        # Normalize to [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Upsample to image size
        attn_map_tensor = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0)
        attn_map_upsampled = F.interpolate(
            attn_map_tensor, 
            size=(image_size, image_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze().numpy()
        
        # Create heatmap overlay using 'hot' colormap (black->red->yellow = low->high attention)
        # 'hot' is more intuitive: black (low) -> red -> yellow (high)
        heatmap = cm.hot(attn_map_upsampled)[:, :, :3]  # Remove alpha channel
        
        # Show attention map with colorbar
        im = axes[0, idx].imshow(attn_map_upsampled, cmap='hot')
        axes[0, idx].set_title(f'{layer_name.upper()}\nAttention Map\n(Red/Yellow = High)', fontsize=11)
        axes[0, idx].axis('off')
        # Add colorbar
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046, pad=0.04)
        
        # Show overlay
        overlay = 0.6 * np.array(original_image_resized) / 255.0 + 0.4 * heatmap
        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'{layer_name.upper()}\nOverlay', fontsize=12)
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{image_name}_attention.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Attention visualization saved to {output_path}")
    return output_path


def process_directory(directory, model, class_names, device, image_size=224, 
                     output_dir='attention_visualizations', layer='all', max_images=None):
    """Process all images in a directory (including subdirectories)"""
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return
    
    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No images found in {directory}")
        return
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images. Processing...")
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        try:
            visualize_attention(image_path, model, class_names, device, 
                              image_size, output_dir, layer)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    parser = argparse.ArgumentParser(description='Visualize attention maps from trained model')
    parser.add_argument('--checkpoint', type=str, default='results/final_model.pth',
                       help='Path to model checkpoint (default: results/final_model.pth)')
    parser.add_argument('--data_dir', type=str, default='examples',
                       help='Directory containing images to visualize (default: examples)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--output_dir', type=str, default='attention_visualizations',
                       help='Directory to save visualization outputs (default: attention_visualizations)')
    parser.add_argument('--layer', type=str, default='all',
                       choices=['all', 'attention1', 'attention2', 'attention3', 'attention4'],
                       help='Which attention layer to visualize (default: all)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (will auto-detect from checkpoint if not provided)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Auto-detect checkpoint if not found
    if not os.path.exists(args.checkpoint):
        # Try alternative locations
        alternatives = ['results/final_model.pth', 'checkpoints/best_model.pth', 'checkpoints/final_model.pth']
        for alt in alternatives:
            if os.path.exists(alt):
                args.checkpoint = alt
                print(f"Using checkpoint: {args.checkpoint}")
                break
        else:
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Please specify --checkpoint")
    
    # Infer number of classes from checkpoint first (most reliable)
    print(f"Loading checkpoint to infer number of classes...")
    num_classes = get_num_classes_from_checkpoint(args.checkpoint)
    
    if args.num_classes is not None:
        num_classes = args.num_classes
        print(f"Using specified number of classes: {num_classes}")
    elif num_classes is not None:
        print(f"Inferred {num_classes} classes from checkpoint")
    else:
        # Try to get from data directory as fallback
        class_names, num_classes_from_dir = get_class_names_from_data_dir(args.data_dir)
        if num_classes_from_dir is not None:
            num_classes = num_classes_from_dir
            print(f"Inferred {num_classes} classes from data directory")
        else:
            raise ValueError("Could not infer number of classes. Please specify --num_classes")
    
    # Get class names (try data directory, but use defaults if not found)
    class_names, _ = get_class_names_from_data_dir(args.data_dir)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
        print(f"Using default class names: {class_names}")
    else:
        if len(class_names) != num_classes:
            print(f"Warning: Found {len(class_names)} classes in directory but model has {num_classes} classes")
            print(f"Using model's {num_classes} classes with default names")
            class_names = [f"Class {i}" for i in range(num_classes)]
        else:
            print(f"Found {num_classes} classes: {class_names}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, num_classes, device)
    print("Model loaded successfully!")
    
    # Process all images in directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory not found at {args.data_dir}")
        return
    
    process_directory(args.data_dir, model, class_names, device, 
                     args.image_size, args.output_dir, args.layer, max_images=None)
    
    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()