"""
Plot trial number vs best validation accuracy from Optuna optimization trials.
Scans a directory for trial_XX subdirectories and extracts best_val_acc from training_history.json files.
"""

import os
import json
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def find_trial_directories(directory):
    """Find all trial_XX directories in the given directory"""
    trial_dirs = []
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return trial_dirs
    
    # Look for directories matching trial_XX pattern
    pattern = re.compile(r'^trial_(\d+)$')
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                trial_num = int(match.group(1))
                trial_dirs.append((trial_num, item_path))
    
    # Sort by trial number
    trial_dirs.sort(key=lambda x: x[0])
    return trial_dirs


def extract_best_val_acc(trial_dir):
    """Extract best_val_acc from training_history.json in trial directory"""
    history_path = os.path.join(trial_dir, 'training_history.json')
    
    if not os.path.exists(history_path):
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        if 'best_val_acc' in history:
            return history['best_val_acc']
        else:
            print(f"Warning: 'best_val_acc' not found in {history_path}")
            return None
    except Exception as e:
        print(f"Error reading {history_path}: {e}")
        return None


def plot_trial_history(directory, output_path='trial_history.png'):
    """Plot trial number vs best_val_acc"""
    
    # Find all trial directories
    trial_dirs = find_trial_directories(directory)
    
    if not trial_dirs:
        print(f"No trial directories found in '{directory}'")
        print("Looking for directories matching pattern: trial_XX")
        return
    
    print(f"Found {len(trial_dirs)} trial directories")
    
    # Extract trial numbers and best_val_acc
    trial_numbers = []
    best_val_accs = []
    
    for trial_num, trial_dir in trial_dirs:
        best_val_acc = extract_best_val_acc(trial_dir)
        if best_val_acc is not None:
            trial_numbers.append(trial_num)
            best_val_accs.append(best_val_acc)
            print(f"Trial {trial_num}: best_val_acc = {best_val_acc:.2f}%")
        else:
            print(f"Warning: Could not extract best_val_acc from trial {trial_num}")
    
    if not trial_numbers:
        print("No valid data found to plot")
        return
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot line and markers
    plt.plot(trial_numbers, best_val_accs, 'o-', linewidth=2, markersize=6, 
             color='steelblue', alpha=0.7, label='Best Validation Accuracy')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Labels and title
    plt.xlabel('Trial Number', fontsize=12, fontweight='bold')
    plt.ylabel('Best Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Hyperparameter Optimization: Trial Number vs Best Validation Accuracy', 
              fontsize=14, fontweight='bold')
    
    # Add statistics
    if len(best_val_accs) > 0:
        max_acc = max(best_val_accs)
        min_acc = min(best_val_accs)
        mean_acc = np.mean(best_val_accs)
        std_acc = np.std(best_val_accs)
        best_trial = trial_numbers[best_val_accs.index(max_acc)]
        
        stats_text = (
            f'Trials: {len(trial_numbers)}\n'
            f'Best: {max_acc:.2f}% (Trial {best_trial})\n'
            f'Mean: {mean_acc:.2f}%\n'
            f'Std: {std_acc:.2f}%'
        )
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Highlight best trial
        plt.plot(best_trial, max_acc, 'o', markersize=12, color='red', 
                label=f'Best Trial: {best_trial}')
    
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    # Show plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot trial number vs best validation accuracy from Optuna trials'
    )
    parser.add_argument('--directory', type=str, default='optuna_checkpoints',
                       help='Directory containing trial_XX subdirectories (default: optuna_checkpoints)')
    parser.add_argument('--output', type=str, default='trial_history.png',
                       help='Output path for the plot (default: trial_history.png)')
    
    args = parser.parse_args()
    
    print(f"Scanning directory: {args.directory}")
    plot_trial_history(args.directory, args.output)


if __name__ == "__main__":
    main()
