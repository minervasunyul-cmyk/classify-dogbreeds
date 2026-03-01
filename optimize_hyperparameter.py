import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
import os
import argparse
import optuna
from optuna.trial import TrialState
import json
import shutil
import pickle
from train import train_epoch, validate, save_checkpoint
from model import create_model
from data_loader import create_data_loaders, create_data_loaders_subset, get_class_distribution


def objective(trial, data_dir, device, num_epochs, train_split, save_dir, save_checkpoints, data_subset):
    """
    Objective function for Optuna optimization
    
    Args:
        trial: Optuna trial object
        data_dir: Path to data directory
        device: PyTorch device
        num_epochs: Number of training epochs per trial
        train_split: Fraction of data for training
        save_dir: Directory to save checkpoints
        save_checkpoints: Whether to save model checkpoints
        data_subset: Fraction of data to use (e.g., 0.25 for 25%)
    
    Returns:
        Best validation accuracy for this trial
    """
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    image_size = trial.suggest_categorical('image_size', [224, 256, 288])
    
    # Optimizer choice
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    
    # SGD-specific parameters
    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
    
    # Scheduler parameters
    scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.8)
    scheduler_patience = trial.suggest_int('scheduler_patience', 3, 10)
    
    # Architecture hyperparameters
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
    attention_reduction = trial.suggest_categorical('attention_reduction', [8, 16, 32])
    spatial_kernel_size = trial.suggest_categorical('spatial_kernel_size', [5, 7, 9])
    fc_hidden_dim = trial.suggest_categorical('fc_hidden_dim', [128, 256, 512])
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print(f"Learning Rate: {lr:.6f}")
    print(f"Batch Size: {batch_size}")
    print(f"Weight Decay: {weight_decay:.6f}")
    print(f"Image Size: {image_size}")
    print(f"Optimizer: {optimizer_name}")
    if optimizer_name == 'SGD':
        print(f"Momentum: {momentum:.3f}")
    print(f"Scheduler Factor: {scheduler_factor:.3f}")
    print(f"Scheduler Patience: {scheduler_patience}")
    print(f"--- Architecture Parameters ---")
    print(f"Dropout Rate: {dropout_rate:.3f}")
    print(f"Attention Reduction: {attention_reduction}")
    print(f"Spatial Kernel Size: {spatial_kernel_size}")
    print(f"FC Hidden Dimension: {fc_hidden_dim}")
    print(f"{'='*60}\n")
    
    # Create data loaders with subset of data for faster optimization
    try:
        # First get full dataset to know the size
        full_dataset = datasets.ImageFolder(root=data_dir)
        total_samples = len(full_dataset)
        subset_size = int(data_subset * total_samples)
        
        # Create data loaders with subset
        train_loader, val_loader, class_names, num_classes = create_data_loaders_subset(
            data_dir,
            batch_size=batch_size,
            image_size=image_size,
            train_split=train_split,
            data_subset=data_subset,
            augment=True
        )
        print(f"Using {data_subset*100:.1f}% of data ({subset_size} samples) for faster optimization")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        raise optuna.TrialPruned()
    
    # Create model with architecture hyperparameters
    model = create_model(
        num_classes, 
        input_channels=3,
        dropout_rate=dropout_rate,
        attention_reduction=attention_reduction,
        spatial_kernel_size=spatial_kernel_size,
        fc_hidden_dim=fc_hidden_dim
    )
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, 
                             momentum=momentum, nesterov=True)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, 
                                  patience=scheduler_patience)
    
    # Training loop with history tracking
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    max_patience = 15  # Early stopping patience
    
    # Track training history for plotting
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Report intermediate value for pruning
        trial.report(val_acc, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch + 1}")
            raise optuna.TrialPruned()
        
        # Track best validation accuracy and save best model state
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save trial results (model, history, architecture params)
    trial_checkpoint_dir = os.path.join(save_dir, f'trial_{trial.number}')
    os.makedirs(trial_checkpoint_dir, exist_ok=True)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc
    }
    history_path = os.path.join(trial_checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save architecture parameters
    arch_params = {
        'dropout_rate': dropout_rate,
        'attention_reduction': attention_reduction,
        'spatial_kernel_size': spatial_kernel_size,
        'fc_hidden_dim': fc_hidden_dim,
        'num_classes': num_classes
    }
    arch_params_path = os.path.join(trial_checkpoint_dir, 'architecture_params.json')
    with open(arch_params_path, 'w') as f:
        json.dump(arch_params, f, indent=2)
    
    # Save best model from this trial if checkpointing is enabled
    if save_checkpoints and best_model_state is not None:
        # Load best model state
        model.load_state_dict(best_model_state)
        checkpoint_path = os.path.join(trial_checkpoint_dir, 'best_model.pth')
        save_checkpoint(model, optimizer, best_epoch, val_losses[-1] if val_losses else 0.0, best_val_acc, checkpoint_path)
        print(f"Trial {trial.number} best model saved to {checkpoint_path}")
    
    print(f"\nTrial {trial.number} completed. Best Val Acc: {best_val_acc:.2f}%")
    print(f"Training history and architecture params saved to {trial_checkpoint_dir}")
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters using Optuna')
    parser.add_argument('--data_dir', type=str, default='pictures', 
                       help='Path to directory containing labeled subdirectories')
    parser.add_argument('--n_trials', type=int, default=50, 
                       help='Number of trials to run')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs per trial')
    parser.add_argument('--train_split', type=float, default=0.8, 
                       help='Fraction of data for training')
    parser.add_argument('--study_name', type=str, default='dog_breed_optimization', 
                       help='Name of the Optuna study')
    parser.add_argument('--storage', type=str, default=None, 
                       help='Storage URL for Optuna study (e.g., sqlite:///optuna.db)')
    parser.add_argument('--load_if_exists', action='store_true', 
                       help='Load existing study if it exists')
    parser.add_argument('--direction', type=str, default='maximize', 
                       choices=['maximize', 'minimize'],
                       help='Direction of optimization')
    parser.add_argument('--pruner', type=str, default='median', 
                       choices=['median', 'nop', 'successive_halving'],
                       help='Pruner type for early stopping')
    parser.add_argument('--n_jobs', type=int, default=1, 
                       help='Number of parallel jobs (1 for sequential)')
    parser.add_argument('--save_checkpoints', action='store_true', 
                       help='Save model checkpoints for each trial (saves disk space if disabled)')
    parser.add_argument('--data_subset', type=float, default=0.25, 
                       help='Fraction of data to use during optimization (default: 0.25 for 25%%, speeds up search)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = 'optuna_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Print class distribution
    get_class_distribution(args.data_dir)
    
    # Create pruner
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    elif args.pruner == 'successive_halving':
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    else:
        pruner = optuna.pruners.NopPruner()
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
        direction=args.direction,
        pruner=pruner
    )
    
    print(f"\nStarting optimization with {args.n_trials} trials...")
    print(f"Each trial will train for {args.epochs} epochs")
    print(f"Optimization direction: {args.direction}")
    print(f"Pruner: {args.pruner}")
    print(f"{'='*60}\n")
    
    # Optimize
    print(f"Using {args.data_subset*100:.1f}% of data for faster optimization")
    study.optimize(
        lambda trial: objective(trial, args.data_dir, device, args.epochs, 
                              args.train_split, save_dir, args.save_checkpoints, args.data_subset),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")
    print(f"Number of finished trials: {len(study.trials)}")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])
    
    print(f"Number of pruned trials: {len(pruned_trials)}")
    print(f"Number of complete trials: {len(complete_trials)}")
    print(f"Number of failed trials: {len(failed_trials)}")
    
    # Check if there are any completed trials
    if len(complete_trials) == 0:
        print(f"\n{'='*60}")
        print("WARNING: No trials completed successfully!")
        print(f"{'='*60}")
        print("All trials were either pruned or failed.")
        print("This could indicate:")
        print("  - All hyperparameter combinations were unpromising (pruned)")
        print("  - Errors occurred during training")
        print("  - Dataset or configuration issues")
        print(f"\nConsider:")
        print("  - Adjusting hyperparameter search ranges")
        print("  - Increasing n_trials")
        print("  - Checking your data and configuration")
        print(f"{'='*60}")
        
        # Save study even if no trials completed
        if args.storage is None:
            study_path = os.path.join(save_dir, f'{args.study_name}.pkl')
            with open(study_path, 'wb') as f:
                pickle.dump(study, f)
            print(f"\nStudy saved to {study_path} (no best trial available)")
        
        return
    
    # Get best trial
    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Validation Accuracy): {trial.value:.2f}%")
    print(f"  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Load best trial's architecture parameters and training history
    best_trial_num = trial.number
    best_trial_dir = os.path.join(save_dir, f'trial_{best_trial_num}')
    
    # Load architecture parameters
    arch_params_path = os.path.join(best_trial_dir, 'architecture_params.json')
    architecture_params = {}
    if os.path.exists(arch_params_path):
        with open(arch_params_path, 'r') as f:
            architecture_params = json.load(f)
    
    # Load training history
    history_path = os.path.join(best_trial_dir, 'training_history.json')
    training_history = {}
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            training_history = json.load(f)
    
    # Save best hyperparameters (including architecture params)
    best_params = {
        'training_hyperparameters': trial.params,
        'architecture_parameters': architecture_params,
        'best_validation_accuracy': trial.value,
        'trial_number': best_trial_num
    }
    best_params_path = os.path.join(save_dir, 'best_hyperparameters.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest hyperparameters (including architecture) saved to {best_params_path}")
    
    # Save training history for plotting
    if training_history:
        final_history_path = os.path.join(save_dir, 'best_trial_training_history.json')
        with open(final_history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Best trial training history saved to {final_history_path}")
    
    # Save best model checkpoint if checkpointing is enabled
    if args.save_checkpoints:
        best_trial_checkpoint = os.path.join(best_trial_dir, 'best_model.pth')
        
        if os.path.exists(best_trial_checkpoint):
            # Copy best model to main directory
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            shutil.copy2(best_trial_checkpoint, best_model_path)
            print(f"Best model checkpoint saved to {best_model_path}")
        else:
            print(f"Warning: Best trial checkpoint not found at {best_trial_checkpoint}")
            print("The checkpoint may not have been saved during the trial.")
    
    # Save study
    if args.storage is None:
        study_path = os.path.join(save_dir, f'{args.study_name}.pkl')
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        print(f"Study saved to {study_path}")
    
    # Plot optimization history
    try:
        import optuna.visualization as vis
        
        # Create plots directory
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_image(os.path.join(plots_dir, 'optimization_history.png'))
        print(f"Optimization history plot saved")
        
        # Parameter importance
        try:
            fig = vis.plot_param_importances(study)
            fig.write_image(os.path.join(plots_dir, 'param_importances.png'))
            print(f"Parameter importance plot saved")
        except Exception as e:
            print(f"Could not generate parameter importance plot: {e}")
        
        # Parallel coordinate plot
        try:
            fig = vis.plot_parallel_coordinate(study)
            fig.write_image(os.path.join(plots_dir, 'parallel_coordinate.png'))
            print(f"Parallel coordinate plot saved")
        except Exception as e:
            print(f"Could not generate parallel coordinate plot: {e}")
        
    except ImportError:
        print("Plotly not available. Install with: pip install plotly kaleido")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
