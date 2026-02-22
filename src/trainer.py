import torch
import numpy as np
from tqdm.auto import tqdm
from .diffusion_model import compute_loss

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, 
                num_epochs=50, cfg_scale=1.5, cfg_dropout=0.1, schedule=None):
    """
    Trains the diffusion model.
    
    Args:
        model: DiffusionDenoiser instance
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        device: torch.device
        num_epochs: Number of epochs
        cfg_scale: Classifier-Free Guidance scale (for logging/reference)
        cfg_dropout: Dropout probability for CFG
        schedule: DiffusionSchedule object
        
    Returns:
        dict: Training history and best model state
    """
    print("=" * 80)
    print(f"STARTING TRAINING ({device})")
    print("=" * 80)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0
        
        # Training phase
        # Use simple progress bar for epochs logic, or just log occasionally
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            # Forward pass with CFG (randomly drops conditions)
            loss = compute_loss(model, targets, features, schedule, use_cfg=True)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            n_batches += 1
            
        avg_train_loss = epoch_train_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device).unsqueeze(1)
                
                # Validation without CFG dropout (standard forward pass)
                # We want to measure the "best" reconstruction error
                loss = compute_loss(model, targets, features, schedule, use_cfg=False)
                epoch_val_loss += loss.item()
                n_val_batches += 1
        
        avg_val_loss = epoch_val_loss / n_val_batches
        val_losses.append(avg_val_loss)
        
        # Scheduler step
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
            
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        # Log
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.6f}")
            
    print("=" * 80)
    print(f"TRAINING COMPLETE. Best Val Loss: {best_val_loss:.6f}")
    print("=" * 80)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_model_state': best_model_state,
        'best_val_loss': best_val_loss
    }
