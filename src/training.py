"""
Training Loop for Autoencoder Anomaly Detection
Clean, standard, beginner-friendly PyTorch training code.
"""

import time
import copy
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from src.models import compute_reconstruction_error


def train_autoencoder(model, train_loader, val_loader, X_val, y_val,
                      epochs=35, lr=1e-3, weight_decay=1e-5, patience=7,
                      device='cpu', loss_type='mse', verbose=True):
    """
    Standard PyTorch training loop with early stopping.
    
    1. Loop through epochs
    2. For each batch: forward pass -> calculate loss -> backward pass -> update weights
    3. Check validation error at the end of each epoch
    4. If validation error stops improving for 'patience' epochs, stop early!
    """
    model = model.to(device)
    
    # 1. Choose loss function (MSE)
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'huber':
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
        
    # 2. Choose optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 3. Learning rate scheduler: slows down learning if loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # History dictionary to save our results for plotting
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_ap': [],
        'time_per_epoch': []
    }
    
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    if verbose:
        print(f"=== Training Autoencoder on {str(device).upper()} ===")
        print(f"Total Epochs: {epochs} | Batch Size: {train_loader.batch_size} | Learning Rate: {lr}")
        print("-" * 65)
        
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # ----------------------------------------------------
        # Part A: Training Phase (Honest Transactions)
        # ----------------------------------------------------
        model.train()
        total_train_loss = 0.0
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            # Step 1: Zero out past gradients
            optimizer.zero_grad()
            
            # Step 2: Forward pass (reconstruct the batch)
            reconstructed_x = model(batch_x)
            
            # Step 3: Compute reconstruction loss (MSE)
            loss = criterion(reconstructed_x, batch_x)
            
            # Step 4: Backpropagation (compute gradients)
            loss.backward()
            
            # Step 5: Update network weights
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ----------------------------------------------------
        # Part B: Validation Phase (Checking on practice exam)
        # ----------------------------------------------------
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                reconstructed_x = model(batch_x)
                loss = criterion(reconstructed_x, batch_x)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # ----------------------------------------------------
        # Part C: Check Anomaly Detection Score (Validation AP)
        # ----------------------------------------------------
        val_errors = compute_reconstruction_error(model, X_val_tensor, device=device, loss_type='mse')
        val_ap = average_precision_score(y_val, val_errors)
        
        # Adjust learning rate if validation loss stopped dropping
        scheduler.step(avg_val_loss)
        elapsed = time.time() - t0
        
        # Record metrics
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_ap'].append(val_ap)
        history['time_per_epoch'].append(elapsed)
        
        if verbose:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val AP: {val_ap:.4f} | ({elapsed:.1f}s)")
            
        # ----------------------------------------------------
        # Part D: Early Stopping Check
        # ----------------------------------------------------
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"--> Early stopping at epoch {epoch} (loss stopped improving). Restoring best model!")
                break
                
    # Load the best saved weights
    model.load_state_dict(best_weights)
    if verbose:
        print(f"=== Done! Best Validation Loss achieved: {best_val_loss:.4f} ===")
        
    return model, history
