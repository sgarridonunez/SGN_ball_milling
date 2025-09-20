#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for SGN using modularized data utilities and model definitions.
"""

import pickle
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

# Import from custom modules
import data_utils
import models

#######################
# --- Configuration --- #
#######################
# Data parameters
DATA_FILE = "extracted_data_300_fine.pkl"
STL_FILE = "ball_mill_jar.stl" # Needed if mesh processing is done here, currently in data_utils
NORM_PARAMS_FILE = "normalization_params.pkl"
LOSS_HISTORY_FILE = "loss_history_modular.pkl"
CHECKPOINT_PATH = "sgn_checkpoint_modular.pth"
BEST_MODEL_PATH = "best_model_modular.pth"

# Model hyperparameters
WINDOW_SIZE = 7
HIDDEN_DIM = 256
MLP_LAYERS = 4
INTERACTION_LAYERS = 1
NODE_OUT_DIM = 3
GLOBAL_OUT_DIM = 1
EDGE_IN_DIM = 3 # Based on distance_vector and sdf_distance_vectors being 3D
AGGREGATION = "mean"
DROPOUT_RATE = 0.0
USE_LAST_SNAPSHOT_GLOBAL = False

# Training parameters
NUM_TRAIN_SAMPLES = 45000
BATCH_SIZE = 2
NUM_EPOCHS = 2000
INITIAL_LR = 1e-4
FINAL_LR = 1e-6
HUBER_BETA = 2.0
LOSS_ALPHA = 3 # Weight for node loss
VELOCITY_NOISE_STD = 0.005 # Noise added during training
RESUME_TRAINING = True # Flag to resume from checkpoint

# Other constants
MASS = 0.00403171 # kg
PARTICLE_FEATURE_DIM = data_utils.PARTICLE_FEATURE_DIM # Get from data_utils

#################################
# --- 1. Load and Prepare Data --- #
#################################

# Load raw data
try:
    extracted_data = data_utils.load_extracted_data(DATA_FILE)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Build dataset with time windows
# Note: Mesh loading/processing happens within data_utils if needed by feature building
sgn_dataset = data_utils.build_sgn_dataset_window(
    extracted_data,
    window_size=WINDOW_SIZE,
    mass=MASS,
    use_last_snapshot_global=USE_LAST_SNAPSHOT_GLOBAL
)

if not sgn_dataset:
    print("Dataset creation failed or resulted in an empty dataset. Exiting.")
    exit()

# Split dataset
if NUM_TRAIN_SAMPLES >= len(sgn_dataset):
     print(f"Warning: NUM_TRAIN_SAMPLES ({NUM_TRAIN_SAMPLES}) >= total samples ({len(sgn_dataset)}). Using all data for training.")
     train_dataset = sgn_dataset
     test_dataset = []
else:
     train_dataset = sgn_dataset[:NUM_TRAIN_SAMPLES]
     test_dataset = sgn_dataset[NUM_TRAIN_SAMPLES:]
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


# Calculate or load normalization parameters
if os.path.exists(NORM_PARAMS_FILE) and RESUME_TRAINING: # Load if resuming and file exists
    try:
        normalization_params = data_utils.load_normalization_params(NORM_PARAMS_FILE)
    except FileNotFoundError:
        print(f"Normalization file {NORM_PARAMS_FILE} not found. Calculating from training data.")
        normalization_params_np = data_utils.calculate_normalization_params(train_dataset)
        data_utils.save_normalization_params(normalization_params_np, NORM_PARAMS_FILE)
        normalization_params = data_utils.load_normalization_params(NORM_PARAMS_FILE) # Reload as tensors
else: # Calculate and save if not resuming or file doesn't exist
    normalization_params_np = data_utils.calculate_normalization_params(train_dataset)
    data_utils.save_normalization_params(normalization_params_np, NORM_PARAMS_FILE)
    normalization_params = data_utils.load_normalization_params(NORM_PARAMS_FILE) # Load as tensors


# Normalize datasets
# Note: Noise is added dynamically during training, so add_noise=False here
train_dataset = data_utils.normalize_dataset_separate(train_dataset, normalization_params, add_noise=False)
if test_dataset:
    test_dataset = data_utils.normalize_dataset_separate(test_dataset, normalization_params, add_noise=False)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True) if test_dataset else None

#######################################
# --- 2. Initialize Model & Training Components --- #
#######################################

# Determine input dimension based on window size and feature dim
node_in_dim = PARTICLE_FEATURE_DIM * WINDOW_SIZE

# Initialize model
model = models.SGN(
    node_in_dim=node_in_dim,
    edge_in_dim=EDGE_IN_DIM,
    hidden_dim=HIDDEN_DIM,
    node_out_dim=NODE_OUT_DIM,
    global_out_dim=GLOBAL_OUT_DIM,
    mlp_layers=MLP_LAYERS,
    interaction_layers=INTERACTION_LAYERS,
    aggregation=AGGREGATION,
    dropout_rate=DROPOUT_RATE,
    use_last_snapshot_global=USE_LAST_SNAPSHOT_GLOBAL,
    particle_feature_dim=PARTICLE_FEATURE_DIM # Pass this for global encoder calculation
)

# Setup device and potential DataParallel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model) # Assuming device_ids=[0, 1] implicitly or adjust as needed

# Loss function
huber_loss = nn.SmoothL1Loss(beta=HUBER_BETA)

# Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
# Calculate gamma for exponential decay
if NUM_EPOCHS > 0:
    gamma = (FINAL_LR / INITIAL_LR) ** (1 / NUM_EPOCHS)
else:
    gamma = 1.0 # No decay if 0 epochs
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# Training state variables
train_loss_history = []
test_loss_history = []
start_epoch = 0
best_train_loss = float('inf')
best_epoch = 0

#######################################
# --- 3. Resume from Checkpoint --- #
#######################################

if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device) # Load to target device

        # Handle DataParallel state dict keys
        state_dict = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
            # If current model is DataParallel but checkpoint isn't, add 'module.' prefix
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not isinstance(model, nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
             # If current model isn't DataParallel but checkpoint is, remove 'module.' prefix
             state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1 # Use .get for safety
        train_loss_history = checkpoint.get('train_loss_history', [])
        test_loss_history = checkpoint.get('test_loss_history', [])
        best_train_loss = checkpoint.get('best_train_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', 0)
        print(f"Resumed training from epoch {start_epoch}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}. Starting training from scratch.")
        start_epoch = 0 # Ensure starting from scratch if loading fails

#######################
# --- 4. Training Loop --- #
#######################

print(f"Starting training for {NUM_EPOCHS - start_epoch} epochs...")

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    total_train_huber_loss = 0.0
    total_train_mse_loss_node = 0.0
    total_train_global_huber = 0.0
    total_train_global_mse = 0.0
    train_batches = 0

    for data in train_loader:
        data = data.to(device)

        # --- Dynamic Noise Injection ---
        if VELOCITY_NOISE_STD > 0:
            with torch.no_grad():
                num_nodes = data.x.size(0)
                if num_nodes > 1: # Ensure there's at least one particle node
                    num_particles = num_nodes - 1 # Assuming wall node is last
                    # Create indices for velocity columns across all windows
                    velocity_indices = []
                    for w in range(WINDOW_SIZE):
                        start_col = w * PARTICLE_FEATURE_DIM
                        velocity_indices.extend(range(start_col, start_col + 3))

                    if velocity_indices: # Check if indices were actually generated
                        noise = VELOCITY_NOISE_STD * torch.randn(num_particles, len(velocity_indices), device=device)
                        data.x[:num_particles, velocity_indices] += noise
        # --- End Noise Injection ---

        optimizer.zero_grad()
        node_pred, global_pred = model(data)

        # Compute Huber losses (exclude wall node from node loss)
        num_nodes = data.y.size(0)
        if num_nodes > 1: # Check if particle nodes exist
            loss_node = huber_loss(node_pred[:-1], data.y[:-1])
        else: # Handle case with only wall node (or empty graph?)
            loss_node = torch.tensor(0.0, device=device) # No particle loss if no particles

        loss_global = huber_loss(global_pred, data.global_target)
        loss = LOSS_ALPHA * loss_node + loss_global

        loss.backward()
        optimizer.step()

        # Accumulate losses for monitoring
        total_train_huber_loss += loss.item()
        if num_nodes > 1:
            mse_node = F.mse_loss(node_pred[:-1], data.y[:-1]).item()
            total_train_mse_loss_node += mse_node
        mse_global = F.mse_loss(global_pred, data.global_target).item()
        total_train_global_mse += mse_global
        total_train_global_huber += loss_global.item()

        train_batches += 1

    # Calculate average training losses for the epoch
    avg_train_huber = total_train_huber_loss / train_batches if train_batches > 0 else 0
    avg_train_mse_node = total_train_mse_loss_node / train_batches if train_batches > 0 else 0
    avg_train_global_huber = total_train_global_huber / train_batches if train_batches > 0 else 0
    avg_train_global_mse = total_train_global_mse / train_batches if train_batches > 0 else 0
    train_loss_history.append(avg_train_huber)

    # --- Testing Loop ---
    avg_test_huber = 0.0
    avg_test_mse_node = 0.0
    avg_test_global_huber = 0.0
    avg_test_global_mse = 0.0

    if test_loader:
        model.eval()
        total_test_huber = 0.0
        total_test_mse_loss_node = 0.0 # Initialize accumulator for node MSE
        total_test_global_huber = 0.0
        total_test_global_mse = 0.0
        test_batches = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                node_pred, global_pred = model(data)

                num_nodes = data.y.size(0)
                if num_nodes > 1:
                    loss_node = huber_loss(node_pred[:-1], data.y[:-1])
                    mse_node = F.mse_loss(node_pred[:-1], data.y[:-1]).item()
                    total_test_mse_loss_node += mse_node
                else:
                    loss_node = torch.tensor(0.0, device=device)

                loss_global = huber_loss(global_pred, data.global_target)
                batch_loss = LOSS_ALPHA * loss_node + loss_global
                batch_loss_orig_calc = loss_node + loss_global
                total_test_huber += batch_loss_orig_calc.item()

                mse_global = F.mse_loss(global_pred, data.global_target).item()
                total_test_global_mse += mse_global
                total_test_global_huber += loss_global.item()
                test_batches += 1

        avg_test_huber = total_test_huber / test_batches if test_batches > 0 else 0
        avg_test_mse_node = total_test_mse_loss_node / test_batches if test_batches > 0 else 0 # Use correct accumulator
        avg_test_global_huber = total_test_global_huber / test_batches if test_batches > 0 else 0
        avg_test_global_mse = total_test_global_mse / test_batches if test_batches > 0 else 0
        test_loss_history.append(avg_test_huber)

    # Step the scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # Print epoch summary
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, LR: {current_lr:.6e}")
    print(f"  Train Huber Loss: {avg_train_huber:.6f} (Node MSE: {avg_train_mse_node:.6f}, Global Huber: {avg_train_global_huber:.6f}, Global MSE: {avg_train_global_mse:.6f})")
    if test_loader:
        print(f"  Test Huber Loss:  {avg_test_huber:.6f} (Node MSE: {avg_test_mse_node:.6f}, Global Huber: {avg_test_global_huber:.6f}, Global MSE: {avg_test_global_mse:.6f})")

    # Save best model based on training loss
    if avg_train_huber < best_train_loss:
        best_train_loss = avg_train_huber
        best_epoch = epoch + 1
        # Save model state dict, handling DataParallel if necessary
        model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(model_state_dict, BEST_MODEL_PATH)
        print(f"  New best model saved to {BEST_MODEL_PATH} (Epoch {best_epoch}, Loss: {best_train_loss:.6f})")

    # Save checkpoint
    try:
        # Prepare state dict, handling DataParallel
        model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss_history': train_loss_history,
            'test_loss_history': test_loss_history,
            'best_train_loss': best_train_loss,
            'best_epoch': best_epoch,
            'config': { # Save config used for this run
                 'window_size': WINDOW_SIZE, 'hidden_dim': HIDDEN_DIM, 'mlp_layers': MLP_LAYERS,
                 'interaction_layers': INTERACTION_LAYERS, 'dropout_rate': DROPOUT_RATE,
                 'use_last_snapshot_global': USE_LAST_SNAPSHOT_GLOBAL, 'batch_size': BATCH_SIZE,
                 'initial_lr': INITIAL_LR, 'final_lr': FINAL_LR, 'huber_beta': HUBER_BETA,
                 'loss_alpha': LOSS_ALPHA, 'velocity_noise_std': VELOCITY_NOISE_STD
            }
        }, CHECKPOINT_PATH)
    except Exception as e:
        print(f"  Error saving checkpoint: {e}")
 
 

#################################
# --- 5. Post-Training --- #
#################################

print("Training finished.")
print(f"Best model achieved at Epoch {best_epoch} with Training Huber Loss: {best_train_loss:.6f}")

# Plot loss history
if train_loss_history or test_loss_history:
    plt.figure(figsize=(8, 6))
    if train_loss_history:
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Huber Loss')
    if test_loss_history:
        plt.plot(range(1, len(test_loss_history) + 1), test_loss_history, label='Test Huber Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Overall Huber Loss vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plot_modular.png") # Save the plot
    print("Loss plot saved to loss_plot_modular.png")
    # plt.show() # Optionally display the plot

# Save final loss history
loss_history_data = {
    'train_loss_history': train_loss_history,
    'test_loss_history': test_loss_history,
    'best_epoch': best_epoch,
    'best_train_loss': best_train_loss
}
try:
    with open(LOSS_HISTORY_FILE, "wb") as f:
        pickle.dump(loss_history_data, f)
    print(f"Saved final loss history to {LOSS_HISTORY_FILE}.")
except Exception as e:
    print(f"Error saving loss history: {e}")

# Optional: Final evaluation printout
if test_loader:
    model.eval()
    with torch.no_grad():
        # Load best model weights for final eval if desired
        if os.path.exists(BEST_MODEL_PATH):
             print(f"Loading best model from {BEST_MODEL_PATH} for final evaluation.")
             best_state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
             # Handle DataParallel keys if necessary when loading
             if isinstance(model, nn.DataParallel) and not list(best_state_dict.keys())[0].startswith('module.'):
                 best_state_dict = {'module.' + k: v for k, v in best_state_dict.items()}
             elif not isinstance(model, nn.DataParallel) and list(best_state_dict.keys())[0].startswith('module.'):
                 best_state_dict = {k.partition('module.')[2]: v for k, v in best_state_dict.items()}
             model.load_state_dict(best_state_dict)

        data = next(iter(test_loader)) # Get one batch
        data = data.to(device)
        node_pred, global_pred = model(data)
        print("\n--- Final Evaluation (Sample Batch) ---")
        print("Node prediction shape:", node_pred.shape)
        print("Global prediction shape:", global_pred.shape)
        # Add more detailed evaluation metrics if needed
else:
    print("\nNo test data loaded, skipping final evaluation printout.")

print("Modular script execution complete.")
