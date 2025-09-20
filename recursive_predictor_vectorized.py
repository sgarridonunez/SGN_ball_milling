#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursive Simulation Script using Modular SGN Model - CoM Opt (PreCalc dt_est Spline) + Original Contacts/Gradient Method (No Profiling)

This script performs a recursive simulation using a pre-trained SGN model.
It initializes with a window of ground truth data, then iteratively:
  - Predicts particle accelerations and global energy using the SGN model.
  - Updates particle positions and velocities using an integration scheme.
  - Re-computes contacts (Original O(N^2)), SDF values, and gradients (Finite Difference)
    based on the updated state and interpolated wall position (using pre-calculated periodic spline with dt_est extension).
  - Constructs the input features for the next step.
  - Compares results against ground truth data (if available) and saves outputs.
"""

#######################
# --- Configuration --- #
#######################
# File Paths
DATA_FILE = "extracted_data_300_fine.pkl" # Ground truth data for initialization and comparison
STL_FILE = "ball_mill_jar.stl"
NORM_PARAMS_FILE = "normalization_params.pkl" # From training
MODEL_PATH = "best_model_modular.pth" # Trained model from main_modular.py

# Simulation Parameters
WINDOW_SIZE = 7 # Must match the window size used during training
DT = 0.0001 # Simulation time step (s)
TOTAL_SIM_TIME = 15 # Restored simulation time
INTEGRATION_TYPE = "euler" # 'euler' or 'trapezoidal'
RPM = 300 # Wall rotational speed (used for wall features)

# Model Hyperparameters (Must match the trained model!)
HIDDEN_DIM = 256
MLP_LAYERS = 4
INTERACTION_LAYERS = 1 # Ensure this matches your training configuration in main_modular.py
NODE_OUT_DIM = 3
GLOBAL_OUT_DIM = 1
EDGE_IN_DIM = 3
AGGREGATION = "mean"
DROPOUT_RATE = 0.0
USE_LAST_SNAPSHOT_GLOBAL = False # Must match training

# Contact & Correction Parameters
CONTACT_THRESHOLD_PP = 0.0015 # Particle-particle contact distance threshold
CONTACT_THRESHOLD_PW_SDF = -0.0052 # Particle-wall contact SDF threshold
PENETRATION_THRESHOLD_CORRECTION = -0.0051 # SDF threshold for applying acceleration correction
PENETRATION_THRESHOLD_SNAPBACK = -0.0049 # SDF threshold for applying snap-back projection
ACCELERATION_CORRECTION_SCALE = 0.0 # Factor for acceleration correction (0 = off)
VELOCITY_DAMPING_FACTOR = 1.0 # Factor for velocity damping upon contact (1 = off)

# Output Parameters
OUTPUT_POSITIONS_FILE = "positions_recursive_vectorized.pkl"
OUTPUT_COMS_FILE = "coms_recursive_vectorized.pkl"
OUTPUT_RESULTS_FILE = "model_results_recursive_vectorized.pkl"
OUTPUT_PLOT_MSE_FILE = "mse_plot_recursive_vectorized.png"
OUTPUT_PLOT_ENERGY_DIRECT_FILE = "energy_plot_direct_recursive_vectorized.png"
OUTPUT_PLOT_ENERGY_BLOCK_AVG_FILE = "energy_plot_block_avg_recursive_vectorized.png"
POSITION_SAVE_INTERVAL = 0.001 # Interval (s) to save particle positions
MOVING_AVERAGE_WINDOW = 5 # Window size for block averaging energy plot. This is typically unnecessary and was only used for debugging.

# Other Constants
MASS = 0.00403171 # kg (Must match training)

###############################
# 1. Imports
###############################
import numpy as np
import trimesh
import pickle
from scipy.interpolate import CubicSpline
import torch
from torch_geometric.data import Data
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
from sys import exit

# Import from custom modules
import data_utils
import models

####################################################
# 2. Load Data, Geometry, Normalization, and Model #
####################################################

# --- Load Ground Truth Data ---
try:
    extracted_data = data_utils.load_extracted_data(DATA_FILE)
    if not extracted_data: raise ValueError("Extracted data is empty.")
except Exception as e:
    print(f"Error loading/processing ground truth data: {e}")
    exit()

# --- Load Geometry ---
try:
    mesh_static = data_utils.load_and_transform_mesh(STL_FILE)
except Exception as e:
    print(f"Error loading STL geometry: {e}")
    exit()

# --- Load Normalization Parameters ---
try:
    normalization_params = data_utils.load_normalization_params(NORM_PARAMS_FILE)
    norm_tensors = {}
    for key, value in normalization_params.items():
        if value is not None:
            if not isinstance(value, torch.Tensor):
                 print(f"Warning: Normalization param '{key}' was not a tensor. Attempting conversion.")
                 try: norm_tensors[key] = torch.tensor(value).float()
                 except Exception as conv_err:
                     print(f"  Conversion failed: {conv_err}. Setting param '{key}' to None."); norm_tensors[key] = None
            else: norm_tensors[key] = value.float()
        else: norm_tensors[key] = None
except FileNotFoundError as e:
    print(f"Error loading normalization parameters: {e}")
    exit()

# --- Initialize Model ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move normalization tensors to device
for key in norm_tensors:
    if norm_tensors[key] is not None:
        norm_tensors[key] = norm_tensors[key].to(device)

node_in_dim = data_utils.PARTICLE_FEATURE_DIM * WINDOW_SIZE
model = models.SGN(
    node_in_dim=node_in_dim, edge_in_dim=EDGE_IN_DIM, hidden_dim=HIDDEN_DIM,
    node_out_dim=NODE_OUT_DIM, global_out_dim=GLOBAL_OUT_DIM, mlp_layers=MLP_LAYERS,
    interaction_layers=INTERACTION_LAYERS, aggregation=AGGREGATION, dropout_rate=DROPOUT_RATE,
    use_last_snapshot_global=USE_LAST_SNAPSHOT_GLOBAL, particle_feature_dim=data_utils.PARTICLE_FEATURE_DIM
)

# --- Load Trained Model Weights ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Trained model file not found at {MODEL_PATH}")
    exit()
try:
    model_state_dict = torch.load(MODEL_PATH, map_location=device)
    if list(model_state_dict.keys())[0].startswith('module.'):
        model_state_dict = {k.partition('module.')[2]: v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    print(f"Successfully loaded trained model from {MODEL_PATH} and set to eval mode.")
except Exception as e:
    print(f"Error loading model state dict: {e}")
    exit()

###################################################
# 3. Pre-calculate CoM Splines (Optimization)     #
###################################################
print("Pre-calculating CoM interpolation splines...")
com_splines = None
last_known_com_from_data = None # Store the last CoM from the input data

try:
    com_times = np.array([snap["time"] for snap in extracted_data])
    com_actual_coms_raw = np.array([snap["wall_node_features"][:3] for snap in extracted_data])
    unique_times, unique_indices = np.unique(com_times, return_index=True)

    if len(unique_times) < 2:
        print("Warning: Not enough unique time points to create CoM splines. CoM interpolation might fail.")
        if len(extracted_data) > 0:
             last_known_com_from_data = com_actual_coms_raw[-1]
    else:
        last_known_com_from_data = com_actual_coms_raw[unique_indices[-1]] # CoM at max time in data
        com_actual_coms = com_actual_coms_raw[unique_indices]
        dt_est = unique_times[1] - unique_times[0]
        # Use dt_est for time extension
        end_time_extended = unique_times[-1] + dt_est
        times_extended = np.append(unique_times, end_time_extended)
        x_extended = np.append(com_actual_coms[:, 0], com_actual_coms[0, 0])
        y_extended = np.append(com_actual_coms[:, 1], com_actual_coms[0, 1])
        z_extended = np.append(com_actual_coms[:, 2], com_actual_coms[0, 2])
        spline_x = CubicSpline(times_extended, x_extended, bc_type='periodic')
        spline_y = CubicSpline(times_extended, y_extended, bc_type='periodic')
        spline_z = CubicSpline(times_extended, z_extended, bc_type='periodic')
        com_splines = {'x': spline_x, 'y': spline_y, 'z': spline_z}
        print("CoM splines pre-calculated using dt_est extension.")

except Exception as e:
    print(f"Error pre-calculating CoM splines: {e}. CoM interpolation might fail.")
    if len(extracted_data) > 0:
        last_known_com_from_data = np.array(extracted_data[-1]["wall_node_features"][:3])

# Ensure a fallback if everything failed
if last_known_com_from_data is None:
    last_known_com_from_data = np.zeros(3)
    print("Warning: Could not determine last known CoM from data. Using [0,0,0] as fallback.")


###################################################
# 4. Helper Functions for Recursive Simulation    #
###################################################

def get_com_at_time_precalculated(query_time, splines, fallback_com):
    """Interpolate wall COM using pre-calculated spline objects, relying on periodic extrapolation."""
    if splines is None: return fallback_com
    try:
        # Directly query the spline, relying on its periodic nature for extrapolation
        x_val = splines['x'](query_time)
        y_val = splines['y'](query_time)
        z_val = splines['z'](query_time)
        return np.array([x_val, y_val, z_val])
    except Exception as e:
        # print(f"Error during spline evaluation at time {query_time}: {e}. Returning fallback CoM.")
        return fallback_com

def compute_contacts(positions, sdf_values, particle_ids, threshold_pp, sdf_threshold):
    """Computes particle-particle and particle-wall contacts based on thresholds (O(N^2) version)."""
    N = positions.shape[0]; contact_pairs_pp = []; distance_vectors_pp = []
    for i in range(N):
        for j in range(i + 1, N):
            vec = positions[j] - positions[i]; dist = np.linalg.norm(vec)
            if dist <= threshold_pp: contact_pairs_pp.append([particle_ids[i], particle_ids[j]]); distance_vectors_pp.append(vec)
    if contact_pairs_pp: contact_pairs_pp = np.array(contact_pairs_pp); distance_vectors_pp = np.array(distance_vectors_pp)
    else: contact_pairs_pp = np.empty((0, 2), dtype=int); distance_vectors_pp = np.empty((0, 3), dtype=float)
    contact_ids_pw = particle_ids[sdf_values.flatten() >= sdf_threshold]
    return {"contact_ids": contact_pairs_pp, "distance_vector": distance_vectors_pp}, {"contact_ids": contact_ids_pw}

def euler_integration(prev_state, linear_acc, dt):
    """Performs one step of Euler integration."""
    particle = prev_state["particle"]; positions = particle["positions"]; velocities = particle["velocities"]
    new_velocities = velocities + linear_acc * dt; new_positions = positions + velocities * dt
    return { "particle": { "positions": new_positions, "velocities": new_velocities, "ids": particle["ids"], "net_forces": linear_acc * MASS } }

def trapezoidal_integration(prev_state, acc_old, acc_new, dt):
    """Performs one step of Trapezoidal integration."""
    particle = prev_state["particle"]; positions_old = particle["positions"]; velocities_old = particle["velocities"]
    new_velocities = velocities_old + 0.5 * (acc_old + acc_new) * dt; new_positions = positions_old + 0.5 * (velocities_old + new_velocities) * dt
    return { "particle": { "positions": new_positions, "velocities": new_velocities, "ids": particle["ids"], "net_forces": acc_new * MASS } }

def SDF_gradient_vectorized(points, target_mesh, epsilon=1e-5):
    """Computes the gradient of the SDF for multiple points using central finite differences."""
    points = np.asarray(points)
    if points.ndim == 1: points = points.reshape(1, -1)
    num_points, dim = points.shape
    if dim != 3: raise ValueError("Input points must be 3D.")

    grad = np.zeros_like(points, dtype=float)
    for i in range(dim):
        d = np.zeros(dim); d[i] = epsilon
        points_plus = points + d
        points_minus = points - d
        sdf_plus = trimesh.proximity.signed_distance(target_mesh, points_plus)
        sdf_minus = trimesh.proximity.signed_distance(target_mesh, points_minus)
        grad[:, i] = (sdf_plus - sdf_minus) / (2 * epsilon)
    return grad

def create_new_snapshot_vectorized(prev_snapshot, new_particle_state, current_time, com_splines, fallback_com, mesh_static):
    """Creates a new snapshot dictionary using vectorized SDF and finite difference gradient."""
    new_snapshot = {}
    new_snapshot["time"] = current_time
    new_snapshot["timestep"] = prev_snapshot.get("timestep", 0) + 1
    new_snapshot["particle"] = new_particle_state["particle"].copy()

    # --- Calculate Wall Position and Mesh ---
    current_wall_com = get_com_at_time_precalculated(current_time, com_splines, fallback_com) # Use precalculated spline
    com_static_center = mesh_static.center_mass
    offset = current_wall_com - com_static_center
    moved_mesh = mesh_static.copy()
    moved_mesh.vertices = moved_mesh.vertices + offset

    # --- Vectorized SDF, Gradients (Finite Diff), and Distance Vectors ---
    positions = new_snapshot["particle"]["positions"]
    sdf_vals = data_utils.SDF_static(positions, moved_mesh).reshape(-1, 1)
    sdf_gradients = SDF_gradient_vectorized(positions, moved_mesh)

    norms = np.linalg.norm(sdf_gradients, axis=1, keepdims=True)
    normalized_gradients = np.where(norms < data_utils.EPSILON, sdf_gradients, sdf_gradients / norms)
    sdf_distance_vectors = sdf_vals * normalized_gradients

    new_snapshot["particle"]["sdf_values"] = sdf_vals
    new_snapshot["particle"]["sdf_gradients"] = sdf_gradients
    new_snapshot["particle"]["sdf_distance_vectors"] = sdf_distance_vectors

    # --- Compute Contacts (Original O(N^2) version) ---
    particle_ids = np.array(new_snapshot["particle"]["ids"])
    contacts_pp, contacts_pw = compute_contacts(
        positions, sdf_vals, particle_ids,
        threshold_pp=CONTACT_THRESHOLD_PP, sdf_threshold=CONTACT_THRESHOLD_PW_SDF
    )
    new_snapshot["contacts_particle_particle"] = contacts_pp
    new_snapshot["contacts_particle_wall"] = contacts_pw

    # --- Wall Features ---
    wall_feature = np.concatenate([current_wall_com, np.array([RPM])])
    new_snapshot["wall_node_features"] = wall_feature

    # --- Energy ---
    new_snapshot["energy_normal_increment"] = 0.0
    new_snapshot["energy_tangential_increment"] = 0.0

    return new_snapshot # Return only snapshot

# build_sgn_features_window_recursive definition
def build_sgn_features_window_recursive(window_snapshots):
    """Builds graph features for the recursive loop using snapshot dictionaries."""
    particle_features_list = []
    for snap in window_snapshots:
        particle = snap["particle"]; velocities = particle["velocities"]
        sdf_values = particle["sdf_values"].reshape(-1, 1); sdf_gradients = particle["sdf_gradients"]
        feat = np.concatenate([velocities, sdf_values, sdf_gradients], axis=1)
        if feat.shape[1] != data_utils.PARTICLE_FEATURE_DIM: raise ValueError(f"Expected {data_utils.PARTICLE_FEATURE_DIM} particle features, got {feat.shape[1]}")
        particle_features_list.append(feat)
    particle_features_window = np.concatenate(particle_features_list, axis=1)
    wall_features_list = []
    for snap in window_snapshots:
        wall_feat = np.array(snap["wall_node_features"]).reshape(1, -1)
        pad_size = data_utils.WALL_FEATURE_PAD_DIM - wall_feat.shape[1]
        if pad_size < 0: raise ValueError("Wall features exceed padding dim")
        if pad_size > 0: wall_feat = np.hstack([wall_feat, np.zeros((1, pad_size))])
        wall_features_list.append(wall_feat)
    wall_features_window = np.concatenate(wall_features_list, axis=1)
    node_features = np.vstack([particle_features_window, wall_features_window])
    last_snap = window_snapshots[-1]; particle = last_snap["particle"]
    particle_ids = np.array(particle["ids"]); num_particles = len(particle_ids)
    id_to_index = {pid: idx for idx, pid in enumerate(particle_ids)}; wall_node_index = num_particles
    contacts_pp = last_snap["contacts_particle_particle"]
    edge_index_pp = torch.empty((2, 0), dtype=torch.long); edge_attr_pp_input = np.empty((0, 3), dtype=np.float32)
    if contacts_pp["contact_ids"].size > 0:
        valid_pairs_mask = np.isin(contacts_pp["contact_ids"], particle_ids).all(axis=1)
        valid_contact_ids_pp = contacts_pp["contact_ids"][valid_pairs_mask]
        valid_distance_vectors_pp = contacts_pp["distance_vector"][valid_pairs_mask]
        if valid_contact_ids_pp.size > 0:
            indices_pairs = np.vectorize(id_to_index.get)(valid_contact_ids_pp)
            edge_index_pp = torch.tensor(indices_pairs.T, dtype=torch.long); edge_attr_pp_input = valid_distance_vectors_pp
    contacts_pw = last_snap["contacts_particle_wall"]
    edge_index_pw = torch.empty((2, 0), dtype=torch.long); edge_attr_pw_input = np.empty((0, 3), dtype=np.float32)
    if contacts_pw["contact_ids"].size > 0:
        valid_contact_ids_pw = contacts_pw["contact_ids"][np.isin(contacts_pw["contact_ids"], particle_ids)]
        if valid_contact_ids_pw.size > 0:
            particle_indices_pw = np.vectorize(id_to_index.get)(valid_contact_ids_pw)
            edge_index_pw = torch.tensor(np.vstack([np.full_like(particle_indices_pw, wall_node_index), particle_indices_pw]), dtype=torch.long)
            if "sdf_distance_vectors" in particle and particle_indices_pw.size > 0 and len(particle["sdf_distance_vectors"]) > max(particle_indices_pw):
                 edge_attr_pw_input = particle["sdf_distance_vectors"][particle_indices_pw]
            else: edge_attr_pw_input = np.zeros((len(particle_indices_pw), 3), dtype=np.float32)
    combined_edge_index = []; combined_edge_attr = []
    if edge_index_pp.size(1) > 0: combined_edge_index.append(edge_index_pp); combined_edge_attr.append(edge_attr_pp_input)
    if edge_index_pw.size(1) > 0: combined_edge_index.append(edge_index_pw); combined_edge_attr.append(edge_attr_pw_input)
    if combined_edge_index:
        final_edge_index = torch.cat(combined_edge_index, dim=1)
        final_edge_attr_np = np.concatenate(combined_edge_attr, axis=0) if combined_edge_attr else np.empty((0, EDGE_IN_DIM), dtype=np.float32)
        final_edge_attr = torch.tensor(final_edge_attr_np, dtype=torch.float)
    else: final_edge_index = torch.empty((2, 0), dtype=torch.long); final_edge_attr = torch.empty((0, EDGE_IN_DIM), dtype=torch.float)
    net_forces = np.array(particle["net_forces"]); y_particles = net_forces / MASS
    wall_target = np.array(last_snap["wall_node_features"])[:NODE_OUT_DIM].reshape(1, NODE_OUT_DIM); y_node = np.vstack([y_particles, wall_target])
    global_target_val = last_snap.get("energy_normal_increment", 0.0) + last_snap.get("energy_tangential_increment", 0.0)
    data = Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=final_edge_index, edge_attr=final_edge_attr, y=torch.tensor(y_node, dtype=torch.float))
    data.time = last_snap["time"]; data.global_target = torch.tensor([global_target_val], dtype=torch.float)
    data.edge_attr_pp = torch.tensor(edge_attr_pp_input, dtype=torch.float); data.edge_attr_pw = torch.tensor(edge_attr_pw_input, dtype=torch.float)
    data.edge_index_pp = edge_index_pp; data.edge_index_pw = edge_index_pw
    if USE_LAST_SNAPSHOT_GLOBAL:
        last_particle_features = particle_features_list[-1]; last_wall_features = wall_features_list[-1]
        last_node_features = np.vstack([last_particle_features, last_wall_features])
        data.x_last = torch.tensor(last_node_features, dtype=torch.float)
    return data


####################################
# 5. Recursive Simulation Loop     #
####################################

def main_simulation_loop(com_splines, fallback_com): # Use fallback_com from pre-calculation
    # --- Initialization ---
    if len(extracted_data) < WINDOW_SIZE:
        print(f"Error: Need at least {WINDOW_SIZE} snapshots in {DATA_FILE} for initialization.")
        return None

    window = [snap.copy() for snap in extracted_data[:WINDOW_SIZE]]
    current_time = window[-1]["time"]
    print(f"Initialization complete. Starting recursive prediction from time: {current_time:.6f} s")

    simulation_results = []
    positions_dict = {}
    last_store_time = current_time

    initial_snapshot = window[-1]
    initial_net_forces = np.array(initial_snapshot["particle"]["net_forces"])
    acc_old = initial_net_forces / MASS

    mse_energy_dict, mse_x_dict, mse_y_dict, mse_z_dict, mse_sdf_dict = {}, {}, {}, {}, {}
    pred_energy_increments, gt_energy_increments = [], []

    # --- Main Loop ---
    while current_time < TOTAL_SIM_TIME:
        step_start_time = current_time

        # 1. Build graph input
        try: graph_input = build_sgn_features_window_recursive(window)
        except ValueError as e: print(f"Error building graph: {e}"); break
        graph_input = graph_input.to(device)

        # 2. Normalize graph input
        graph_input_list = data_utils.normalize_dataset_separate([graph_input], norm_tensors, add_noise=False)
        graph_input_normalized = graph_input_list[0]
        if not hasattr(graph_input_normalized, 'batch') or graph_input_normalized.batch is None:
            graph_input_normalized.batch = torch.zeros(graph_input_normalized.x.size(0), dtype=torch.long, device=device)

        # 3. Model prediction
        with torch.no_grad():
            try: node_pred_norm, global_pred_norm = model(graph_input_normalized)
            except Exception as e: print(f"Error predicting: {e}"); break

        # 4. Denormalize predictions
        linear_acc_norm = node_pred_norm[:-1].cpu().numpy()
        acc_new = linear_acc_norm * norm_tensors['norm_target_std'][:NODE_OUT_DIM].cpu().numpy() + norm_tensors['norm_target_mean'][:NODE_OUT_DIM].cpu().numpy()
        global_energy_increment_predicted = global_pred_norm.item() * norm_tensors['norm_global_std'].item() + norm_tensors['norm_global_mean'].item()
        pred_energy_increments.append(global_energy_increment_predicted)

        # 5. Acceleration correction (vectorized)
        if ACCELERATION_CORRECTION_SCALE > 0:
            prev_sdf_vals = window[-1]["particle"]["sdf_values"].flatten()
            prev_sdf_grads = window[-1]["particle"]["sdf_gradients"]
            correction_mask = prev_sdf_vals > PENETRATION_THRESHOLD_CORRECTION
            if np.any(correction_mask):
                 grads_to_correct = prev_sdf_grads[correction_mask]; acc_to_correct = acc_new[correction_mask]
                 grad_norms = np.linalg.norm(grads_to_correct, axis=1, keepdims=True)
                 valid_grad_mask = (grad_norms > data_utils.EPSILON).flatten()
                 if np.any(valid_grad_mask):
                     normal_dirs = grads_to_correct[valid_grad_mask] / grad_norms[valid_grad_mask]
                     accel_dots = np.einsum('ij,ij->i', acc_to_correct[valid_grad_mask], normal_dirs)
                     correction_needed_mask = accel_dots > 0
                     if np.any(correction_needed_mask):
                         corrections = accel_dots[correction_needed_mask, np.newaxis] * normal_dirs[correction_needed_mask]
                         indices_to_correct = np.where(correction_mask)[0][valid_grad_mask][correction_needed_mask]
                         acc_new[indices_to_correct] -= ACCELERATION_CORRECTION_SCALE * corrections

        # 6. Integration step
        prev_snapshot = window[-1]
        if INTEGRATION_TYPE.lower() == "euler": new_particle_state = euler_integration(prev_snapshot, acc_new, DT)
        elif INTEGRATION_TYPE.lower() == "trapezoidal": new_particle_state = trapezoidal_integration(prev_snapshot, acc_old, acc_new, DT)
        else: print(f"Error: Unknown INTEGRATION_TYPE: {INTEGRATION_TYPE}"); break

        # 7. Velocity damping (vectorized)
        if VELOCITY_DAMPING_FACTOR != 1.0:
             prev_sdf_vals = window[-1]["particle"]["sdf_values"].flatten()
             damping_mask = prev_sdf_vals > PENETRATION_THRESHOLD_CORRECTION
             if np.any(damping_mask): new_particle_state["particle"]["velocities"][damping_mask] *= VELOCITY_DAMPING_FACTOR

        # 8. Snap-back projection (vectorized SDF check)
        next_time = step_start_time + DT
        # Use the precalculated CoM function (which relies on spline's periodic extrapolation)
        current_wall_com = get_com_at_time_precalculated(next_time, com_splines, fallback_com)
        com_static_center = mesh_static.center_mass
        offset = current_wall_com - com_static_center
        moved_mesh_next = mesh_static.copy(); moved_mesh_next.vertices += offset

        current_positions = new_particle_state["particle"]["positions"]
        sdf_vals_next = data_utils.SDF_static(current_positions, moved_mesh_next)
        snapback_mask = sdf_vals_next > PENETRATION_THRESHOLD_SNAPBACK
        if np.any(snapback_mask):
            positions_to_correct = current_positions[snapback_mask]
            grads_snapback = SDF_gradient_vectorized(positions_to_correct, moved_mesh_next)
            grad_norms = np.linalg.norm(grads_snapback, axis=1, keepdims=True)
            valid_grad_mask_sb = (grad_norms > data_utils.EPSILON).flatten()
            if np.any(valid_grad_mask_sb):
                 positions_to_correct_valid = positions_to_correct[valid_grad_mask_sb]
                 sdf_vals_to_correct = sdf_vals_next[snapback_mask][valid_grad_mask_sb]
                 normal_dirs = grads_snapback[valid_grad_mask_sb] / grad_norms[valid_grad_mask_sb] # Use normalized finite diff gradient
                 correction_distances = sdf_vals_to_correct - PENETRATION_THRESHOLD_SNAPBACK
                 pos_corrected = positions_to_correct_valid - correction_distances[:, np.newaxis] * normal_dirs
                 indices_to_snapback = np.where(snapback_mask)[0][valid_grad_mask_sb]
                 new_particle_state["particle"]["positions"][indices_to_snapback] = pos_corrected

        # 9. Create New Snapshot (using vectorized function)
        try:
            # Pass precalculated splines and fallback CoM
            new_snapshot = create_new_snapshot_vectorized(prev_snapshot, new_particle_state, next_time,
                                                          com_splines, fallback_com, mesh_static)
            new_snapshot["predicted_energy_increment"] = global_energy_increment_predicted
        except Exception as e: print(f"Error creating snapshot: {e}"); break

        simulation_results.append(new_snapshot)

        # 10. Store Positions Periodically
        if current_time == window[-1]["time"] or (next_time - last_store_time >= POSITION_SAVE_INTERVAL - (DT/2.0)):
            positions_dict[next_time] = new_snapshot["particle"]["positions"].copy(); last_store_time = next_time

        # 11. Ground Truth Comparison
        matched_dem = None; min_time_diff = float('inf')
        for dem_snap in extracted_data:
            time_diff = abs(dem_snap["time"] - next_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff; matched_dem = dem_snap if time_diff < (DT / 1.9) else None
        if matched_dem:
            gt_time = matched_dem["time"]; sim_positions = new_snapshot["particle"]["positions"]; dem_positions = matched_dem["particle"]["positions"]
            if sim_positions.shape == dem_positions.shape:
                mse_x = np.mean((sim_positions[:, 0] - dem_positions[:, 0]) ** 2); mse_y = np.mean((sim_positions[:, 1] - dem_positions[:, 1]) ** 2); mse_z = np.mean((sim_positions[:, 2] - dem_positions[:, 2]) ** 2)
                mse_x_dict[gt_time] = mse_x; mse_y_dict[gt_time] = mse_y; mse_z_dict[gt_time] = mse_z
                sim_sdf = new_snapshot["particle"]["sdf_values"]; dem_sdf = matched_dem["particle"]["sdf_values"]
                mse_sdf = np.mean((sim_sdf - dem_sdf) ** 2); mse_sdf_dict[gt_time] = mse_sdf
                gt_inc_n = matched_dem.get("energy_normal_increment", 0.0)
                gt_inc_t = matched_dem.get("energy_tangential_increment", 0.0)
                gt_inc = float(gt_inc_n[0] if isinstance(gt_inc_n, (np.ndarray, list)) and len(gt_inc_n)>0 else gt_inc_n) + \
                         float(gt_inc_t[0] if isinstance(gt_inc_t, (np.ndarray, list)) and len(gt_inc_t)>0 else gt_inc_t)
                gt_energy_increments.append(gt_inc); diff_energy = global_energy_increment_predicted - gt_inc; mse_energy_dict[gt_time] = diff_energy**2
            else: gt_energy_increments.append(np.nan)
        else: gt_energy_increments.append(np.nan)

        # 12. Update Window and Time
        window.pop(0); window.append(new_snapshot)
        current_time = next_time; acc_old = acc_new
        if int(current_time / DT) % 100 == 0: print(f"Simulated up to time: {current_time:.6f} s")

    print("Recursive simulation finished.")
    return simulation_results, positions_dict, mse_x_dict, mse_y_dict, mse_z_dict, mse_sdf_dict, mse_energy_dict, pred_energy_increments, gt_energy_increments


###############################
# 6. Main Execution           #
###############################
if __name__ == "__main__":
    # Start timer
    start_run_time = time.time()

    # Run the main simulation loop
    # Pass precalculated splines and the fallback CoM
    results = main_simulation_loop(com_splines, last_known_com_from_data)

    # End timer
    end_run_time = time.time()
    print(f"\nTotal Simulation Loop Time: {end_run_time - start_run_time:.3f} seconds")


    ###############################
    # 7. Post-Processing & Saving #
    ###############################
    if results: # Check if simulation ran successfully
        simulation_results, positions_dict, mse_x_dict, mse_y_dict, mse_z_dict, mse_sdf_dict, mse_energy_dict, pred_energy_increments, gt_energy_increments = results

        # --- Save Simulation Outputs ---
        try:
            with open(OUTPUT_POSITIONS_FILE, "wb") as f: pickle.dump(positions_dict, f)
            print(f"Saved predicted particle positions to {OUTPUT_POSITIONS_FILE}")
        except Exception as e: print(f"Error saving positions: {e}")

        com_static_center = mesh_static.center_mass
        coms_dict = {}
        saved_times = sorted(positions_dict.keys())
        # Use the same CoM calculation method for saving as used in the loop
        for t in saved_times:
            current_wall_com = get_com_at_time_precalculated(t, com_splines, last_known_com_from_data)
            coms_dict[t] = current_wall_com # Store wall CoM used at this time
        try:
            with open(OUTPUT_COMS_FILE, "wb") as f: pickle.dump(coms_dict, f)
            print(f"Saved wall CoM positions to {OUTPUT_COMS_FILE}")
        except Exception as e: print(f"Error saving CoMs: {e}")

        num_pred_steps = len(pred_energy_increments)
        gt_energy_increments_padded = np.pad(gt_energy_increments, (0, num_pred_steps - len(gt_energy_increments)), 'constant', constant_values=np.nan)
        results_summary = {
            "mse_x_dict": mse_x_dict, "mse_y_dict": mse_y_dict, "mse_z_dict": mse_z_dict,
            "mse_sdf_dict": mse_sdf_dict, "mse_energy_dict": mse_energy_dict,
            "pred_energy_increments": pred_energy_increments, "gt_energy_increments": gt_energy_increments_padded
        }
        try:
            with open(OUTPUT_RESULTS_FILE, "wb") as f: pickle.dump(results_summary, f)
            print(f"Saved MSE and energy results to {OUTPUT_RESULTS_FILE}")
        except Exception as e: print(f"Error saving results: {e}")

        # --- Plotting ---
        if mse_x_dict:
            sorted_times_mse = sorted(mse_x_dict.keys())
            mse_x_values = [mse_x_dict[t] for t in sorted_times_mse]
            mse_y_values = [mse_y_dict[t] for t in sorted_times_mse]
            mse_z_values = [mse_z_dict[t] for t in sorted_times_mse]
            mse_sdf_values = [mse_sdf_dict.get(t, np.nan) for t in sorted_times_mse]
            plt.figure(figsize=(10, 6))
            plt.plot(sorted_times_mse, mse_x_values, label='MSE_x', marker='.', linestyle='-')
            plt.plot(sorted_times_mse, mse_y_values, label='MSE_y', marker='.', linestyle='-')
            plt.plot(sorted_times_mse, mse_z_values, label='MSE_z', marker='.', linestyle='-')
            plt.plot(sorted_times_mse, mse_sdf_values, label='MSE_SDF', marker='.', linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('Mean Squared Error')
            plt.title('Recursive Simulation: MSE of Particle Positions and SDF vs Time')
            plt.legend()
            plt.grid(True)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(OUTPUT_PLOT_MSE_FILE)
            print(f"Saved MSE plot to {OUTPUT_PLOT_MSE_FILE}")
        else:
            print("No position/SDF MSE data to plot.")

        if pred_energy_increments and gt_energy_increments_padded.size > 0:
            pred_increments_np = np.array(pred_energy_increments)
            gt_increments_np = gt_energy_increments_padded
            valid_gt_mask = ~np.isnan(gt_increments_np)
            cumulative_pred = np.cumsum(pred_increments_np)
            cumulative_gt = np.cumsum(gt_increments_np[valid_gt_mask])
            sim_times = np.array([snap["time"] for snap in simulation_results]) if simulation_results else np.array([])
            gt_times_for_plot = sim_times[valid_gt_mask] if len(sim_times) == len(valid_gt_mask) else sim_times[:len(cumulative_gt)]

            plt.figure(figsize=(10, 6))
            if len(sim_times) > 0:
                plt.plot(sim_times, cumulative_pred, label="SGN Predicted Cumulative Energy", marker='.', linestyle='-')
            if len(cumulative_gt) > 0:
                plt.plot(gt_times_for_plot, cumulative_gt, label="DEM Ground Truth Cumulative Energy", marker='.', linestyle='--')
            plt.xlabel("Time (s)")
            plt.ylabel("Cumulative Energy Dissipation")
            plt.title("Recursive Simulation: Cumulative Energy Dissipation Comparison (Direct Sum)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUTPUT_PLOT_ENERGY_DIRECT_FILE)
            print(f"Saved direct cumulative energy plot to {OUTPUT_PLOT_ENERGY_DIRECT_FILE}")

            if MOVING_AVERAGE_WINDOW > 0:
                n_blocks_pred = len(pred_increments_np) // MOVING_AVERAGE_WINDOW
                if n_blocks_pred > 0:
                    block_means_pred = np.array([np.mean(pred_increments_np[i*MOVING_AVERAGE_WINDOW:(i+1)*MOVING_AVERAGE_WINDOW]) for i in range(n_blocks_pred)])
                    cumulative_pred_block = np.cumsum(block_means_pred)
                    cumulative_gt_full = np.cumsum(gt_increments_np[~np.isnan(gt_increments_np)])
                    gt_indices_for_downsample = np.arange(len(gt_increments_np))[~np.isnan(gt_increments_np)]
                    downsampled_indices = []; downsampled_cumulative_gt = []
                    for i in range(n_blocks_pred):
                         block_end_step_index = (i + 1) * MOVING_AVERAGE_WINDOW - 1
                         valid_indices_in_block = gt_indices_for_downsample[gt_indices_for_downsample <= block_end_step_index]
                         if len(valid_indices_in_block) > 0:
                             closest_gt_index = valid_indices_in_block[-1]
                             cumsum_indices = np.where(gt_indices_for_downsample == closest_gt_index)[0]
                             if cumsum_indices.size > 0:
                                 cumsum_index = cumsum_indices[0]
                                 if cumsum_index < len(cumulative_gt_full):
                                     downsampled_cumulative_gt.append(cumulative_gt_full[cumsum_index])
                                     downsampled_indices.append(block_end_step_index)
                                 else:
                                     print(f"Warning: cumsum_index {cumsum_index} out of bounds for cumulative_gt_full (len {len(cumulative_gt_full)}) at block {i}")
                             else:
                                 print(f"Warning: closest_gt_index {closest_gt_index} not found in gt_indices_for_downsample at block {i}")

                    if downsampled_cumulative_gt and len(downsampled_indices) == len(cumulative_pred_block):
                        downsampled_cumulative_gt = np.array(downsampled_cumulative_gt)
                        sim_times_array = np.array([snap["time"] for snap in simulation_results]) if simulation_results else np.array([])
                        if len(sim_times_array) > 0 and max(downsampled_indices) < len(sim_times_array): # Check index validity
                            time_array_downsampled = sim_times_array[downsampled_indices]
                            plt.figure(figsize=(10, 6))
                            plt.plot(time_array_downsampled, cumulative_pred_block, label=f"SGN Predicted (Cumulative Block Avg, {MOVING_AVERAGE_WINDOW} steps)", marker='o', linestyle='-')
                            plt.plot(time_array_downsampled, downsampled_cumulative_gt, label="DEM Ground Truth (Downsampled Cumulative)", marker='s', linestyle='--')
                            plt.xlabel("Time (s)")
                            plt.ylabel("Cumulative Value")
                            plt.title("Cumulative Block Average Energy vs. Downsampled Cumulative Ground Truth")
                            plt.grid(True)
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(OUTPUT_PLOT_ENERGY_BLOCK_AVG_FILE)
                            print(f"Saved block-averaged cumulative energy plot to {OUTPUT_PLOT_ENERGY_BLOCK_AVG_FILE}")
                            # plt.show()
                        else:
                             print("Index mismatch or simulation results too short for block averaging plot.")
                    else: print("Not enough data points or length mismatch for block averaging plot.")
                else: print("Not enough data points for block averaging plot.")
        else: print("No energy increment data to plot.")

    print("Script execution complete.")
