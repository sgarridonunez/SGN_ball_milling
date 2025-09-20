#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading, preprocessing, feature engineering, and normalization utilities
for the SGN training.
"""

import pickle
import numpy as np
import torch
from torch_geometric.data import Data
import trimesh
import os


# --- Constants ---
PARTICLE_FEATURE_DIM = 7 # velocities (3), sdf_value (1), sdf_gradient (3)
WALL_FEATURE_PAD_DIM = 7 # Padded dimension for wall features per snapshot
EPSILON = 1e-6 # For safe division during normalization
DESIRED_Y_SHIFT = 0.00192903 # Determined from COM method at rest to match STL with DEM simulation. Adapt to your case.

# --- Mesh Loading and Processing ---

def load_and_transform_mesh(stl_path="ball_mill_jar.stl"):
    """Loads, scales, and transforms the static mesh."""
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL file not found at: {stl_path}")
    mesh_static = trimesh.load(stl_path)
    mesh_static.vertices = mesh_static.vertices / 1000.0  # Convert mm to meters

    orig_bounds = mesh_static.bounds
    center_x = (orig_bounds[0, 0] + orig_bounds[1, 0]) / 2.0
    center_z = (orig_bounds[0, 2] + orig_bounds[1, 2]) / 2.0
    top_y = orig_bounds[1, 1]

    def transform_coords(coords):
        coords = np.atleast_2d(np.array(coords))
        transformed = coords.copy()
        transformed[:, 0] = coords[:, 0] - center_x
        transformed[:, 2] = coords[:, 2] - center_z
        transformed[:, 1] = coords[:, 1] - top_y + DESIRED_Y_SHIFT
        return transformed

    mesh_static.vertices = transform_coords(mesh_static.vertices)
    print(f"Loaded and transformed mesh from {stl_path}. New bounds: {mesh_static.bounds}")
    return mesh_static

# --- SDF and Mesh Movement Functions ---

def SDF_static(points, target_mesh):
    """Calculates signed distance from points to a static mesh."""
    return trimesh.proximity.signed_distance(target_mesh, points)

def SDF_normal_direct(point, target_mesh):
    """Calculates the surface normal at the closest point on the mesh."""
    closest, distance, face_index = target_mesh.nearest.on_surface(point.reshape(1, 3))
    normal = target_mesh.face_normals[face_index[0]]
    norm_val = np.linalg.norm(normal)
    if norm_val < EPSILON: # Use EPSILON for consistency
        return normal
    return normal / norm_val



# --- Data Loading ---

def load_extracted_data(filepath="extracted_data_300_fine.pkl"):
    """Loads the extracted simulation data from a pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Extracted data file not found at: {filepath}")
    with open(filepath, "rb") as f:
        extracted_data = pickle.load(f)
    print(f"Loaded extracted data for {len(extracted_data)} snapshots from {filepath}.")
    return extracted_data

# --- Feature Engineering ---

def build_sgn_features_window(window_snapshots, mass, use_last_snapshot_global=False):
    """Builds node and edge features for a window of snapshots."""
    # Build particle features.
    particle_features_list = []
    for snap in window_snapshots:
        particle = snap["particle"]
        velocities = particle["velocities"]
        sdf_values = particle["sdf_values"].reshape(-1, 1)
        sdf_gradients = particle["sdf_gradients"]
        # Ensure correct shape (N, 7)
        feat = np.concatenate([velocities, sdf_values, sdf_gradients], axis=1)
        if feat.shape[1] != PARTICLE_FEATURE_DIM:
             raise ValueError(f"Expected {PARTICLE_FEATURE_DIM} particle features per snapshot, got {feat.shape[1]}")
        particle_features_list.append(feat)
    # Concatenate features across the time window -> (N, window_size * 7)
    particle_features_window = np.concatenate(particle_features_list, axis=1)

    # Build wall features (padded to WALL_FEATURE_PAD_DIM entries per snapshot).
    wall_features_list = []
    for snap in window_snapshots:
        wall_feat = np.array(snap["wall_node_features"]).reshape(1, -1)
        pad_size = WALL_FEATURE_PAD_DIM - wall_feat.shape[1]
        if pad_size < 0:
             raise ValueError(f"Wall features ({wall_feat.shape[1]}) exceed padding dim ({WALL_FEATURE_PAD_DIM})")
        if pad_size > 0:
            wall_feat = np.hstack([wall_feat, np.zeros((1, pad_size))])
        wall_features_list.append(wall_feat)
    # Concatenate features across the time window -> (1, window_size * 7)
    wall_features_window = np.concatenate(wall_features_list, axis=1)

    # Combine particle and wall features -> (N+1, window_size * 7)
    node_features = np.vstack([particle_features_window, wall_features_window])

    # --- Process last snapshot for targets and edges ---
    last_snap = window_snapshots[-1]
    particle = last_snap["particle"]
    particle_ids = np.array(particle["ids"])
    num_particles = len(particle_ids)
    id_to_index = {pid: idx for idx, pid in enumerate(particle_ids)}
    wall_node_index = num_particles # Index of the single wall node

    # Global target from the last snapshot.
    global_target = last_snap["energy_normal_increment"] + last_snap["energy_tangential_increment"]

    # Node targets (Accelerations for particles, first 3 wall features for wall)
    # Ensure wall target has shape (1, 3)
    wall_target_feat = np.array(last_snap["wall_node_features"])
    wall_target = wall_target_feat[:3].reshape(1, 3) if len(wall_target_feat) >= 3 else np.zeros((1, 3))
    node_targets = np.vstack([particle["net_forces"] / mass, wall_target])

    # Process particle-particle contacts.
    contacts_pp = last_snap["contacts_particle_particle"]
    if contacts_pp["contact_ids"].size > 0:
        converted_pairs = []
        valid_indices = [] # Keep track of which original contacts are valid
        for i, pair in enumerate(contacts_pp["contact_ids"]):
            if (pair[0] in id_to_index) and (pair[1] in id_to_index):
                converted_pairs.append([id_to_index[pair[0]], id_to_index[pair[1]]])
                valid_indices.append(i)

        if len(converted_pairs) > 0:
            edge_index_pp = torch.tensor(np.array(converted_pairs).T, dtype=torch.long)
            # Select only the distance vectors corresponding to valid pairs
            distance_vectors = np.array(contacts_pp["distance_vector"])[valid_indices]
            edge_attr_pp_input = distance_vectors
        else:
            edge_index_pp = torch.empty((2, 0), dtype=torch.long)
            edge_attr_pp_input = np.empty((0, 3), dtype=np.float32)
    else:
        edge_index_pp = torch.empty((2, 0), dtype=torch.long)
        edge_attr_pp_input = np.empty((0, 3), dtype=np.float32)

    # Process particle-wall contacts.
    contacts_pw = last_snap["contacts_particle_wall"]
    if contacts_pw["contact_ids"].size > 0:
        converted_pw_ids = []
        valid_indices_pw = [] # Indices of particles involved in valid contacts
        for i, pid in enumerate(contacts_pw["contact_ids"]):
            if pid in id_to_index:
                particle_idx = id_to_index[pid]
                converted_pw_ids.append(particle_idx)
                valid_indices_pw.append(particle_idx) # Store the particle index

        if len(converted_pw_ids) > 0:
             # Edge index: [wall_node_index, wall_node_index, ...], [particle_idx1, particle_idx2, ...]
            edge_index_pw = torch.tensor(np.vstack([np.full((len(converted_pw_ids),), wall_node_index),
                                                    converted_pw_ids]), dtype=torch.long)

            # Gather sdf_distance_vectors for valid contacts using particle indices
            sdf_distance_vectors = np.array(particle["sdf_distance_vectors"])[valid_indices_pw]
            edge_attr_pw_input = sdf_distance_vectors
        else:
            edge_index_pw = torch.empty((2, 0), dtype=torch.long)
            edge_attr_pw_input = np.empty((0, 3), dtype=np.float32)
    else:
        edge_index_pw = torch.empty((2, 0), dtype=torch.long)
        edge_attr_pw_input = np.empty((0, 3), dtype=np.float32)

    # Create the Data object.
    data = Data(x=torch.tensor(node_features, dtype=torch.float),
                y=torch.tensor(node_targets, dtype=torch.float))
    data.timestep = last_snap["timestep"]
    data.time = last_snap["time"]
    data.global_target = torch.tensor([global_target], dtype=torch.float)

    # Store separate edge features and indices before combining.
    data.edge_attr_pp = torch.tensor(edge_attr_pp_input, dtype=torch.float)
    data.edge_attr_pw = torch.tensor(edge_attr_pw_input, dtype=torch.float)
    data.edge_index_pp = edge_index_pp
    data.edge_index_pw = edge_index_pw

    # Combine edge indices and attributes for the model.
    # Ensure attributes are combined only if indices exist.
    combined_edge_index = []
    combined_edge_attr = []
    if edge_index_pp.size(1) > 0:
        combined_edge_index.append(edge_index_pp)
        combined_edge_attr.append(edge_attr_pp_input)
    if edge_index_pw.size(1) > 0:
        combined_edge_index.append(edge_index_pw)
        combined_edge_attr.append(edge_attr_pw_input)

    if combined_edge_index:
        data.edge_index = torch.cat(combined_edge_index, dim=1)
        data.edge_attr = torch.tensor(np.concatenate(combined_edge_attr, axis=0), dtype=torch.float)
    else:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        data.edge_attr = torch.empty((0, 3), dtype=torch.float) # Match edge_attr_dim

    # Store features from the last snapshot if needed for global prediction
    if use_last_snapshot_global:
        last_particle_features = particle_features_list[-1] # Shape (N, 7)
        last_wall_features = wall_features_list[-1]         # Shape (1, 7)
        last_node_features = np.vstack([last_particle_features, last_wall_features])
        data.x_last = torch.tensor(last_node_features, dtype=torch.float)

    return data


def build_sgn_dataset_window(extracted_data, window_size, mass, use_last_snapshot_global=False):
    """Builds the full dataset by applying build_sgn_features_window over the data."""
    dataset = []
    num_snapshots = len(extracted_data)
    if num_snapshots < window_size:
        print(f"Warning: Not enough snapshots ({num_snapshots}) for window size ({window_size}). Dataset will be empty.")
        return dataset

    for i in range(num_snapshots - window_size + 1):
        window_snapshots = extracted_data[i : i + window_size]
        try:
            sample = build_sgn_features_window(window_snapshots, mass, use_last_snapshot_global)
            dataset.append(sample)
        except ValueError as e:
            print(f"Skipping window starting at index {i} due to error: {e}")
            continue # Skip this window if feature dimensions mismatch

    print(f"Built SGN dataset (window={window_size}) for {len(dataset)} samples.")
    return dataset


# --- Normalization ---

def calculate_normalization_params(train_dataset):
    """Calculates mean and std for features and targets from the training dataset."""
    if not train_dataset:
        raise ValueError("Training dataset is empty, cannot calculate normalization parameters.")

    # Node features (particles and wall separately)
    particle_features_list = [data.x[:-1] for data in train_dataset if data.x.size(0) > 1]
    wall_features_list = [data.x[-1].unsqueeze(0) for data in train_dataset if data.x.size(0) > 0]

    norm_particle_mean = torch.zeros_like(particle_features_list[0][0])
    norm_particle_std = torch.ones_like(particle_features_list[0][0])
    if particle_features_list:
        all_particle_features = torch.cat(particle_features_list, dim=0)
        norm_particle_mean = all_particle_features.mean(dim=0)
        norm_particle_std = all_particle_features.std(dim=0)

    norm_wall_mean = torch.zeros_like(wall_features_list[0][0])
    norm_wall_std = torch.ones_like(wall_features_list[0][0])
    if wall_features_list:
        all_wall_features = torch.cat(wall_features_list, dim=0)
        norm_wall_mean = all_wall_features.mean(dim=0)
        norm_wall_std = all_wall_features.std(dim=0)

    # Edge features (pp and pw separately)
    edge_pp_list = [data.edge_attr_pp for data in train_dataset if hasattr(data, 'edge_attr_pp') and data.edge_attr_pp.size(0) > 0]
    edge_pw_list = [data.edge_attr_pw for data in train_dataset if hasattr(data, 'edge_attr_pw') and data.edge_attr_pw.size(0) > 0]

    norm_edge_pp_mean, norm_edge_pp_std = None, None
    if edge_pp_list:
        all_edge_pp = torch.cat(edge_pp_list, dim=0)
        if all_edge_pp.numel() > 0: # Check if tensor is not empty
             norm_edge_pp_mean = all_edge_pp.mean(dim=0)
             norm_edge_pp_std = all_edge_pp.std(dim=0)

    norm_edge_pw_mean, norm_edge_pw_std = None, None
    if edge_pw_list:
        all_edge_pw = torch.cat(edge_pw_list, dim=0)
        if all_edge_pw.numel() > 0: # Check if tensor is not empty
            norm_edge_pw_mean = all_edge_pw.mean(dim=0)
            norm_edge_pw_std = all_edge_pw.std(dim=0)

    # Node targets (particles and wall separately)
    # Assuming target y has shape (N+1, 3)
    target_particle_list = [data.y[:-1] for data in train_dataset if data.y.size(0) > 1]
    target_wall_list = [data.y[-1].unsqueeze(0) for data in train_dataset if data.y.size(0) > 0]

    norm_target_particle_mean = torch.zeros(3)
    norm_target_particle_std = torch.ones(3)
    if target_particle_list:
        all_target_particles = torch.cat(target_particle_list, dim=0)
        norm_target_particle_mean = all_target_particles.mean(dim=0)
        norm_target_particle_std = all_target_particles.std(dim=0)

    norm_target_wall_mean = torch.zeros(3)
    norm_target_wall_std = torch.ones(3)
    if target_wall_list:
        all_target_walls = torch.cat(target_wall_list, dim=0)
        norm_target_wall_mean = all_target_walls.mean(dim=0)
        norm_target_wall_std = all_target_walls.std(dim=0)

    # Global targets
    global_target_list = [data.global_target for data in train_dataset if hasattr(data, 'global_target')]
    norm_global_mean = torch.tensor([0.0])
    norm_global_std = torch.tensor([1.0])
    if global_target_list:
        all_global = torch.cat(global_target_list, dim=0)
        norm_global_mean = all_global.mean(dim=0)
        norm_global_std = all_global.std(dim=0)

    # Consolidate target normalization (optional, could keep separate)
    # For simplicity, using combined mean/std as before, but calculated separately
    all_targets = torch.cat([data.y for data in train_dataset], dim=0)
    norm_target_mean = all_targets.mean(dim=0)
    norm_target_std = all_targets.std(dim=0)


    normalization_params = {
        'norm_particle_mean': norm_particle_mean.cpu().numpy(),
        'norm_particle_std': norm_particle_std.cpu().numpy(),
        'norm_wall_mean': norm_wall_mean.cpu().numpy(),
        'norm_wall_std': norm_wall_std.cpu().numpy(),
        'norm_edge_pp_mean': norm_edge_pp_mean.cpu().numpy() if norm_edge_pp_mean is not None else None,
        'norm_edge_pp_std': norm_edge_pp_std.cpu().numpy() if norm_edge_pp_std is not None else None,
        'norm_edge_pw_mean': norm_edge_pw_mean.cpu().numpy() if norm_edge_pw_mean is not None else None,
        'norm_edge_pw_std': norm_edge_pw_std.cpu().numpy() if norm_edge_pw_std is not None else None,
        'norm_target_mean': norm_target_mean.cpu().numpy(), # Using combined target norm
        'norm_target_std': norm_target_std.cpu().numpy(),   # Using combined target norm
        # 'norm_target_particle_mean': norm_target_particle_mean.cpu().numpy(), # Alternative: separate target norms
        # 'norm_target_particle_std': norm_target_particle_std.cpu().numpy(),
        # 'norm_target_wall_mean': norm_target_wall_mean.cpu().numpy(),
        # 'norm_target_wall_std': norm_target_wall_std.cpu().numpy(),
        'norm_global_mean': norm_global_mean.cpu().numpy(),
        'norm_global_std': norm_global_std.cpu().numpy()
    }
    return normalization_params

def save_normalization_params(params, filepath="normalization_params.pkl"):
    """Saves normalization parameters to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(params, f)
    print(f"Saved normalization parameters to {filepath}.")

def load_normalization_params(filepath="normalization_params.pkl"):
    """Loads normalization parameters from a pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Normalization parameters file not found at: {filepath}")
    with open(filepath, "rb") as f:
        params = pickle.load(f)
    print(f"Loaded normalization parameters from {filepath}.")
    # Convert numpy arrays back to tensors
    tensor_params = {}
    for key, value in params.items():
        if value is not None:
            tensor_params[key] = torch.from_numpy(value)
        else:
            tensor_params[key] = None
    return tensor_params


def normalize_dataset_separate(dataset, params, add_noise=False, noise_std_percentage=0.05):
    """Normalizes dataset features and targets using pre-calculated parameters."""
    norm_particle_mean = params['norm_particle_mean']
    norm_particle_std = params['norm_particle_std']
    norm_wall_mean = params['norm_wall_mean']
    norm_wall_std = params['norm_wall_std']
    norm_edge_pp_mean = params.get('norm_edge_pp_mean') # Use .get for optional params
    norm_edge_pp_std = params.get('norm_edge_pp_std')
    norm_edge_pw_mean = params.get('norm_edge_pw_mean')
    norm_edge_pw_std = params.get('norm_edge_pw_std')
    norm_target_mean = params['norm_target_mean']
    norm_target_std = params['norm_target_std']
    norm_global_mean = params['norm_global_mean']
    norm_global_std = params['norm_global_std']

    # Ensure std deviations are not zero or too small
    norm_particle_std = torch.clamp(norm_particle_std, min=EPSILON)
    norm_wall_std = torch.clamp(norm_wall_std, min=EPSILON)
    if norm_edge_pp_std is not None:
        norm_edge_pp_std = torch.clamp(norm_edge_pp_std, min=EPSILON)
    if norm_edge_pw_std is not None:
        norm_edge_pw_std = torch.clamp(norm_edge_pw_std, min=EPSILON)
    norm_target_std = torch.clamp(norm_target_std, min=EPSILON)
    norm_global_std = torch.clamp(norm_global_std, min=EPSILON)


    for data in dataset:
        # Move params to data's device if necessary (though usually done in training loop)
        device = data.x.device
        norm_particle_mean_dev = norm_particle_mean.to(device)
        norm_particle_std_dev = norm_particle_std.to(device)
        norm_wall_mean_dev = norm_wall_mean.to(device)
        norm_wall_std_dev = norm_wall_std.to(device)
        norm_target_mean_dev = norm_target_mean.to(device)
        norm_target_std_dev = norm_target_std.to(device)
        norm_global_mean_dev = norm_global_mean.to(device)
        norm_global_std_dev = norm_global_std.to(device)

        # Normalize node features
        if data.x.size(0) > 1: # Check if particles exist
             data.x[:-1] = (data.x[:-1] - norm_particle_mean_dev) / norm_particle_std_dev
        if data.x.size(0) > 0: # Check if wall exists
             data.x[-1] = (data.x[-1] - norm_wall_mean_dev) / norm_wall_std_dev

        # Normalize edge features (pp)
        if hasattr(data, 'edge_attr_pp') and data.edge_attr_pp.size(0) > 0 and norm_edge_pp_mean is not None:
            norm_edge_pp_mean_dev = norm_edge_pp_mean.to(device)
            norm_edge_pp_std_dev = norm_edge_pp_std.to(device)
            data.edge_attr_pp = (data.edge_attr_pp - norm_edge_pp_mean_dev) / norm_edge_pp_std_dev

        # Normalize edge features (pw)
        if hasattr(data, 'edge_attr_pw') and data.edge_attr_pw.size(0) > 0 and norm_edge_pw_mean is not None:
            norm_edge_pw_mean_dev = norm_edge_pw_mean.to(device)
            norm_edge_pw_std_dev = norm_edge_pw_std.to(device)
            data.edge_attr_pw = (data.edge_attr_pw - norm_edge_pw_mean_dev) / norm_edge_pw_std_dev

        # Recombine normalized edge features into data.edge_attr
        combined_edge_attr = []
        if hasattr(data, 'edge_attr_pp') and data.edge_attr_pp.size(0) > 0:
             combined_edge_attr.append(data.edge_attr_pp)
        if hasattr(data, 'edge_attr_pw') and data.edge_attr_pw.size(0) > 0:
             combined_edge_attr.append(data.edge_attr_pw)

        if combined_edge_attr:
             data.edge_attr = torch.cat(combined_edge_attr, dim=0)
        elif hasattr(data, 'edge_index') and data.edge_index.size(1) == 0: # Handle case with no edges
             # Ensure edge_attr has correct shape (0, edge_dim) if edge_index is (2,0)
             edge_dim = 3 # Assuming edge dim is 3
             data.edge_attr = torch.empty((0, edge_dim), dtype=torch.float, device=device)


        # Normalize targets
        data.y = (data.y - norm_target_mean_dev) / norm_target_std_dev
        if hasattr(data, 'global_target'):
            data.global_target = (data.global_target - norm_global_mean_dev) / norm_global_std_dev

        # Optional: Add noise to targets (usually done dynamically in training loop)
        if add_noise:
            # Calculate noise std based on original target std
            noise_std_val = params['norm_target_std'].mean() * noise_std_percentage # Use original std
            noise = torch.randn_like(data.y) * noise_std_val.to(device)
            # Add noise in the original scale, then re-normalize (or add in normalized scale)
            # Adding in normalized scale:
            data.y = data.y + noise / norm_target_std_dev # Scale noise by target std

    return dataset
