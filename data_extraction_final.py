#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:03:07 2025

@author: Santiago
"""

import numpy as np
import trimesh
import glob
import sys
from edempy import Deck, BoxBin
import pickle
import matplotlib.pyplot as plt



###############################################
# --- 1. Load STL, Scale, and Transform Mesh --- #
###############################################

mesh_static = trimesh.load("ball_mill_jar.stl")  # Replace with your STL file path.
mesh_static.vertices = mesh_static.vertices / 1000.0  # Convert mm to meters. This to match EDEM simulation. Adapt to your case.

orig_bounds = mesh_static.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
center_x = (orig_bounds[0, 0] + orig_bounds[1, 0]) / 2.0
center_z = (orig_bounds[0, 2] + orig_bounds[1, 2]) / 2.0
top_y = orig_bounds[1, 1]

def transform_coords(coords):
    desired_y_shift = 0.00192903  # transformations to match STL with DEM simulation. Adapt to your case.
    coords = np.atleast_2d(np.array(coords))
    transformed = coords.copy()
    transformed[:, 0] = coords[:, 0] - center_x
    transformed[:, 2] = coords[:, 2] - center_z
    transformed[:, 1] = coords[:, 1] - top_y + desired_y_shift
    return transformed

mesh_static.vertices = transform_coords(mesh_static.vertices)
new_bounds = mesh_static.bounds

###############################################
# --- 2. Define SDF and SDF Normal Functions --- #
###############################################

def SDF_static(points, target_mesh):
    return trimesh.proximity.signed_distance(target_mesh, points)

def SDF_normal_direct(point, target_mesh):
    closest, distance, face_index = target_mesh.nearest.on_surface(point.reshape(1, 3))
    normal = target_mesh.face_normals[face_index[0]]
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-8:
        return normal
    return normal / norm_val

def SDF_gradient_direct(point, target_mesh, epsilon=1e-5):
    """
    Computes the gradient of the SDF at a given point using central finite differences.
    
    Args:
        point (array-like): A point in space (3D coordinates).
        target_mesh: A trimesh object.
        epsilon (float): A small perturbation for finite difference computation.
        
    Returns:
        grad (np.array): The gradient vector of the SDF at the given point.
    """
    point = np.array(point).reshape(1, 3)
    grad = np.zeros(3)
    for i in range(3):
    # Create perturbation along the i-th axis.
        d = np.zeros(3)
        d[i] = epsilon
        # Compute SDF at slightly displaced points.
        sdf_plus = trimesh.proximity.signed_distance(target_mesh, (point + d).reshape(1, 3))[0]
        sdf_minus = trimesh.proximity.signed_distance(target_mesh, (point - d).reshape(1, 3))[0]
        grad[i] = (sdf_plus - sdf_minus) / (2 * epsilon)
    return grad



###############################################
# --- 3. Define COM Method for Moving the Mesh --- #
###############################################

def get_moved_mesh_using_CoM(timestep_index, deck):
    com_edem = np.array(deck.timestep[timestep_index].geometry['First try jar'].getCoM())
    com_static = mesh_static.center_mass
    offset = com_edem - com_static
    moved_mesh = mesh_static.copy()
    moved_mesh.vertices = moved_mesh.vertices + offset
    return moved_mesh

def SDF_global_moved(x, timestep_index, deck):
    moved_mesh = get_moved_mesh_using_CoM(timestep_index, deck)
    return SDF_static(x, moved_mesh)

###############################################
# --- 4. Load EDEM Data and Define Snapshot Range --- #
###############################################

decks = []
for fileName in glob.glob("*.dem"):
    decks.append(fileName)
if len(decks) == 0:
    print("No deck file found in the directory")
    sys.exit()
deck_file = decks[0]
deck = Deck(deck_file)
print("Using deck file:", deck_file)

start_timestep = 50000   # starting snapshot index
last_timestep  = 100000   # adjust as needed for your extraction range
total_timesteps = deck.numTimesteps
print("Total timesteps in simulation:", total_timesteps)

custom_properties_dict = deck.creatorData.simulationPropertyData
custom_properties_name = np.array(deck.creatorData.simulationCustomPropertyNames)
custom_properties_num = deck.creatorData.numSimulationCustomProperties

###############################################
# --- 5. Extract Data from Simulation Snapshots --- #
###############################################

def extract_data(deck, start_timestep, last_timestep):
    extracted_data = []
    prev_joint_normal = None
    prev_joint_tangential = None
    for t in range(start_timestep, last_timestep + 1):
        snapshot = {}
        snapshot["time"] = deck.timestepValues[t]
        snapshot["timestep"] = t  # store snapshot index
        
        # --- Particle Data ---
        particle = {}
        particle["positions"] = np.array(deck.timestep[t].particle[0].getPositions())
        particle["ids"] = deck.timestep[t].particle[0].getIds()
        particle["velocities"] = np.array(deck.timestep[t].particle[0].getVelocities())
        particle["angular_velocities"] = np.array(deck.timestep[t].particle[0].getAngularVelocities())
        particle["net_forces"] = np.array(deck.timestep[t].particle[0].getForce())
        particle["torques"] = np.array(deck.timestep[t].particle[0].getTorque())
        particle["mass"] = np.array(deck.timestep[t].particle[0].getMass())
        particle["radius"] = np.array(deck.timestep[t].particle[0].getSphereRadii())
        particle["inertia"] = np.array((2/5) * particle["mass"] * (particle["radius"]**2))
        particle["num_particles"] = deck.timestep[t].particle[0].getNumParticles()
        particle["sdf_values"] = SDF_global_moved(particle["positions"], t, deck)
        moved_mesh = get_moved_mesh_using_CoM(t, deck)
        particle["sdf_normals"] = np.array([SDF_normal_direct(pos, moved_mesh) for pos in particle["positions"]])
        particle["sdf_gradients"] = np.array([SDF_gradient_direct(pos, moved_mesh) for pos in particle["positions"]])
        # Ensure sdf_values is a column vector (shape: [N, 1])
        sdf_values = particle["sdf_values"].reshape(-1, 1)  

        # Get the gradients (shape: [N, 3])
        sdf_gradients = particle["sdf_gradients"]

        # Compute the norm for each gradient vector.
        norms = np.linalg.norm(sdf_gradients, axis=1, keepdims=True)

        # Avoid division by zero: if norm is too small, use the original gradient.
        normalized_gradients = np.where(norms < 1e-8, sdf_gradients, sdf_gradients / norms)

        # Compute the distance vectors.
        particle["sdf_distance_vectors"] = sdf_values * normalized_gradients
        
    
        
        
        snapshot["particle"] = particle
        
        # --- Particle-Particle Contacts ---
        contacts_pp = {}
        try:
            contacts_pp["contact_ids"] = np.array(deck.timestep[t].contact.surfSurf.getIds())
            contacts_pp["normal_forces"] = np.array(deck.timestep[t].contact.surfSurf.getNormalForce())
            contacts_pp["tangential_forces"] = np.array(deck.timestep[t].contact.surfSurf.getTangentialForce())
            contact_vectors = np.array(deck.timestep[t].contact.surfSurf.getContactVector1())  # Shape (N, 3)
            contacts_pp["relative_distances"] = np.linalg.norm(contact_vectors, axis=1)  # Shape (N,)
            contacts_pp["distance_vector"] = np.array(deck.timestep[t].contact.surfSurf.getContactVector1()) * 2
        except KeyError:
            contacts_pp["contact_ids"] = np.empty((0, 2))
            contacts_pp["normal_forces"] = np.empty((0, 3))
            contacts_pp["tangential_forces"] = np.empty((0, 3))
            contacts_pp["relative_distances"] = np.empty((0,))
            contacts_pp["distance_vector"] = np.empty((0,3))
        snapshot["contacts_particle_particle"] = contacts_pp
        
        # --- Particle-Wall Contacts ---
        contacts_pw = {}
        try:
            contacts_pw["normal_forces"] = np.array(deck.timestep[t].contact.surfGeom.getNormalForce())
            contacts_pw["tangential_forces"] = np.array(deck.timestep[t].contact.surfGeom.getTangentialForce())
            contacts_pw["contact_ids"] = np.array(deck.timestep[t].contact.surfGeom.getIds()[:, 0])
            
        except KeyError:
            contacts_pw["normal_forces"] = np.empty((0, 3))
            contacts_pw["tangential_forces"] = np.empty((0, 3))
            contacts_pw["contact_ids"] = np.empty((0,))
            
        snapshot["contacts_particle_wall"] = contacts_pw
        
        # --- Wall Data ---
        moved_bounds = moved_mesh.bounds
        wall_center = np.array([
            (moved_bounds[0, 0] + moved_bounds[1, 0]) / 2.0,
            (moved_bounds[0, 1] + moved_bounds[1, 1]) / 2.0,
            (moved_bounds[0, 2] + moved_bounds[1, 2]) / 2.0
        ])
        wall_rotational_speed = 300.0  # placeholder value
        wall_com = np.array(deck.timestep[t].geometry['First try jar'].getCoM())
        wall_node_features = np.concatenate([wall_com, np.array([wall_rotational_speed])])
        snapshot["wall_node_features"] = wall_node_features
        
        # --- Energy Dissipation ---
        # For normal energy loss:
        try:
            pw_n_loss = deck.timestep[t].customProperties[np.where(custom_properties_name=='Particle-Wall Normal Energy Loss')[0][0]].getData()[0]
            pp_n_loss = deck.timestep[t].customProperties[np.where(custom_properties_name=='Particle-Particle Normal Energy Loss')[0][0]].getData()[0]
            joint_normal = pw_n_loss + pp_n_loss
        except:
            joint_normal = 0.0
        
        # For tangential energy loss:
        try:
            pw_t_loss = deck.timestep[t].customProperties[np.where(custom_properties_name=='Particle-Wall Tangential Energy Loss')[0][0]].getData()[0]
            pp_t_loss = deck.timestep[t].customProperties[np.where(custom_properties_name=='Particle-Particle Tangential Energy Loss')[0][0]].getData()[0]
            joint_tangential = pw_t_loss + pp_t_loss
        except:
            joint_tangential = 0.0
        
        # Compute incremental loss (difference from previous snapshot)
        if prev_joint_normal is None:
            incremental_normal = 0.0
            incremental_tangential = 0.0
        else:
            incremental_normal = joint_normal - prev_joint_normal
            incremental_tangential = joint_tangential - prev_joint_tangential
        
        prev_joint_normal = joint_normal
        prev_joint_tangential = joint_tangential
        
        snapshot["energy_normal_increment"] = np.array([incremental_normal])
        snapshot["energy_tangential_increment"] = np.array([incremental_tangential])
        
        extracted_data.append(snapshot)
    return extracted_data

extracted_data = extract_data(deck, start_timestep, last_timestep)
print("Extracted data for", len(extracted_data), "snapshots.")

# Save the extracted data to a file.
with open("extracted_data_300_fine.pkl", "wb") as f:
    pickle.dump(extracted_data, f)
print("Saved extracted data to 'extracted_data_300_fine.pkl'.")






# Easy knobs to customize
BASE_FONTSIZE = 21   # axis labels; ticks/legend use ±1 around this
LINEWIDTH     = 3  # default line thickness for all lines
MARKERSIZE    = 7  # default marker size

plt.rcParams.update({
    # Fonts & text
    "font.size": BASE_FONTSIZE,
    "axes.labelsize": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE + 1,
    "xtick.labelsize": BASE_FONTSIZE - 1,
    "ytick.labelsize": BASE_FONTSIZE - 1,
    "legend.fontsize": BASE_FONTSIZE - 1,

    # Lines & markers
    "lines.linewidth": LINEWIDTH,
    "lines.markersize": MARKERSIZE,

    # Spines, ticks, grid
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.7,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    # Save/export defaults
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 120,   # interactive display
    "savefig.dpi": 600,  # file output (PNG)
    "pdf.fonttype": 42,  # embed TrueType fonts in PDF (no Type 3)
    "ps.fonttype": 42,
})



# Load the extracted data containing snapshots
with open("extracted_data_300_fine.pkl", "rb") as f:
    extracted_data = pickle.load(f)

# Container for all SDF values at particle-wall contacts
contact_sdf_values = []

# Loop through each snapshot
for snapshot in extracted_data:
    # Get the SDF values computed for all particles in the snapshot.
    # We flatten the array to ensure it's 1D.
    particle_sdf = np.array(snapshot["particle"]["sdf_values"]).flatten()
    
    # Retrieve the contact indices from the particle-wall contact info.
    contacts_pw = snapshot.get("contacts_particle_wall", {})
    contact_ids = contacts_pw.get("contact_ids", np.empty((0,)))
    
    # Only proceed if we have any contact indices.
    if contact_ids.size > 0:
        contact_ids = contact_ids.astype(int)
        # If the maximum index is out-of-bound (e.g., 24 for an array of size 24),
        # subtract 1 from all indices to convert from 1-indexed to 0-indexed.
        if contact_ids.max() >= particle_sdf.shape[0]:
            contact_ids = contact_ids - 1
        
        # Get the SDF values corresponding to these contact indices.
        contact_sdfs = particle_sdf[contact_ids]
        contact_sdf_values.extend(contact_sdfs.tolist())

# Convert the collected list to a numpy array for statistical analysis.
contact_sdf_values = np.array(contact_sdf_values)

# Compute and display statistics if any contact SDF values were found.
if contact_sdf_values.size > 0:
    mean_val = np.mean(contact_sdf_values)
    median_val = np.median(contact_sdf_values)
    std_val = np.std(contact_sdf_values)
    min_val = np.min(contact_sdf_values)
    max_val = np.max(contact_sdf_values)
    p5 = np.percentile(contact_sdf_values, 5)
    p95 = np.percentile(contact_sdf_values, 95)

    print("Statistics of SDF values for particle-wall contacts:")
    print(f"Mean: {mean_val:.6f} m")
    print(f"Median: {median_val:.6f} m")
    print(f"Standard Deviation: {std_val:.6f} m")
    print(f"Min: {min_val:.6f} m")
    print(f"Max: {max_val:.6f} m")
    print(f"5th Percentile: {p5:.6f} m")
    print(f"95th Percentile: {p95:.6f} m")

    # Plot a histogram of the contact SDF values
    plt.figure(figsize=(8, 6))
    plt.hist(contact_sdf_values, bins=50, edgecolor='black')
    plt.xlabel("SDF value (m)")
    plt.ylabel("Frequency")
    #plt.title("Histogram of SDF values for particle-wall contacts")
    # Mark the current threshold (-0.005 m) for reference.
    plt.axvline(-0.0052, color='red', linestyle='--', label="Threshold (-0.0052 m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No particle-wall contact SDF values found in the data.")
    
    
# Load the extracted data (assumed saved in "extracted_data.pkl")
with open("extracted_data_300_fine.pkl", "rb") as f:
    extracted_data = pickle.load(f)

# Container for all relative distances from particle–particle contacts
relative_distances_list = []

# Loop through each snapshot
for snapshot in extracted_data:
    # Retrieve the particle-particle contacts dictionary
    contacts_pp = snapshot.get("contacts_particle_particle", {})
    # Get the relative distances array (computed as the Euclidean norm of contact vectors)
    rel_distances = contacts_pp.get("relative_distances", np.empty((0,)))
    
    if rel_distances.size > 0:
        # Flatten in case it's not 1D
        rel_distances = np.array(rel_distances).flatten()
        relative_distances_list.extend(rel_distances.tolist())

# Convert the collected relative distances into a numpy array
relative_distances = np.array(relative_distances_list)

# Compute and display statistics if we found any data
if relative_distances.size > 0:
    mean_val = np.mean(relative_distances)
    median_val = np.median(relative_distances)
    std_val = np.std(relative_distances)
    min_val = np.min(relative_distances)
    max_val = np.max(relative_distances)
    p5 = np.percentile(relative_distances, 5)
    p95 = np.percentile(relative_distances, 95)

    print("Statistics of relative distances for particle-particle contacts:")
    print(f"Mean: {mean_val:.6f} m")
    print(f"Median: {median_val:.6f} m")
    print(f"Standard Deviation: {std_val:.6f} m")
    print(f"Min: {min_val:.6f} m")
    print(f"Max: {max_val:.6f} m")
    print(f"5th Percentile: {p5:.6f} m")
    print(f"95th Percentile: {p95:.6f} m")

    # Plot a histogram for visual inspection
    plt.figure(figsize=(6, 4))
    plt.hist(relative_distances, bins=50, edgecolor='black')
    plt.xlabel("Relative Distance (m)")
    plt.ylabel("Frequency")
    plt.title("Histogram of relative distances for particle-particle contacts")
    # Mark the threshold of 2*r (with r = 0.005 m, so 2*r = 0.01 m)
    plt.axvline(0.004998, color='red', linestyle='--', label="Threshold (2*r = 0.01 m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No particle-particle contact relative distance data found.")
    
    
    
    # Container for all magnitudes of SDF distance vectors at particle-wall contacts
contact_sdf_distance_magnitudes = []

# Loop through each snapshot
for snapshot in extracted_data:
    # Retrieve the particle-wall contact information
    contacts_pw = snapshot.get("contacts_particle_wall", {})
    # Access the SDF distance vectors (assumed to be a list or array of 3D vectors)
    sdf_distance_vectors = contacts_pw.get("sdf_distance_vectors", None)
    
    if sdf_distance_vectors is not None:
        # Convert to a numpy array if not already
        sdf_distance_vectors = np.array(sdf_distance_vectors)
        
        # Compute the magnitude (Euclidean norm) for each vector.
        # Assumes each row is a vector with 3 components.
        magnitudes = np.linalg.norm(sdf_distance_vectors, axis=1)
        
        # Add these magnitudes to our container list.
        contact_sdf_distance_magnitudes.extend(magnitudes.tolist())

# Convert the collected list to a numpy array for statistical analysis.
contact_sdf_distance_magnitudes = np.array(contact_sdf_distance_magnitudes)

# Compute and display statistics if any contact SDF distance magnitudes were found.
if contact_sdf_distance_magnitudes.size > 0:
    mean_val = np.mean(contact_sdf_distance_magnitudes)
    median_val = np.median(contact_sdf_distance_magnitudes)
    std_val = np.std(contact_sdf_distance_magnitudes)
    min_val = np.min(contact_sdf_distance_magnitudes)
    max_val = np.max(contact_sdf_distance_magnitudes)
    p5 = np.percentile(contact_sdf_distance_magnitudes, 5)
    p95 = np.percentile(contact_sdf_distance_magnitudes, 95)

    print("Statistics of SDF distance vector magnitudes for particle-wall contacts:")
    print(f"Mean: {mean_val:.6f} m")
    print(f"Median: {median_val:.6f} m")
    print(f"Standard Deviation: {std_val:.6f} m")
    print(f"Min: {min_val:.6f} m")
    print(f"Max: {max_val:.6f} m")
    print(f"5th Percentile: {p5:.6f} m")
    print(f"95th Percentile: {p95:.6f} m")

    # Plot a histogram of the magnitudes
    plt.figure(figsize=(6, 4))
    plt.hist(contact_sdf_distance_magnitudes, bins=50, edgecolor='black')
    plt.xlabel("Magnitude of SDF distance vector (m)")
    plt.ylabel("Frequency")
    #plt.title("Histogram of SDF distance vector magnitudes for particle-wall contacts")
    # Mark a threshold line if needed (example threshold at 0.0052 m)
    plt.axvline(0.0052, color='red', linestyle='--', label="Threshold (0.0052 m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No particle-wall contact SDF distance vector values found in the data.")
    
    
    # Container for all magnitudes of SDF distance vectors at particle-wall contacts
contact_sdf_distance_magnitudes = []

# Loop through each snapshot
for snapshot in extracted_data:
    # Retrieve the particle-wall contact information.
    contacts_pw = snapshot.get("contacts_particle_wall", {})
    
    # Retrieve the contact indices from the particle-wall contact info.
    contact_ids = contacts_pw.get("contact_ids", np.empty((0,)))
    
    # Only proceed if we have any contact indices.
    if contact_ids.size > 0:
        contact_ids = contact_ids.astype(int)
        
        # Retrieve the SDF distance vectors (assumed stored under the key "sdf_distance_vectors")
        sdf_distance_vectors = np.array(contacts_pw.get("sdf_distance_vectors", []))
        
        # If the maximum index is out-of-bound (e.g., 24 for an array of size 24),
        # subtract 1 from all indices to convert from 1-indexed to 0-indexed.
        if sdf_distance_vectors.shape[0] > 0 and contact_ids.max() >= sdf_distance_vectors.shape[0]:
            contact_ids = contact_ids - 1
        
        # Get the SDF distance vectors corresponding to these contact indices.
        contact_vectors = sdf_distance_vectors[contact_ids]
        
        # Compute the magnitude (Euclidean norm) for each distance vector.
        magnitudes = np.linalg.norm(contact_vectors, axis=1)
        contact_sdf_distance_magnitudes.extend(magnitudes.tolist())

# Convert the collected list to a numpy array for statistical analysis.
contact_sdf_distance_magnitudes = np.array(contact_sdf_distance_magnitudes)

# Compute and display statistics if any contact SDF distance magnitudes were found.
if contact_sdf_distance_magnitudes.size > 0:
    mean_val = np.mean(contact_sdf_distance_magnitudes)
    median_val = np.median(contact_sdf_distance_magnitudes)
    std_val = np.std(contact_sdf_distance_magnitudes)
    min_val = np.min(contact_sdf_distance_magnitudes)
    max_val = np.max(contact_sdf_distance_magnitudes)
    p5 = np.percentile(contact_sdf_distance_magnitudes, 5)
    p95 = np.percentile(contact_sdf_distance_magnitudes, 95)

    print("Statistics of SDF distance vector magnitudes for particle-wall contacts:")
    print(f"Mean: {mean_val:.6f} m")
    print(f"Median: {median_val:.6f} m")
    print(f"Standard Deviation: {std_val:.6f} m")
    print(f"Min: {min_val:.6f} m")
    print(f"Max: {max_val:.6f} m")
    print(f"5th Percentile: {p5:.6f} m")
    print(f"95th Percentile: {p95:.6f} m")

    # Plot a histogram of the contact SDF distance vector magnitudes
    plt.figure(figsize=(6, 4))
    plt.hist(contact_sdf_distance_magnitudes, bins=50, edgecolor='black')
    plt.xlabel("Magnitude of SDF distance vector (m)")
    plt.ylabel("Frequency")
    plt.title("Histogram of SDF distance vector magnitudes for particle-wall contacts")
    # Mark a reference threshold line if needed (example threshold at 0.0052 m)
    plt.axvline(0.0052, color='red', linestyle='--', label="Threshold (0.0052 m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No particle-wall contact SDF distance vector values found in the data.")


# Container for all magnitudes of SDF distance vectors at particle-wall contacts across snapshots
all_contact_magnitudes = []

# Loop through each snapshot in the extracted data
for snapshot in extracted_data:
    # Get particle IDs and corresponding SDF distance vectors
    particle_ids = np.array(snapshot["particle"]["ids"])
    sdf_distance_vectors = np.array(snapshot["particle"]["sdf_distance_vectors"])
    
    # Get the contact info for particle-wall contacts
    contacts_pw = snapshot.get("contacts_particle_wall", {})
    contact_ids = contacts_pw.get("contact_ids", np.empty((0,)))
    
    # Only proceed if there are any contact indices.
    if contact_ids.size > 0:
        for cid in contact_ids:
            # Ensure the contact id is an integer.
            cid = int(cid)
            # Find the index in particle_ids where the id equals cid.
            idx_array = np.where(particle_ids == cid)[0]
            if idx_array.size > 0:
                idx = idx_array[0]
                # Retrieve the SDF distance vector corresponding to this particle.
                vec = sdf_distance_vectors[idx]
                # Compute its magnitude.
                mag = np.linalg.norm(vec)
                all_contact_magnitudes.append(mag)

# Convert the collected magnitudes to a NumPy array.
all_contact_magnitudes = np.array(all_contact_magnitudes)

# Compute statistics and plot the distribution if any contact magnitudes were found.
if all_contact_magnitudes.size > 0:
    mean_val = np.mean(all_contact_magnitudes)
    median_val = np.median(all_contact_magnitudes)
    std_val = np.std(all_contact_magnitudes)
    min_val = np.min(all_contact_magnitudes)
    max_val = np.max(all_contact_magnitudes)
    p5 = np.percentile(all_contact_magnitudes, 5)
    p90 = np.percentile(all_contact_magnitudes, 90)

    print("Statistics of SDF distance vector magnitudes for particle-wall contacts (all snapshots):")
    print(f"Mean: {mean_val:.6f} m")
    print(f"Median: {median_val:.6f} m")
    print(f"Standard Deviation: {std_val:.6f} m")
    print(f"Min: {min_val:.6f} m")
    print(f"Max: {max_val:.6f} m")
    print(f"5th Percentile: {p5:.6f} m")
    print(f"90th Percentile: {p90:.6f} m")

    # Plot a histogram of the magnitudes.
    plt.figure(figsize=(6, 4))
    plt.hist(all_contact_magnitudes, bins=50, edgecolor='black')
    plt.xlabel("Magnitude of SDF distance vector (m)")
    plt.ylabel("Frequency")
    plt.title("Histogram of SDF distance vector magnitudes\nfor particle-wall contacts (all snapshots)")
    # Optionally, add a reference threshold line.
    plt.axvline(0.0052, color='red', linestyle='--', label="Threshold (0.0052 m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No particle-wall contact SDF distance vector values found in the data.")
    
    
    
# Lists to collect durations for each collision type
pp_durations = []  # particle-particle durations
pw_durations = []  # particle-wall durations

# Loop over timesteps 5005 to 15000 (adjust indices as needed)
for i in range(5005, 15000):
    timestep = deck.timestep[i]
    
    # Particle-Particle collisions (surfSurf)
    try:
        # Try to extract start and end times for particle-particle collisions
        start_times_pp = timestep.collision.surfSurf.getStartTime()  # returns an array
        end_times_pp   = timestep.collision.surfSurf.getEndTimes()    # returns an array
        durations_pp = end_times_pp - start_times_pp
        pp_durations.extend(durations_pp.tolist())
    except Exception as e:
        # If extraction fails (e.g., no contacts), skip this timestep for PP collisions
        pass

    # Particle-Wall collisions (surfGeom)
    try:
        # Try to extract start and end times for particle-wall collisions
        start_times_pw = timestep.collision.surfGeom.getStartTime()
        end_times_pw   = timestep.collision.surfGeom.getEndTimes()
        durations_pw = end_times_pw - start_times_pw
        pw_durations.extend(durations_pw.tolist())
    except Exception as e:
        # If extraction fails (e.g., no contacts), skip this timestep for PW collisions
        pass

# Convert lists to NumPy arrays for statistics and plotting
pp_durations = np.array(pp_durations)
pw_durations = np.array(pw_durations)

# Compute basic statistics
pp_mean = np.mean(pp_durations) if pp_durations.size > 0 else 0
pp_std  = np.std(pp_durations) if pp_durations.size > 0 else 0
pw_mean = np.mean(pw_durations) if pw_durations.size > 0 else 0
pw_std  = np.std(pw_durations) if pw_durations.size > 0 else 0
pp_median = np.median(pp_durations)
pw_median = np.median(pw_durations)

# Assuming pp_durations and pw_durations are numpy arrays
pp_90 = np.percentile(pp_durations, 90)
pp_95 = np.percentile(pp_durations, 95)
pw_90 = np.percentile(pw_durations, 90)
pw_95 = np.percentile(pw_durations, 95)

print(f"Particle-Particle collision durations:")
print(f"  90th percentile: {pp_90:.6f} s")
print(f"  95th percentile: {pp_95:.6f} s")

print(f"Particle-Wall collision durations:")
print(f"  90th percentile: {pw_90:.6f} s")
print(f"  95th percentile: {pw_95:.6f} s")

print("Particle-Particle Collision Durations:")
print(f"  Mean: {pp_mean:.6f} s, Std: {pp_std:.6f} s")
print("Particle-Wall Collision Durations:")
print(f"  Mean: {pw_mean:.6f} s, Std: {pw_std:.6f} s")
print(f"Particle-Particle median duration: {pp_median:.6f} s")
print(f"Particle-Wall median duration: {pw_median:.6f} s")



# 1) Filter out any zero or negative durations if they exist (optional but recommended)
pp_durations = pp_durations[pp_durations > 0]
pw_durations = pw_durations[pw_durations > 0]

# 2) Build logarithmically spaced bins.
#    We'll set the lower bound as the min of the durations, the upper bound as the max.
#    You can also hard-code them if you want to ignore outliers.
bins_pp = np.logspace(np.log10(pp_durations.min()),
                      np.log10(pp_durations.max()),
                      50)  # 50 bins
bins_pw = np.logspace(np.log10(pw_durations.min()),
                      np.log10(pw_durations.max()),
                      50)

plt.figure(figsize=(12, 5))

# Particle-Particle
plt.subplot(1, 2, 1)
plt.hist(pp_durations, bins=bins_pp, color='blue', alpha=0.7)
plt.xscale('log')          # Use a log scale on the x-axis
plt.xlabel("Duration (s) [log scale]")
plt.ylabel("Frequency")
plt.title("Particle-Particle Collision Durations")

# Particle-Wall
plt.subplot(1, 2, 2)
plt.hist(pw_durations, bins=bins_pw, color='orange', alpha=0.7)
plt.xscale('log')
plt.xlabel("Duration (s) [log scale]")
plt.ylabel("Frequency")
plt.title("Particle-Wall Collision Durations")

plt.tight_layout()
plt.show()

