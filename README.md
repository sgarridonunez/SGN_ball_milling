# Simulation Graph Network (SGN) Training

This project trains a Simulation Graph Network (SGN) model based on particle simulation data, specifically designed for predicting particle accelerations and global energy dissipation in a ball mill scenario.

The code is structured into modules for better organization and maintainability. Some modules contain post-saving analysis that is not critical, but can aid experimentation. Make sure to modify/delete them if they are not needed or incompatible with your work. 

## File Structure

*   **`main_modular.py`**: The main executable script. It orchestrates the data loading, model initialization, training loop, evaluation, and saving of results. Configuration parameters are set within this file.
*   **`data_utils.py`**: Contains utility functions for:
    *   Loading the raw simulation data (`.pkl` file).
    *   Loading and processing the geometry (`.stl` file).
    *   Building graph features (nodes, edges) from simulation snapshots, incorporating a time window.
    *   Calculating and applying normalization parameters for features and targets.
*   **`models.py`**: Defines the PyTorch neural network architectures, including:
    *   `MLP`: A basic Multi-Layer Perceptron block.
    *   `Encoder`: Encodes initial node and edge features.
    *   `MessagePassingLayer`: Performs a single layer of graph message passing.
    *   `Processor`: Stacks multiple message passing layers.
    *   `NodeDecoder`: Decodes node embeddings into predictions (e.g., accelerations).
    *   `GlobalReadout`: Pools node features to make global predictions (e.g., energy dissipation).
    *   `SGN`: The main graph neural network model integrating all components.
*   **`recursive_predictor_vectorized.py`**: Performs a recursive simulation using a trained model (vectorized). It initializes from ground truth data, then iteratively predicts future states, updates particle properties (SDF, contacts), and compares results against the ground truth, if available. 
*   **`data_extraction_final.py`**: Extracts training/validation data from an EDEM `.dem` simulation and a `stl` mesh. It:
    *   Loads and transforms the STL mesh to a consistent coordinate system.
    *   Reads snapshots from the `.dem` deck via `edempy`.
    *   Computes per-particle SDF values, normals, and finite-difference gradients w.r.t. the moved mesh (via CoM alignment).
    *   Collects particle-particle and particle-wall contact info and energy dissipation increments.
    *   Saves a list of snapshot dictionaries to `extracted_data_300_fine.pkl`.

## Dependencies

The following Python libraries are required:

*   `torch`: PyTorch deep learning framework.
*   `torch-geometric`: Geometric deep learning extension library for PyTorch.
*   `numpy`: Numerical computing library.
*   `matplotlib`: Plotting library (for generating loss curves).
*   `trimesh`: Library for loading and processing mesh files (STL).
*   `torch-scatter` (Optional, Recommended): For optimized scatter operations within PyTorch Geometric. If not installed, the code will fall back to less efficient manual scatter operations.
*   `edempy` (For data extraction only): Required to parse EDEM `.dem` decks when running `data_extraction_final.py`.

You can typically install these using pip:
```bash
pip install torch numpy matplotlib trimesh torch-geometric
# Optional, but recommended: Install torch-scatter matching your PyTorch/CUDA version
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html
# Also requires scipy for interpolation in recursive_predictor_vectorized.py
pip install scipy
# For data extraction from EDEM decks (requires EDEM installation/license)
# edempy is typically provided with EDEM; ensure it's importable in your environment
```

## Configuration

Key parameters for the training and recursive prediction processes can be configured directly within the respective scripts (`main_modular.py`, `recursive_predictor_vectorized.py`) under the `# --- Configuration --- #` section. This includes:

*   **File Paths**: `DATA_FILE`, `STL_FILE`, `NORM_PARAMS_FILE`, `LOSS_HISTORY_FILE`, `CHECKPOINT_PATH`, `BEST_MODEL_PATH`.
*   **Model Hyperparameters**: `WINDOW_SIZE`, `HIDDEN_DIM`, `MLP_LAYERS`, `INTERACTION_LAYERS`, `DROPOUT_RATE`, etc.
*   **Training Parameters**: `NUM_TRAIN_SAMPLES`, `BATCH_SIZE`, `NUM_EPOCHS`, `INITIAL_LR`, `FINAL_LR`, `HUBER_BETA`, `LOSS_ALPHA`, `VELOCITY_NOISE_STD`.
*   **Flags**: `USE_LAST_SNAPSHOT_GLOBAL`, `RESUME_TRAINING` (in `main_modular.py`).
*   **Simulation Parameters**: `DT`, `TOTAL_SIM_TIME`, `INTEGRATION_TYPE`, `CONTACT_THRESHOLD_PP`, `CONTACT_THRESHOLD_PW_SDF`, correction/snapback thresholds (in `recursive_predictor_vectorized.py`).

Modify these values as needed before running the scripts.

## Input Data

The script expects the following input files in the same directory (or adjust paths in configuration):

1.  **`extracted_data_300_fine.pkl`**: A Python pickle file containing a list of dictionaries, where each dictionary represents a simulation snapshot. Each snapshot contains particle data (IDs, velocities, SDF values/gradients, net forces), contact information (particle-particle, particle-wall), wall features, and energy increments. This file can be generated from an EDEM simulation deck (`.dem`) using `data_extraction_final.py` (included here). It is used for both training (`main_modular.py`) and initialization/comparison in recursive prediction (`recursive_predictor_vectorized.py`).
2.  **`ball_mill_jar.stl`**: An STL file representing the static geometry of the simulation environment (e.g., the ball mill jar). This file is used by `data_extraction_final.py`, `data_utils.py`, and `recursive_predictor_vectorized.py`.
3.  **`best_model_modular.pth`**: The trained model weights saved by `main_modular.py`. This is required as input for `recursive_predictor_vectorized.py`.
4.  **`normalization_params.pkl`**: Normalization statistics saved by `main_modular.py`. This is required as input for `recursive_predictor_vectorized.py`.

## Usage

### 1. Training the Model

To train the SGN model, navigate to the directory containing the scripts in your terminal and run:

```bash
python main_modular.py
```

This script will:
*   Load the data (`extracted_data_300_fine.pkl`) and geometry (`ball_mill_jar.stl`).
*   Build the dataset using the specified time window.
*   Split the data into training and testing sets.
*   Calculate and save normalization parameters (`normalization_params.pkl`).
*   Normalize the datasets.
*   Initialize the SGN model, optimizer, and learning rate scheduler.
*   Optionally resume training from a checkpoint (`sgn_checkpoint_modular.pth`) if `RESUME_TRAINING` is `True`.
*   Run the training loop, printing losses and saving checkpoints.
*   Save the best model weights (`best_model_modular.pth`).
*   Save the final loss history (`loss_history_modular.pkl`) and generate a loss plot (`loss_plot_modular.png`).

### 2. Running Recursive Prediction

After training the model and ensuring `best_model_modular.pth` and `normalization_params.pkl` are present, run the recursive simulation:

```bash
python recursive_predictor_vectorized.py
```

This script will:
*   Load the trained model, normalization parameters, ground truth data, and geometry.
*   Initialize a simulation window using the first few snapshots from the ground truth data.
*   Iteratively perform the following steps for the configured `TOTAL_SIM_TIME`:
    *   Build graph features for the current window.
    *   Normalize the features.
    *   Predict particle accelerations and global energy increment using the loaded SGN model.
    *   Denormalize the predictions.
    *   Optionally apply acceleration corrections based on SDF penetration.
    *   Update particle positions and velocities using the chosen integration scheme (`euler` or `trapezoidal`).
    *   Optionally apply velocity damping based on SDF penetration.
    *   Optionally project particles back ("snap-back") if they penetrate the wall beyond a threshold.
    *   Create a new snapshot dictionary containing the updated particle state, recomputed contacts, SDF values/gradients, and interpolated wall features.
    *   Update the simulation window.
    *   Compare the predicted state (positions, SDF, energy) against the ground truth data at corresponding time steps and store MSE values.
    *   Save predicted particle positions periodically.
*   After the loop, save simulation results (MSE dictionaries, energy increments, predicted positions, wall CoMs).
*   Generate plots comparing predicted vs. ground truth MSE and cumulative energy dissipation.

### 3. Generating Input Data (optional)

To produce `extracted_data_300_fine.pkl` from an EDEM simulation:

```bash
python data_extraction_final.py
```

Requirements:
- The `.dem` file must be in the working directory.
- `ball_mill_jar.stl` must be present.
- `edempy` must be available in your Python environment.

## Outputs

During and after execution, the scripts generate the following files:

### Training (`main_modular.py`)

*   `normalization_params.pkl`: Saved normalization statistics (mean, std) calculated from the training data. Used by `recursive_predictor_vectorized.py`.
*   `sgn_checkpoint_modular.pth`: Checkpoint file saved periodically during training.
*   `best_model_modular.pth`: State dictionary of the best performing model. Used by `recursive_predictor.py`.
*   `loss_history_modular.pkl`: Training and testing loss history.
*   `loss_plot_modular.png`: Plot of training and testing loss vs. epoch.

### Recursive Prediction (`recursive_predictor_vectorized.py`)
*   `positions_recursive_vectorized.pkl`: Dictionary mapping time steps to predicted particle positions during the recursive simulation.
*   `coms_recursive_vectorized.pkl`: Dictionary mapping time steps to the interpolated wall center-of-mass used during the recursive simulation.
*   `model_results_recursive_vectorized.pkl`: Dictionary containing MSE values (position, SDF, energy) and predicted/ground truth energy increments.
*   `mse_plot_recursive_vectorized.png`: Plot showing MSE of positions and SDF over time.
*   `energy_plot_direct_recursive_vectorized.png`: Plot comparing the direct cumulative sum of predicted vs. ground truth energy dissipation.
*   `energy_plot_block_avg_recursive_vectorized.png`: Plot comparing the block-averaged cumulative sum of predicted vs. ground truth energy dissipation.
