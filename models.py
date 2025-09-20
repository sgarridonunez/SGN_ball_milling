#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch model definitions for the SGN architecture, including MLP blocks,
encoder, processor, decoders, and the main SGN model. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for torch_scatter availability
try:
    from torch_scatter import scatter_add, scatter_mean, scatter_max
    use_torch_scatter = True
except ImportError:
    use_torch_scatter = False
    print("Warning: torch_scatter not found. Falling back to manual scatter operations for 'mean' aggregation.")

# --- Basic MLP Block ---
class MLP(nn.Module):
    """Multi-Layer Perceptron with ReLU activations and optional dropout."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=5, dropout_rate=0.2):
        super(MLP, self).__init__()
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")
        
        layers = []
        dim = in_dim
        # Input layer (if num_layers == 1, this is also the output layer)
        if num_layers > 1:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            dim = hidden_dim
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(dim, out_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Global Readout Module ---
class GlobalReadout(nn.Module):
    """Pools node features and applies an MLP for global prediction."""
    def __init__(self, hidden_dim, out_dim=1, mlp_layers=3, dropout_rate=0.2):
        super(GlobalReadout, self).__init__()
        # Define the MLP for processing pooled features
        self.mlp = MLP(hidden_dim, hidden_dim, out_dim, num_layers=mlp_layers, dropout_rate=dropout_rate)

    def forward(self, h_v, edge_index, batch):
        """
        Forward pass for global readout.

        Args:
            h_v (Tensor): Node features (shape: [num_nodes, hidden_dim]).
            edge_index (Tensor): Edge indices (shape: [2, num_edges]).
            batch (Tensor): Batch assignment for each node (shape: [num_nodes]).

        Returns:
            Tensor: Global predictions for each graph in the batch (shape: [num_graphs, out_dim]).
        """
        # --- Option 1: Pool only nodes involved in edges (as in original paper) ---
        # nodes_with_edge = torch.zeros(h_v.size(0), dtype=torch.bool, device=h_v.device)
        # if edge_index.numel() > 0: # Check if there are any edges
        #     nodes_with_edge[edge_index.view(-1)] = True
        # --- Option 2: Pool all nodes ---
        nodes_with_edge = torch.ones(h_v.size(0), dtype=torch.bool, device=h_v.device) # Uncomment this line to pool all nodes

        global_preds = []
        unique_graphs = torch.unique(batch)

        for g in unique_graphs:
            # Create mask for nodes belonging to the current graph 'g'
            graph_mask = (batch == g)
            # Combine with node selection criteria (e.g., nodes_with_edge)
            mask = graph_mask & nodes_with_edge

            if mask.sum() > 0:
                # Pool features of selected nodes (mean pooling)
                pooled_g = h_v[mask].mean(dim=0, keepdim=True)
                # Apply MLP to pooled features
                pred = self.mlp(pooled_g)
            else:
                # Handle cases where no nodes meet the criteria (e.g., isolated graph with pooling option 1)
                # Get output dimension from the last layer of the MLP
                out_dim = self.mlp.net[-1].out_features if isinstance(self.mlp.net[-1], nn.Linear) else 1
                pred = torch.zeros((1, out_dim), device=h_v.device, dtype=h_v.dtype)

            global_preds.append(pred)

        # Concatenate predictions for all graphs in the batch
        global_pred = torch.cat(global_preds, dim=0)
        return global_pred

# --- Encoder Module ---
class Encoder(nn.Module):
    """Encodes node and edge features using separate MLPs."""
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, mlp_layers=5, dropout_rate=0.2):
        super(Encoder, self).__init__()
        self.node_encoder = MLP(node_in_dim, hidden_dim, hidden_dim, num_layers=mlp_layers, dropout_rate=dropout_rate)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, hidden_dim, num_layers=mlp_layers, dropout_rate=dropout_rate)

    def forward(self, node_features, edge_features):
        h_v = self.node_encoder(node_features)
        # Handle case with no edges
        if edge_features.numel() > 0:
             h_e = self.edge_encoder(edge_features)
        else:
             # Create an empty tensor with the correct hidden dimension if there are no edges
             h_e = torch.empty((0, self.edge_encoder.net[-1].out_features),
                               dtype=node_features.dtype, device=node_features.device)
        return h_v, h_e

# --- Message Passing Layer ---
class MessagePassingLayer(nn.Module):
    """Performs one layer of message passing."""
    def __init__(self, hidden_dim=64, aggregation="mean", mlp_layers=5, dropout_rate=0.2):
        super(MessagePassingLayer, self).__init__()
        if aggregation not in ["mean", "sum"]: # Add more aggregations if needed
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        self.aggregation = aggregation
        self.message_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim, num_layers=mlp_layers, dropout_rate=dropout_rate)
        self.update_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layers=mlp_layers, dropout_rate=dropout_rate)

    def forward(self, h_v, edge_index, h_e):
        # Handle case with no edges
        if edge_index.numel() == 0:
            # If no edges, the aggregated message is zero, update is just based on original h_v
            aggregated = torch.zeros_like(h_v)
            update_input = torch.cat([h_v, aggregated], dim=-1)
            h_v_updated = self.update_mlp(update_input)
            return h_v_updated

        src, dst = edge_index
        h_src = h_v[src]
        h_dst = h_v[dst]

        # Create messages
        message_input = torch.cat([h_src, h_dst, h_e], dim=-1)
        messages = self.message_mlp(message_input)

        # Aggregate messages
        num_nodes = h_v.size(0)
        if self.aggregation == "mean":
            if use_torch_scatter:
                aggregated = scatter_mean(messages, dst, dim=0, dim_size=num_nodes)
            else:
                aggregated = torch.zeros_like(h_v)
                aggregated = aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
                counts = torch.zeros(num_nodes, dtype=messages.dtype, device=h_v.device)
                ones = torch.ones(dst.size(0), dtype=messages.dtype, device=h_v.device)
                counts = counts.scatter_add_(0, dst, ones)
                aggregated = aggregated / counts.clamp(min=1).unsqueeze(-1)
        elif self.aggregation == "sum":
             if use_torch_scatter:
                 aggregated = scatter_add(messages, dst, dim=0, dim_size=num_nodes)
             else:
                 aggregated = torch.zeros_like(h_v)
                 aggregated = aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        else:
             raise NotImplementedError(f"Aggregation '{self.aggregation}' not implemented without torch_scatter.")


        # Update node features
        update_input = torch.cat([h_v, aggregated], dim=-1)
        h_v_updated = self.update_mlp(update_input)
        return h_v_updated

# --- Processor Module ---
class Processor(nn.Module):
    """Applies multiple MessagePassingLayers sequentially."""
    def __init__(self, hidden_dim=64, interaction_layers=6, aggregation="mean", mlp_layers=5, dropout_rate=0.2):
        super(Processor, self).__init__()
        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim=hidden_dim, aggregation=aggregation, mlp_layers=mlp_layers, dropout_rate=dropout_rate)
            for _ in range(interaction_layers)
        ])

    def forward(self, h_v, edge_index, h_e):
        for layer in self.layers:
            h_v = layer(h_v, edge_index, h_e)
        return h_v

# --- Node Decoder Module ---
class NodeDecoder(nn.Module):
    """Decodes final node embeddings into node-level predictions."""
    def __init__(self, hidden_dim=64, out_dim=3, mlp_layers=5, dropout_rate=0.2):
        super(NodeDecoder, self).__init__()
        self.decoder = MLP(hidden_dim, hidden_dim, out_dim, num_layers=mlp_layers, dropout_rate=dropout_rate)

    def forward(self, h_v):
        return self.decoder(h_v)

# --- Main SGN Model ---
class SGN(nn.Module):
    """
    The main Simulation Graph Network (SGN) model combining Encoder, Processor,
    NodeDecoder, and GlobalReadout.
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64,
                 node_out_dim=3, global_out_dim=1, mlp_layers=5, interaction_layers=4,
                 aggregation="mean", dropout_rate=0.2, use_last_snapshot_global=False,
                 particle_feature_dim=7): # Add particle_feature_dim
        super(SGN, self).__init__()
        self.use_last_snapshot_global = use_last_snapshot_global
        self.particle_feature_dim = particle_feature_dim # Store for global encoder

        # Core components
        self.encoder = Encoder(node_in_dim, edge_in_dim, hidden_dim=hidden_dim, mlp_layers=mlp_layers, dropout_rate=dropout_rate)
        self.processor = Processor(hidden_dim=hidden_dim, interaction_layers=interaction_layers, aggregation=aggregation, mlp_layers=mlp_layers, dropout_rate=dropout_rate)
        self.node_decoder = NodeDecoder(hidden_dim, node_out_dim, mlp_layers, dropout_rate=dropout_rate)
        self.global_readout = GlobalReadout(hidden_dim, out_dim=global_out_dim, mlp_layers=mlp_layers, dropout_rate=dropout_rate)

        # Optional separate encoder for global prediction using only the last snapshot's features
        if self.use_last_snapshot_global:
            # Calculate input dimension for the last snapshot's features
            # Assumes node_in_dim is window_size * particle_feature_dim
            # window_size = node_in_dim // self.particle_feature_dim # Calculate window size dynamically
            node_in_dim_last = self.particle_feature_dim # Input dim is just features from one snapshot
            self.global_encoder = MLP(node_in_dim_last, hidden_dim, hidden_dim, num_layers=mlp_layers, dropout_rate=dropout_rate)
        else:
            self.global_encoder = None # Not needed if using processed features

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1. Encode initial node and edge features
        h_v, h_e = self.encoder(x, edge_attr)

        # 2. Process features through interaction layers
        h_v_processed = self.processor(h_v, edge_index, h_e)

        # 3. Decode node features for node-level predictions
        node_pred = self.node_decoder(h_v_processed)

        # 4. Global prediction
        if self.use_last_snapshot_global:
            if not hasattr(data, 'x_last'):
                 raise AttributeError("Data object must have 'x_last' attribute when use_last_snapshot_global is True.")
            # Encode features from the last snapshot separately
            h_v_last = self.global_encoder(data.x_last)
            # Apply global readout to these encoded features
            global_pred = self.global_readout(h_v_last, edge_index, batch)
        else:
            # Apply global readout to the processed node features from the interaction layers
            global_pred = self.global_readout(h_v_processed, edge_index, batch)

        return node_pred, global_pred
