"""
HAN (Heterogeneous Graph Attention Network)

References
----------
Paper: https://arxiv.org/abs/1903.07293
Author's code: https://github.com/Jhy1993/HAN
DGL code: https://github.com/dmlc/dgl/tree/master/examples/pytorch/han
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv


class HAN(nn.Module):
    """
    Heterogeneous Graph Attention Network (HAN) model.

    Parameters
    ----------
    n_meta_paths : int
        Number of meta-paths.
    n_in_feats : int
        Number of input features.
    n_hidden : int
        Number of hidden units.
    n_out_feats : int
        Number of output features.
    n_heads : list of int
        Number of attention heads for each layer.
    p_drop : float
        Dropout probability.
    """
    def __init__(self, n_meta_paths, n_in_feats, n_hidden, n_out_feats, n_heads, p_drop):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()  # List of HAN layers
        # First HAN layer
        self.layers.append(
            HANLayer(n_meta_paths, n_in_feats, n_hidden, n_heads[0], p_drop)
        )
        # Additional HAN layers
        for h in range(1, len(n_heads)):
            self.layers.append(
                HANLayer(n_meta_paths, n_hidden * n_heads[h - 1], n_hidden, n_heads[h], p_drop)
            )
        # Fully connected output layer
        self.fc = nn.Linear(n_hidden * n_heads[-1], n_out_feats)

    def forward(self, graphs, feats):
        # Pass input through each HAN layer
        for han_layer in self.layers:
            feats = han_layer(graphs, feats)

        return self.fc(feats)  # Final output through fully connected layer


class HANLayer(nn.Module):
    """
    HAN layer with GAT and semantic attention.

    Parameters
    ----------
    n_meta_paths : int
        Number of meta-paths.
    n_in_feats : int
        Number of input features.
    n_out_feats : int
        Number of output features.
    n_head : int
        Number of attention heads.
    p_drop : float
        Dropout probability.
    """
    def __init__(self, n_meta_paths, n_in_feats, n_out_feats, n_head, p_drop):
        super(HANLayer, self).__init__()
        # One GAT layer for each meta-path-based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(n_meta_paths):
            self.gat_layers.append(
                GATConv(n_in_feats, n_out_feats, n_head, p_drop, p_drop, activation=F.elu)
            )
        # Semantic attention mechanism
        self.semantic_attention = SemanticAttention(n_in_feats=n_out_feats * n_head)
        self.n_meta_paths = n_meta_paths

    def forward(self, graphs, feats):
        semantic_embeddings = []
        # Apply GAT layers to each meta-path graph
        for i, g in enumerate(graphs):
            semantic_embeddings.append(self.gat_layers[i](g, feats).flatten(1))  # Flatten the output
        # Stack the semantic embeddings for attention
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # Apply semantic attention and return (N, D * K)


class SemanticAttention(nn.Module):
    """
    Semantic attention mechanism for aggregating meta-path based embeddings.

    Parameters
    ----------
    n_in_feats : int
        Number of input features.
    n_hidden : int, optional
        Number of hidden units, by default 128.
    """
    def __init__(self, n_in_feats, n_hidden=128):
        super(SemanticAttention, self).__init__()

        # Two-layer feedforward network for computing attention scores
        self.project = nn.Sequential(
            nn.Linear(n_in_feats, n_hidden),  # Linear layer with tanh activation
            nn.Tanh(),
            nn.Linear(n_hidden, 1, bias=False),  # Output layer without bias
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # Compute attention scores across meta-paths (M, 1)
        beta = torch.softmax(w, dim=0)  # Normalize attention scores (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # Expand scores to match z (N, M, 1)
        return (beta * z).sum(1)  # Weighted sum of embeddings based on attention scores (N, D * K)
