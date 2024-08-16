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
    Heterogeneous Graph Attention Network (HAN) model for learning node representations on heterogeneous graphs.

    Parameters
    ----------
    n_meta_paths : int
        Number of meta-paths.
    n_in_feats : int
        Number of input features per node.
    n_hidden : int
        Number of hidden units in each HAN layer.
    n_out_feats : int
        Number of output features or classes.
    n_heads : list of int
        Number of attention heads for each HAN layer.
    p_drop : float
        Dropout probability for each layer.
    """

    def __init__(self, n_meta_paths, n_in_feats, n_hidden, n_out_feats, n_heads, p_drop):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()  # List to hold all HAN layers

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
        self.fc = nn.Linear(n_hidden * n_heads[-1], n_out_feats)  # Linear layer for final output

    def forward(self, graphs, feats):
        """
        Forward pass through the HAN model.

        Parameters
        ----------
        graphs : list of dgl.DGLGraph
            List of meta-path based graphs.
        feats : torch.Tensor
            Input node features.

        Returns
        -------
        torch.Tensor
            Output node representations or class scores.
        """
        for han_layer in self.layers:
            feats = han_layer(graphs, feats)  # Pass input through each HAN layer

        return self.fc(feats)  # Final output through the fully connected layer


class HANLayer(nn.Module):
    """
    Single HAN layer combining GATConv and semantic attention mechanisms.

    Parameters
    ----------
    n_meta_paths : int
        Number of meta-paths.
    n_in_feats : int
        Number of input features per node.
    n_out_feats : int
        Number of output features per node.
    n_head : int
        Number of attention heads in the GAT layer.
    p_drop : float
        Dropout probability for the GAT layer.
    """

    def __init__(self, n_meta_paths, n_in_feats, n_out_feats, n_head, p_drop):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()  # List to hold GAT layers for each meta-path
        for _ in range(n_meta_paths):
            self.gat_layers.append(
                GATConv(n_in_feats, n_out_feats, n_head, p_drop, p_drop, activation=F.elu)
            )
        self.semantic_attention = SemanticAttention(n_in_feats=n_out_feats * n_head)  # Semantic attention mechanism
        self.n_meta_paths = n_meta_paths  # Number of meta-paths

    def forward(self, graphs, feats):
        """
        Forward pass through the HAN layer.

        Parameters
        ----------
        graphs : list of dgl.DGLGraph
            List of meta-path based graphs.
        feats : torch.Tensor
            Input node features.

        Returns
        -------
        torch.Tensor
            Node embeddings after applying GAT and semantic attention.
        """
        semantic_embeddings = []
        for i, g in enumerate(graphs):
            # Apply GAT to each meta-path graph and flatten output
            semantic_embeddings.append(self.gat_layers[i](g, feats).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # Stack the embeddings along a new dimension

        # Apply semantic attention and return the aggregated embeddings
        return self.semantic_attention(semantic_embeddings)


class SemanticAttention(nn.Module):
    """
    Semantic attention mechanism for aggregating meta-path-based embeddings.

    Parameters
    ----------
    n_in_feats : int
        Number of input features per node.
    n_hidden : int, optional
        Number of hidden units in the attention mechanism. Default is 128.
    """

    def __init__(self, n_in_feats, n_hidden=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(n_in_feats, n_hidden),  # Linear transformation followed by Tanh activation
            nn.Tanh(),
            nn.Linear(n_hidden, 1, bias=False),  # Linear transformation without bias to get attention scores
        )

    def forward(self, z):
        """
        Forward pass through the semantic attention mechanism.

        Parameters
        ----------
        z : torch.Tensor
            Stacked node embeddings from different meta-paths (shape: (N, M, D)).

        Returns
        -------
        torch.Tensor
            Aggregated node embeddings based on the learned attention scores (shape: (N, D)).
        """
        w = self.project(z).mean(0)  # Compute attention scores across meta-paths (M, 1)
        beta = torch.softmax(w, dim=0)  # Normalize attention scores across meta-paths
        beta = beta.expand((z.shape[0],) + beta.shape)  # Expand the attention scores to match the shape of z
        return (beta * z).sum(1)  # Compute the weighted sum of embeddings based on attention scores and return
