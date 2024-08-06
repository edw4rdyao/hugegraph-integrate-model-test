"""
GRAND (Graph Random Neural Network)

References
----------
Papers: https://arxiv.org/abs/2005.11079
Author's code: https://github.com/THUDM/GRAND
DGL code: https://github.com/dmlc/dgl/tree/master/examples/pytorch/grand
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GRAND(nn.Module):
    """
    Implementation of the GRAND model for graph representation learning.

    Parameters
    ----------
    n_in_feats : int
        Number of input features.
    n_hidden : int
        Number of hidden units.
    n_out_feats : int
        Number of output features.
    sample : int
        Number of augmentations.
    order : int
        Order of the graph convolution.
    p_drop_node : float
        Dropout rate for nodes.
    p_drop_input : float
        Dropout rate for input features.
    p_drop_hidden : float
        Dropout rate for hidden features.
    bn : bool
        Whether to use batch normalization.
    """
    def __init__(self, n_in_feats, n_hidden, n_out_feats, sample, order, p_drop_node, p_drop_input, p_drop_hidden, bn):
        super(GRAND, self).__init__()
        self.n_in_feats = n_in_feats
        self.n_hidden = n_hidden
        self.sample = sample
        self.order = order
        self.n_out_feats = n_out_feats

        # MLP for final prediction
        self.mlp = MLP(n_in_feats, n_hidden, n_out_feats, p_drop_input, p_drop_hidden, bn)
        # Graph convolution layer without weight
        self.graph_conv = GraphConv(n_in_feats, n_in_feats, norm='both', weight=False, bias=False)
        self.p_drop_node = p_drop_node  # Node dropout rate

    @staticmethod
    def consis_loss(logits, temp, lam):
        ps = torch.stack([torch.exp(logit) for logit in logits], dim=2)  # Stack logits and apply softmax
        avg_p = torch.mean(ps, dim=2)  # Average probabilities
        sharp_p = torch.pow(avg_p, 1.0 / temp)  # Sharpen probabilities
        sharp_p = sharp_p / sharp_p.sum(dim=1, keepdim=True)  # Normalize
        sharp_p = sharp_p.unsqueeze(2).detach()  # Detach from computation graph
        loss = lam * torch.mean((ps - sharp_p).pow(2).sum(dim=1))  # Consistency loss
        return loss

    def drop_node(self, feats):
        n = feats.shape[0]
        drop_rates = torch.FloatTensor(np.ones(n) * self.p_drop_node).to(feats.device)  # Node dropout rates
        masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(1)  # Dropout masks
        feats = masks.to(feats.device) * feats  # Apply dropout masks to features
        return feats

    def scale_node(self, feats):
        feats = feats * (1.0 - self.p_drop_node)  # Scale node features
        return feats

    def forward(self, graph, feats):
        logits_list = []
        for s in range(self.sample):
            X = self.drop_node(feats)  # Apply node dropout
            y = X
            for _ in range(self.order):
                X = self.graph_conv(graph, X)  # Graph convolution
                y = y + X  # Residual connection
            y = y / (self.order + 1)  # Normalize by order
            logits_list.append(torch.log_softmax(self.mlp(y), dim=-1))  # Prediction
        return logits_list

    def inference(self, graph, feats):
        X = self.scale_node(feats)  # Scale node features
        y = X
        for _ in range(self.order):
            X = self.graph_conv(graph, X)  # Graph convolution
            y = y + X  # Residual connection
        y = y / (self.order + 1)  # Normalize by order
        return torch.log_softmax(self.mlp(y), dim=-1)  # Final prediction


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for node feature transformation.

    Parameters
    ----------
    n_in_feats : int
        Number of input features.
    n_hidden : int
        Number of hidden units.
    n_out_feats : int
        Number of output features.
    p_input_drop : float
        Dropout rate for input features.
    p_hidden_drop : float
        Dropout rate for hidden features.
    bn : bool
        Whether to use batch normalization.
    """
    def __init__(self, n_in_feats, n_hidden, n_out_feats, p_input_drop, p_hidden_drop, bn):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(n_in_feats, n_hidden, bias=True)  # First linear layer
        self.layer2 = nn.Linear(n_hidden, n_out_feats, bias=True)  # Second linear layer
        self.p_input_drop = nn.Dropout(p_input_drop)  # Dropout for input features
        self.p_hidden_drop = nn.Dropout(p_hidden_drop)  # Dropout for hidden features
        self.bn1 = nn.BatchNorm1d(n_in_feats)  # Batch normalization for input features
        self.bn2 = nn.BatchNorm1d(n_hidden)  # Batch normalization for hidden features
        self.bn = bn  # Whether to use batch normalization
        self.layer1.reset_parameters()  # Reset parameters for the first layer
        self.layer2.reset_parameters()  # Reset parameters for the second layer

    def forward(self, x):
        if self.bn:
            x = self.bn1(x)  # Apply batch normalization to input features
        x = self.p_input_drop(x)  # Apply dropout to input features
        x = F.relu(self.layer1(x))  # Apply ReLU activation to the first layer

        if self.bn:
            x = self.bn2(x)  # Apply batch normalization to hidden features
        x = self.p_hidden_drop(x)  # Apply dropout to hidden features
        x = self.layer2(x)  # Apply the second linear layer

        return x  # Return final features
