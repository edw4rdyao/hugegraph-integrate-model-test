"""
Deep Graph Infomax(DGI)

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
DGL code: https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgi
"""

import math
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class GCNEncoder(nn.Module):
    r"""
    A GCN-based encoder module which applies graph convolutions to input features.

    Parameters
    -----------
    n_in_feats: int
        Input feature size.
    n_hidden: int
        Hidden feature size.
    n_layers: int
        Number of the GNN encoder layers(>=2).
    p_dropout: float
        Dropout rate for regularization.
    """

    def __init__(self, n_in_feats, n_hidden, n_layers, p_dropout):
        super(GCNEncoder, self).__init__()
        # Initialize the input layer with PReLU activation function.
        self.input_layer = GraphConv(n_in_feats, n_hidden, activation=nn.PReLU(n_hidden))
        # Initialize hidden layers within a ModuleList.
        self.hidden_layers = nn.ModuleList()
        assert n_layers >= 2
        for _ in range(n_layers - 2):
            self.hidden_layers.append(GraphConv(n_hidden, n_hidden, activation=nn.PReLU(n_hidden)))
        # Initialize the output layer without an activation function.
        self.output_layer = GraphConv(n_hidden, n_hidden)
        # Initialize the dropout module.
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, graph, feat, corrupt=False):
        # Optionally corrupt the features to simulate negative samples.
        if corrupt:
            perm = torch.randperm(graph.num_nodes())
            feat = feat[perm]
        # Apply graph convolutions interleaved with dropout.
        feat = self.input_layer(graph, feat)
        feat = self.dropout(feat)
        for hidden_layer in self.hidden_layers:
            feat = hidden_layer(graph, feat)
            feat = self.dropout(feat)
        feat = self.output_layer(graph, feat)
        return feat


class Discriminator(nn.Module):
    r"""
    A discriminator module for distinguishing between real and corrupted embeddings.

    Parameters
    -----------
    n_hidden: int
        Hidden dimension size used for bilinear transformation.
    """

    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        # Initialize weight parameters for bilinear transformation.
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.uniform_weight()

    def uniform_weight(self):
        # Uniformly initialize weights within a certain bound.
        bound = 1.0 / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-bound, bound)

    def forward(self, feat, summary):
        # Apply bilinear transformation to the feature and summary vector.
        feat = torch.matmul(feat, torch.matmul(self.weight, summary))
        return feat


class DGI(nn.Module):
    r"""
    Deep Graph Infomax (DGI) model that maximizes mutual information between node embeddings and a graph summary.

    Parameters
    -----------
    n_in_feats: int
        Input feature size.
    n_hidden: int
        Hidden feature size.
    n_layers: int
        Number of the GNN encoder layers.
    dropout: float
        Dropout rate for regularization.
    """

    def __init__(self, n_in_feats, n_hidden=512, n_layers=1, dropout=0):
        super(DGI, self).__init__()
        # Initialize the encoder.
        self.encoder = GCNEncoder(n_in_feats, n_hidden, n_layers, dropout)
        # Initialize the discriminator.
        self.discriminator = Discriminator(n_hidden)
        # Initialize the loss function for binary cross-entropy.
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, graph, feat):
        # Encode positive and negative samples.
        positive = self.encoder(graph, feat, corrupt=False)
        negative = self.encoder(graph, feat, corrupt=True)
        # Compute summary vector.
        summary = torch.sigmoid(positive.mean(dim=0))
        # Discriminate between positive and negative samples.
        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)
        # Calculate losses for positive and negative samples.
        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))
        return l1 + l2


class Classifier(nn.Module):
    r"""
    Classifier module to predict node classes based on embeddings.

    Parameters
    -----------
    n_hidden: int
        Hidden feature size.
    n_classes: int
        Number of classes to predict.
    """

    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        # Initialize the fully connected layer for classification.
        self.fc = nn.Linear(n_hidden, n_classes)
        self.fc.reset_parameters()

    def forward(self, feat):
        # Apply linear transformation and log softmax for classification probabilities.
        feat = self.fc(feat)
        return torch.log_softmax(feat, dim=-1)
