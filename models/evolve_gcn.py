"""
EvolveGCN (Evolving Graph Convolutional Networks)

References
----------
Paper: https://arxiv.org/abs/1902.10191
Author's code: https://github.com/IBM/EvolveGCN
Ref DGL code: https://github.com/dmlc/dgl/tree/master/examples/pytorch/evolveGCN
Ref pyG code: https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn
/recurrent/evolvegcno.py
"""

import torch
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
from torch.nn.parameter import Parameter


class MatGRUCell(nn.Module):
    """
    Matrix-based Gated Recurrent Unit (GRU) Cell for EvolveGCN.
    This cell handles the update, reset, and candidate hidden state calculations in a GRU cell.

    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = MatGRUGate(in_feats, out_feats, activation=torch.nn.Sigmoid())
        self.reset = MatGRUGate(in_feats, out_feats, activation=torch.nn.Sigmoid())
        self.htilda = MatGRUGate(in_feats, out_feats, activation=torch.nn.Tanh())

    def forward(self, prev_Q, z_topk=None):
        """
        Forward pass for the GRU cell.

        Parameters
        ----------
        prev_Q : torch.Tensor
            The previous hidden state (Q) or weight matrix.
        z_topk : torch.Tensor, optional
            The top-K node embeddings or features. If not provided, prev_Q is used.

        Returns
        -------
        torch.Tensor
            The updated hidden state or weight matrix (new_Q).
        """
        if z_topk is None:
            z_topk = prev_Q

        update = self.update(z_topk, prev_Q)  # Update gate
        reset = self.reset(z_topk, prev_Q)  # Reset gate

        h_cap = reset * prev_Q  # Apply reset gate
        h_cap = self.htilda(z_topk, h_cap)  # Calculate candidate hidden state

        new_Q = (1 - update) * prev_Q + update * h_cap  # Compute the new hidden state

        return new_Q


class MatGRUGate(torch.nn.Module):
    """
    A gate in the GRU cell performing a bilinear transformation.

    Parameters
    ----------
    rows : int
        Number of rows in the weight matrices.
    cols : int
        Number of columns in the bias.
    activation : nn.Module
        Activation function for the gate.
    """

    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))  # Weight matrix W
        self.U = Parameter(torch.Tensor(rows, rows))  # Weight matrix U
        self.bias = Parameter(torch.Tensor(rows, cols))  # Bias term
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights using Xavier uniform distribution.
        """
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.bias)

    def forward(self, x, hidden):
        """
        Forward pass for the GRU gate.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (current input).
        hidden : torch.Tensor
            Hidden state tensor (previous hidden state).

        Returns
        -------
        torch.Tensor
            The output of the gate after applying the activation function.
        """
        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)  # Bilinear transformation and activation
        return out


class TopK(torch.nn.Module):
    """
    A layer that selects the top-K nodes based on their feature importance scores.
    Similar to the official `egcn_h.py`. We only consider the node in a timestamp based subgraph,
    so we need to pay attention to `K` should be less than the min node numbers in all subgraph.
    Please refer to section 3.4 of the paper for the formula.

    Parameters
    ----------
    feats : int
        Number of features in the input node embeddings.
    k : int
        Number of top nodes to select.
    """

    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))  # Scorer vector to compute node importance
        self.k = k  # Number of top-K nodes to select
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the scorer vector using Xavier uniform distribution.
        """
        nn.init.xavier_uniform_(self.scorer)

    def forward(self, node_embs):
        """
        Forward pass to select top-K nodes.

        Parameters
        ----------
        node_embs : torch.Tensor
            Node embeddings.

        Returns
        -------
        torch.Tensor
            Transposed embeddings of the selected top-K nodes, weighted by their scores.
        """
        scores = node_embs.matmul(self.scorer) / self.scorer.norm().clamp(min=1e-6)  # Compute importance scores
        vals, topk_indices = scores.view(-1).topk(self.k)  # Select top-K nodes
        out = node_embs[topk_indices] * torch.tanh(scores[topk_indices].view(-1, 1))  # Weight embeddings by their scores
        return out.t()  # Transpose the output


class EvolveGCNH(nn.Module):
    """
    EvolveGCN-H model which evolves GCN parameters using a GRU.

    Parameters
    ----------
    in_feats : int, optional
        Number of input features per node. Default is 166.
    n_hidden : int, optional
        Number of hidden features per node. Default is 76.
    num_layers : int, optional
        Number of GCN layers. Default is 2.
    n_classes : int, optional
        Number of output classes. Default is 2.
    classifier_hidden : int, optional
        Number of hidden units in the classifier MLP. Default is 510.
    """

    def __init__(
        self,
        in_feats=166,
        n_hidden=76,
        num_layers=2,
        n_classes=2,
        classifier_hidden=510,
    ):
        super(EvolveGCNH, self).__init__()
        self.num_layers = num_layers
        self.pooling_layers = nn.ModuleList()
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        # Initialize pooling, recurrent, and GCN layers
        self.pooling_layers.append(TopK(in_feats, n_hidden))
        self.recurrent_layers.append(MatGRUCell(in_feats=in_feats, out_feats=n_hidden))
        self.gcn_weights_list.append(Parameter(torch.Tensor(in_feats, n_hidden)))
        self.gnn_convs.append(
            GraphConv(
                in_feats=in_feats,
                out_feats=n_hidden,
                bias=False,
                activation=nn.RReLU(),
                weight=False,
            )
        )
        for _ in range(num_layers - 1):
            self.pooling_layers.append(TopK(n_hidden, n_hidden))
            self.recurrent_layers.append(MatGRUCell(in_feats=n_hidden, out_feats=n_hidden))
            self.gcn_weights_list.append(Parameter(torch.Tensor(n_hidden, n_hidden)))
            self.gnn_convs.append(
                GraphConv(
                    in_feats=n_hidden,
                    out_feats=n_hidden,
                    bias=False,
                    activation=nn.RReLU(),
                    weight=False,
                )
            )

        # Initialize MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Linear(classifier_hidden, n_classes),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize GCN weights using Xavier uniform distribution.
        """
        for gcn_weight in self.gcn_weights_list:
            nn.init.xavier_uniform_(gcn_weight)

    def forward(self, g_list):
        """
        Forward pass through the EvolveGCN-H model.

        Parameters
        ----------
        g_list : list of dgl.DGLGraph
            List of graph snapshots (one for each timestamp).

        Returns
        -------
        torch.Tensor
            Output predictions of the model.
        """
        feature_list = [g.ndata["feat"] for g in g_list]  # Extract node features from each graph snapshot
        for i in range(self.num_layers):
            W = self.gcn_weights_list[i]
            for j, g in enumerate(g_list):
                X_tilde = self.pooling_layers[i](feature_list[j])
                W = self.recurrent_layers[i](W, X_tilde)
                feature_list[j] = self.gnn_convs[i](g, feature_list[j], weight=W)
        return self.mlp(feature_list[-1])  # Apply MLP classifier to the last snapshot


class EvolveGCNO(nn.Module):
    """
    EvolveGCN-O model which evolves GCN parameters using a GRU.

    Parameters
    ----------
    in_feats : int, optional
        Number of input features per node. Default is 166.
    n_hidden : int, optional
        Number of hidden features per node. Default is 256.
    num_layers : int, optional
        Number of GCN layers. Default is 2.
    n_classes : int, optional
        Number of output classes. Default is 2.
    classifier_hidden : int, optional
        Number of hidden units in the classifier MLP. Default is 307.
    """

    def __init__(
        self,
        in_feats=166,
        n_hidden=256,
        num_layers=2,
        n_classes=2,
        classifier_hidden=307,
    ):
        super(EvolveGCNO, self).__init__()
        self.num_layers = num_layers
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        # Initialize recurrent and GCN layers

        # In the paper, EvolveGCN-O use LSTM as RNN layer. According to the official code, EvolveGCN-O use GRU as RNN
        # layer. Here we follow the official code. See:
        # https://github.com/IBM/EvolveGCN/blob/90869062bbc98d56935e3d92e1d9b1b4c25be593/egcn_o.py#L53 PS: I try to
        # use torch.nn.LSTM directly, like [pyg_temporal](
        # github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent
        # /evolvegcno.py) but the performance is worse than use torch.nn.GRU. PPS: I think torch.nn.GRU can't match
        # the manually implemented GRU cell in the official code, we follow the official code here.
        self.recurrent_layers.append(MatGRUCell(in_feats=in_feats, out_feats=n_hidden))
        self.gcn_weights_list.append(Parameter(torch.Tensor(in_feats, n_hidden)))
        self.gnn_convs.append(
            GraphConv(
                in_feats=in_feats,
                out_feats=n_hidden,
                bias=False,
                activation=nn.RReLU(),
                weight=False,
            )
        )
        for _ in range(num_layers - 1):
            self.recurrent_layers.append(MatGRUCell(in_feats=n_hidden, out_feats=n_hidden))
            self.gcn_weights_list.append(Parameter(torch.Tensor(n_hidden, n_hidden)))
            self.gnn_convs.append(
                GraphConv(
                    in_feats=n_hidden,
                    out_feats=n_hidden,
                    bias=False,
                    activation=nn.RReLU(),
                    weight=False,
                )
            )

        # Initialize MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Linear(classifier_hidden, n_classes),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize GCN weights using Xavier uniform distribution.
        """
        for gcn_weight in self.gcn_weights_list:
            nn.init.xavier_uniform_(gcn_weight)

    def forward(self, g_list):
        """
        Forward pass through the EvolveGCN-O model.

        Parameters
        ----------
        g_list : list of dgl.DGLGraph
            List of graph snapshots (one for each timestamp).

        Returns
        -------
        torch.Tensor
            Output predictions of the model.
        """
        feature_list = [g.ndata["feat"] for g in g_list]  # Extract node features from each graph snapshot
        for i in range(self.num_layers):
            W = self.gcn_weights_list[i]
            for j, g in enumerate(g_list):
                # Attention: I try to use the below code to set gcn.weight(similar to pyG_temporal),
                # but it doesn't work. It seems that the gradient function lost in this situation,
                # more discussion see here: https://github.com/benedekrozemberczki/pytorch_geometric_temporal/issues/80
                # ====================================================
                # W = self.gnn_convs[i].weight[None, :, :]
                # W, _ = self.recurrent_layers[i](W)
                # self.gnn_convs[i].weight = nn.Parameter(W.squeeze())
                # ====================================================

                # Remove the following line of code, it will become `GCN`.
                W = self.recurrent_layers[i](W)  # Update GCN weights using GRU
                feature_list[j] = self.gnn_convs[i](g, feature_list[j], weight=W)
        return self.mlp(feature_list[-1])  # Apply MLP classifier to the last snapshot
