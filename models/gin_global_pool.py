import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling, Set2Set
from torch import nn


class GIN(nn.Module):
    def __init__(self, n_in_feats, n_hidden, n_out_feats, n_layers=5, p_drop=0.5, pooling="sum"):
        super().__init__()
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(n_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = _MLP(n_in_feats, n_hidden, n_hidden)
            else:
                mlp = _MLP(n_hidden, n_hidden, n_hidden)
            self.gin_layers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(n_hidden))
        # linear functions for graph sum pooling of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(n_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(n_in_feats, n_out_feats))
            else:
                self.linear_prediction.append(nn.Linear(n_hidden, n_out_feats))
        self.drop = nn.Dropout(p_drop)
        if pooling == "sum":
            self.pool = SumPooling()
        elif pooling == "mean":
            self.pool = AvgPooling()
        elif pooling == "max":
            self.pool = MaxPooling()
        elif pooling == "global_attention":
            gate_nn = nn.Linear(n_hidden, 1)
            self.pool = GlobalAttentionPooling(gate_nn)
        elif pooling == "set2set":
            self.pool = Set2Set(n_hidden, n_iters=2, n_layers=1)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.gin_layers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            if i > 0:
                pooled_h = self.pool(g, h)
                score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer


class _MLP(nn.Module):
    def __init__(self, n_in_feats, n_hidden, n_out_feats):
        super().__init__()
        # two-layer MLP
        self.fc1 = nn.Linear(n_in_feats, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_hidden, n_out_feats, bias=False)
        self.batch_norm = nn.BatchNorm1d(n_hidden)

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.fc1(h)))
        return self.fc2(h)