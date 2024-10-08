"""
DiffPool (Differentiable Pooling)

References
----------
Paper: https://arxiv.org/abs/1806.08804
Author's code: https://github.com/RexYing/diffpool
Ref DGL code: https://github.com/dmlc/dgl/tree/master/examples/pytorch/diffpool
"""

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from torch.nn import init


class DiffPool(nn.Module):
    def __init__(
            self,
            n_input_feat,
            n_hidden,
            n_embedding,
            n_out,
            activation,
            n_layers,
            dropout,
            n_pooling,
            linkpred,
            batch_size,
            aggregator_type,
            n_assign,
            pool_ratio,
            cat=False,
    ):
        super(DiffPool, self).__init__()
        self.link_pred = linkpred
        self.concat = cat
        self.n_pooling = n_pooling
        self.batch_size = batch_size
        self.link_pred_loss = []
        self.entropy_loss = []

        # list of GNN modules before the first diffpool operation
        self.gc_before_pool = nn.ModuleList()
        self.diffpool_layers = nn.ModuleList()

        # list of GNN modules, each list after one diffpool operation
        self.gc_after_pool = nn.ModuleList()
        self.assign_dim = n_assign
        self.bn = True
        self.num_aggs = 1

        # constructing layers before diffpool
        assert n_layers >= 3, "n_layers too few"
        self.gc_before_pool.append(
            SAGEConv(
                n_input_feat,
                n_hidden,
                aggregator_type,
                feat_drop=dropout,
                activation=activation,
                bias=self.bn
            )
        )
        for _ in range(n_layers - 2):
            self.gc_before_pool.append(
                SAGEConv(
                    n_hidden,
                    n_hidden,
                    aggregator_type,
                    feat_drop=dropout,
                    activation=activation,
                    bias=self.bn
                )
            )
        self.gc_before_pool.append(
            SAGEConv(
                n_hidden,
                n_embedding,
                aggregator_type,
                feat_drop=dropout,
                activation=None,
            )
        )

        assign_dims = []
        assign_dims.append(self.assign_dim)
        if self.concat:
            # diffpool layer receive pool_embedding_dim node feature tensor
            # and return pool_embedding_dim node embedding
            pool_embedding_dim = n_hidden * (n_layers - 1) + n_embedding
        else:
            pool_embedding_dim = n_embedding

        self.first_diffpool_layer = DiffPoolBatchedGraphLayer(
            pool_embedding_dim,
            self.assign_dim,
            n_hidden,
            activation,
            dropout,
            aggregator_type,
            self.link_pred,
        )

        gc_after_per_pool = nn.ModuleList()
        for _ in range(n_layers - 1):
            gc_after_per_pool.append(BatchedGraphSAGE(n_hidden, n_hidden))
        gc_after_per_pool.append(BatchedGraphSAGE(n_hidden, n_embedding))
        self.gc_after_pool.append(gc_after_per_pool)

        self.assign_dim = int(self.assign_dim * pool_ratio)
        # each pooling module
        for _ in range(n_pooling - 1):
            self.diffpool_layers.append(
                BatchedDiffPool(
                    pool_embedding_dim,
                    self.assign_dim,
                    n_hidden,
                    self.link_pred,
                )
            )
            gc_after_per_pool = nn.ModuleList()
            for _ in range(n_layers - 1):
                gc_after_per_pool.append(
                    BatchedGraphSAGE(n_hidden, n_hidden)
                )
            gc_after_per_pool.append(
                BatchedGraphSAGE(n_hidden, n_embedding)
            )
            self.gc_after_pool.append(gc_after_per_pool)
            assign_dims.append(self.assign_dim)
            self.assign_dim = int(self.assign_dim * pool_ratio)

        # predicting layer
        if self.concat:
            self.pred_input_dim = (
                    pool_embedding_dim * self.num_aggs * (n_pooling + 1)
            )
        else:
            self.pred_input_dim = n_embedding * self.num_aggs
        self.pred_layer = nn.Linear(self.pred_input_dim, n_out)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def gcn_forward(self, g, h, gc_layers, cat=False):
        """
        Return gc_layer embedding cat.
        """
        block_readout = []
        for gc_layer in gc_layers[:-1]:
            h = gc_layer(g, h)
            block_readout.append(h)
        h = gc_layers[-1](g, h)
        block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=1)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def gcn_forward_tensorized(self, h, adj, gc_layers, cat=False):
        block_readout = []
        for gc_layer in gc_layers:
            h = gc_layer(h, adj)
            block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=2)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def forward(self, g, feat):
        self.link_pred_loss = []
        self.entropy_loss = []
        h = feat
        # node feature for assignment matrix computation is the same as the
        # original node feature

        out_all = []

        # we use GCN blocks to get an embedding first
        g_embedding = self.gcn_forward(g, h, self.gc_before_pool, self.concat)

        g.ndata["h"] = g_embedding

        readout = dgl.sum_nodes(g, "h")
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = dgl.max_nodes(g, "h")
            out_all.append(readout)

        adj, h = self.first_diffpool_layer(g, g_embedding)
        node_per_pool_graph = int(adj.size()[0] / len(g.batch_num_nodes()))

        h, adj = batch2tensor(adj, h, node_per_pool_graph)
        h = self.gcn_forward_tensorized(
            h, adj, self.gc_after_pool[0], self.concat
        )
        readout = torch.sum(h, dim=1)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout, _ = torch.max(h, dim=1)
            out_all.append(readout)

        for i, diffpool_layer in enumerate(self.diffpool_layers):
            h, adj = diffpool_layer(h, adj)
            h = self.gcn_forward_tensorized(
                h, adj, self.gc_after_pool[i + 1], self.concat
            )
            readout = torch.sum(h, dim=1)
            out_all.append(readout)
            if self.num_aggs == 2:
                readout, _ = torch.max(h, dim=1)
                out_all.append(readout)
        if self.concat or self.num_aggs > 1:
            final_readout = torch.cat(out_all, dim=1)
        else:
            final_readout = readout
        ypred = self.pred_layer(final_readout)
        return ypred

    def loss(self, pred, label):
        """
        loss function
        """
        # softmax + CE
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        for key, value in self.first_diffpool_layer.loss_log.items():
            loss += value
        for diffpool_layer in self.diffpool_layers:
            for key, value in diffpool_layer.loss_log.items():
                loss += value
        return loss


def batch2tensor(batch_adj, batch_feat, node_per_pool_graph):
    """
    transform a batched graph to batched adjacency tensor and node feature tensor
    """
    batch_size = int(batch_adj.size()[0] / node_per_pool_graph)
    adj_list = []
    feat_list = []
    for i in range(batch_size):
        start = i * node_per_pool_graph
        end = (i + 1) * node_per_pool_graph
        adj_list.append(batch_adj[start:end, start:end])
        feat_list.append(batch_feat[start:end, :])
    adj_list = list(map(lambda x: torch.unsqueeze(x, 0), adj_list))
    feat_list = list(map(lambda x: torch.unsqueeze(x, 0), feat_list))
    adj = torch.cat(adj_list, dim=0)
    feat = torch.cat(feat_list, dim=0)

    return feat, adj


def masked_softmax(matrix, mask, dim=-1, memory_efficient=True, mask_fill_value=-1e32):
    """
    masked_softmax for dgl batch graph
    code snippet contributed by AllenNLP (https://github.com/allenai/allennlp)
    """
    if mask is None:
        result = torch.nn.functional.softmax(matrix, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < matrix.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = torch.nn.functional.softmax(matrix * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_matrix = matrix.masked_fill(
                (1 - mask).byte(), mask_fill_value
            )
            result = torch.nn.functional.softmax(masked_matrix, dim=dim)
    return result


class BatchedGraphSAGE(nn.Module):
    def __init__(
            self, infeat, outfeat, use_bn=True, mean=False, add_self=False
    ):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)

        nn.init.xavier_uniform_(
            self.W.weight, gain=nn.init.calculate_gain("relu")
        )

    def forward(self, x, adj):
        num_node_per_graph = adj.size(1)
        if self.use_bn and not hasattr(self, "bn"):
            self.bn = nn.BatchNorm1d(num_node_per_graph).to(adj.device)

        if self.add_self:
            adj = adj + torch.eye(num_node_per_graph).to(adj.device)

        if self.mean:
            adj = adj / adj.sum(-1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            h_k = self.bn(h_k)
        return h_k

    def __repr__(self):
        if self.use_bn:
            return "BN" + super(BatchedGraphSAGE, self).__repr__()
        else:
            return super(BatchedGraphSAGE, self).__repr__()


class DiffPoolAssignment(nn.Module):
    def __init__(self, nfeat, nnext):
        super().__init__()
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, use_bn=True)

    def forward(self, x, adj, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l = F.softmax(s_l_init, dim=-1)
        return s_l


class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, link_pred=False, entropy=True):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.log = {}
        self.link_pred_layer = LinkPredLoss()
        self.embed = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        self.assign = DiffPoolAssignment(nfeat, nnext)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        if link_pred:
            self.reg_loss.append(LinkPredLoss())
        if entropy:
            self.reg_loss.append(EntropyLoss())

    def forward(self, x, adj, log=False):
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)
        if log:
            self.log["s"] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, anext, s_l)
        if log:
            self.log["a"] = anext.cpu().numpy()
        return xnext, anext


class DiffPoolBatchedGraphLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            assign_dim,
            output_feat_dim,
            activation,
            dropout,
            aggregator_type,
            link_pred,
    ):
        super(DiffPoolBatchedGraphLayer, self).__init__()
        self.embedding_dim = input_dim
        self.assign_dim = assign_dim
        self.hidden_dim = output_feat_dim
        self.link_pred = link_pred
        self.feat_gc = SAGEConv(
            input_dim,
            output_feat_dim,
            aggregator_type,
            feat_drop=dropout,
            activation=activation,
        )
        self.pool_gc = SAGEConv(
            input_dim,
            assign_dim,
            aggregator_type,
            feat_drop=dropout,
            activation=activation,
        )
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        self.reg_loss.append(EntropyLoss())

    def forward(self, g, h):
        feat = self.feat_gc(
            g, h
        )  # size = (sum_N, F_out), sum_N is num of nodes in this batch
        device = feat.device
        assign_tensor = self.pool_gc(
            g, h
        )  # size = (sum_N, N_a), N_a is num of nodes in pooled graph.
        assign_tensor = F.softmax(assign_tensor, dim=1)
        assign_tensor = torch.split(assign_tensor, g.batch_num_nodes().tolist())
        assign_tensor = torch.block_diag(
            *assign_tensor
        )  # size = (sum_N, batch_size * N_a)

        h = torch.matmul(torch.t(assign_tensor), feat)
        adj = g.adj_external(transpose=True, ctx=device)
        adj_new = torch.sparse.mm(adj, assign_tensor)
        adj_new = torch.mm(torch.t(assign_tensor), adj_new)

        if self.link_pred:
            current_lp_loss = torch.norm(
                adj.to_dense() - torch.mm(assign_tensor, torch.t(assign_tensor))
            ) / np.power(g.num_nodes(), 2)
            self.loss_log["LinkPredLoss"] = current_lp_loss

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, adj_new, assign_tensor)

        return adj_new, h


class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, adj, anext, s_l):
        entropy = (
            (torch.distributions.Categorical(probs=s_l).entropy())
            .sum(-1)
            .mean(-1)
        )
        assert not torch.isnan(entropy)
        return entropy


class LinkPredLoss(nn.Module):
    def forward(self, adj, anext, s_l):
        link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(
            dim=(1, 2)
        )
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        return link_pred_loss.mean()
