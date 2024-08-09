import os
import pickle

import dgl
import torch
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, get_download_dir
from dgl.data.utils import _get_dgl_url, download


def dataset_from_dgl_built(dataset_name):
    if dataset_name == "cora":
        dataset = CoraGraphDataset()
    elif dataset_name == "citeseer":
        dataset = CiteseerGraphDataset()
    elif dataset_name == "pubmed":
        dataset = PubmedGraphDataset()
    else:
        raise ValueError("do not support")
    return dataset


def dataset_from_dgl_download(dataset_name):
    if dataset_name == "acm":
        url = "dataset/ACM3025.pkl"
        data_save_path = get_download_dir() + "/ACM3025.pkl"
        if not os.path.exists(data_save_path):
            download(_get_dgl_url(url), path=data_save_path)
        with open(data_save_path, "rb") as f:
            dataset = pickle.load(f)

        author_graph = dgl.from_scipy(dataset["PAP"])
        subject_graph = dgl.from_scipy(dataset["PLP"])
        graphs = [author_graph, subject_graph]
        # sparse matrix to dense matrix
        labels = torch.tensor(dataset["label"].todense(), dtype=torch.long).argmax(dim=1)
        feats = torch.tensor(dataset["feature"].todense(), dtype=torch.float)
        num_classes = labels.shape[-1]

        train_idx = torch.tensor(dataset["train_idx"], dtype=torch.long).squeeze(0)
        val_idx = torch.tensor(dataset["val_idx"], dtype=torch.long).squeeze(0)
        test_idx = torch.tensor(dataset["test_idx"], dtype=torch.long).squeeze(0)
        num_nodes = graphs[0].num_nodes()
        train_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, train_idx, True)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, val_idx, True)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, test_idx, True)
        return graphs, feats, labels, num_classes, train_mask, val_mask, test_mask
    else:
        raise ValueError("do not support")
