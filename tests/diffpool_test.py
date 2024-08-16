import copy
import random

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from dgl.data import tu
from tqdm import trange

from models.diffpool import DiffPool


def diffpool_test(
        dataset_name,
        train_ratio=0.7,
        test_ratio=0.1,
        batch_size=20,
        pool_ratio=0.1,
        gc_per_block=3,
        dropout=0.0,
        num_pool=1,
        linkpred=False,
        n_epochs=1000,
        early_stopping=200,
        clip=2.0,
        gpu=0
):
    dataset = tu.LegacyTUDataset(name=dataset_name)
    device = "cuda:{}".format(gpu) if gpu != -1 and torch.cuda.is_available() else "cpu"

    train_size, test_size = int(train_ratio * len(dataset)), int(test_ratio * len(dataset))
    val_size = int(len(dataset) - train_size - test_size)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_size, val_size, test_size)
    )
    train_dataloader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = dgl.dataloading.GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    n_feat_in, n_labels, max_n_nodes = dataset.statistics()

    print("dataset feature dimension is", n_feat_in)
    print("dataset classes num is", n_labels)
    print("the max num node is", max_n_nodes)
    print("number of graphs is", len(dataset))

    n_hidden = 64
    n_embedding = 64
    # calculate assignment dimension: pool_ratio * the largest graph's maximum
    # number of nodes  in the dataset
    n_assign = int(max_n_nodes * pool_ratio)
    print("model hidden dim is", n_hidden)
    print("model embedding dim for graph instance embedding", n_embedding)
    print("initial batched pool graph dim is", n_assign)

    activation = F.relu
    model = DiffPool(
        n_feat_in,
        n_hidden,
        n_embedding,
        n_labels,
        activation,
        gc_per_block,
        dropout,
        num_pool,
        linkpred,
        batch_size,
        "mean",
        n_assign,
        pool_ratio,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    best_val_acc = 0.0
    best_model = copy.deepcopy(model)
    cnt_wait = 0

    epochs = trange(n_epochs)
    for epoch in epochs:
        model.train()
        correct = 0
        total = 0
        for batch_graph, graph_labels in train_dataloader:
            total += graph_labels.size()[0]

            feats = torch.FloatTensor(batch_graph.ndata["feat"]).to(device)
            labels = torch.LongTensor(graph_labels.long()).to(device)
            batch_graph = batch_graph.to(device)

            model.zero_grad()
            probs = model(batch_graph, feats)
            preds = torch.argmax(probs, dim=1)
            correct += (preds == labels).sum().item()
            loss = model.loss(probs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        train_acc = correct / total

        val_acc = evaluate(val_dataloader, model, device)
        # val_acc = evaluate(test_dataloader, model, device)

        if best_val_acc <= val_acc <= train_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            cnt_wait = 0
        else:
            cnt_wait += 1
            if cnt_wait >= early_stopping:
                print("early stopping in epoch {}".format(epoch))
                break
        epochs.set_description(
            "epoch {:04d} | tacc {:.4f} | vacc {:.4f} | best_vacc {:.4f}".format(
                epoch, train_acc, val_acc, best_val_acc
            )
        )
        torch.cuda.empty_cache()
    test_acc = evaluate(test_dataloader, best_model, device)
    print("DiffPool: dataset {} test acc {:.4f}".format(dataset_name, test_acc))


def evaluate(dataloader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_graph, graph_labels in dataloader:
            total += graph_labels.size()[0]

            feats = torch.FloatTensor(batch_graph.ndata["feat"].float()).to(device)
            labels = torch.LongTensor(graph_labels.long()).to(device)
            batch_graph = batch_graph.to(device)

            probs = model(batch_graph, feats)
            preds = torch.argmax(probs, dim=1)
            correct += (preds == labels).sum().item()
    acc = correct / total
    return acc


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

