import copy

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import trange

from dataset.dataset_from_dgl import dataset_from_dgl
from models.jknet import JKNet


def jknet_test(
        dataset_name,
        n_hidden=32,
        n_layers=2,
        dropout=0.5,
        epochs=200,
        mode="cat",
        lr=0.005,
        lamb=0.0005,
        gpu=0,
):
    if gpu != -1 and torch.cuda.is_available():
        device = "cuda:{}".format(gpu)
    else:
        device = "cpu"
    dataset = dataset_from_dgl(dataset_name=dataset_name)
    graph = dataset[0].to(device)
    n_classes = dataset.num_classes
    labels = graph.ndata.pop("label").to(device).long()
    feats = graph.ndata.pop("feat").to(device)
    n_in_feats = feats.shape[1]
    n_nodes = graph.num_nodes()
    idx = torch.arange(n_nodes).to(device)
    train_idx, test_idx = train_test_split(idx, test_size=0.2)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25)

    model = JKNet(
        n_in_feats=n_in_feats,
        n_hidden=n_hidden,
        n_out_feats=n_classes,
        n_layers=n_layers,
        mode=mode,
        dropout=dropout,
    ).to(device)

    best_model = copy.deepcopy(model)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=lamb)
    acc = 0
    epochs = trange(epochs)
    for _ in epochs:
        model.train()
        logits = model(graph, feats)

        train_loss = loss_fn(logits[train_idx], labels[train_idx])
        train_preds = logits[train_idx].argmax(dim=1)
        train_acc = accuracy_score(labels[train_idx].cpu(), train_preds.cpu())

        opt.zero_grad()
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(logits[val_idx], labels[val_idx])
            valid_preds = logits[val_idx].argmax(dim=1)
            valid_acc = accuracy_score(labels[val_idx].cpu(), valid_preds.cpu())

        epochs.set_description(
            "Train Acc {:.4f} | Train Loss {:.4f} | Val Acc {:.4f} | Val loss {:.4f}".format(
                train_acc, train_loss.item(), valid_acc, valid_loss.item()
            )
        )

        if valid_acc > acc:
            acc = valid_acc
            best_model = copy.deepcopy(model)

    best_model.eval()
    logits = best_model(graph, feats)
    test_preds = logits[test_idx].argmax(dim=1)
    test_acc = accuracy_score(labels[test_idx].cpu(), test_preds.cpu())

    print("JKNet: dataset {} test accuracy {:.4f}".format(dataset_name, test_acc))
    torch.cuda.empty_cache()
