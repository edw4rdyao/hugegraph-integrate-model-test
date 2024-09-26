import copy
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange

from dataset.dataset_from_dgl import dataset_from_dgl_download
from models import HAN


def han_test(
        dataset_name,
        n_hidden,
        p_drop,
        n_heads,
        lr,
        n_epochs,
        wd,
        gpu=0,
        seed=0
):
    set_random_seed(seed)
    (
        graphs,
        feats,
        labels,
        n_classes,
        train_mask,
        val_mask,
        test_mask,
    ) = dataset_from_dgl_download(dataset_name)
    for g in graphs:
        print(g)
    print(feats.shape)
    print(labels.shape)
    device = "cuda:{}".format(gpu) if gpu != -1 and torch.cuda.is_available() else "cpu"

    feats = feats.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    graphs = [graph.to(device) for graph in graphs]

    model = HAN(
        n_meta_paths=len(graphs),
        n_in_feats=feats.shape[1],
        n_hidden=n_hidden,
        n_out_feats=n_classes,
        n_heads=n_heads,
        p_drop=p_drop,
    ).to(device)
    best_model = copy.deepcopy(model)
    best_acc = 0
    best_loss = np.inf

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = trange(n_epochs)
    for epoch in epochs:
        model.train()
        logits = model(graphs, feats)
        train_loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_preds = torch.argmax(logits[train_mask], dim=1)
        train_acc = accuracy_score(labels[train_mask].cpu(), train_preds.cpu())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(logits[val_mask], labels[val_mask])
            val_preds = torch.argmax(logits[val_mask], dim=1)
            val_acc = accuracy_score(labels[val_mask].cpu(), val_preds.cpu())

        epochs.set_description(
            "epoch {} | tloss {:.4f} | tacc: {:.4f} | vloss {:.4f} | vacc {:.4f}".format(
                epoch, train_loss.item(), train_acc, val_loss.item(), val_acc
            )
        )

        if val_acc > best_acc and val_loss < best_loss:
            best_acc = val_acc
            best_loss = train_loss
            best_model = copy.deepcopy(model)

    best_model.eval()
    logits = best_model(graphs, feats)
    test_preds = torch.argmax(logits[test_mask], dim=1)
    macro_f1 = f1_score(labels[test_mask].cpu(), test_preds.cpu(), average="macro")
    micro_f1 = f1_score(labels[test_mask].cpu(), test_preds.cpu(), average="micro")
    print("HAN: dataset {} test Macro F1 {:.4f}, Micro F1 {:.4f}".format(dataset_name, macro_f1, micro_f1))
    torch.cuda.empty_cache()


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
