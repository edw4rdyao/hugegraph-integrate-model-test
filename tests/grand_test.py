import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import trange

from dataset.dataset_from_dgl import dataset_from_dgl_built
from models.grand import GRAND


def grand_test(
        dataset_name,
        sample,
        order,
        p_drop_input,
        p_drop_hidden,
        lr,
        temp,
        lam,
        early_stopping=200,
        n_epochs=2000,
        bn=False,
        wd=5e-4,
        p_drop_node=0.5,
        gpu=0,
        n_hidden=32,
):
    device = "cuda:{}".format(gpu) if gpu != -1 and torch.cuda.is_available() else "cpu"
    dataset = dataset_from_dgl_built(dataset_name=dataset_name)
    graph = dataset[0].to(device)
    n_classes = dataset.num_classes
    labels = graph.ndata.pop("label").to(device).long()
    feats = graph.ndata.pop("feat").to(device)
    n_in_feats = feats.shape[1]
    train_mask = graph.ndata.pop("train_mask")
    val_mask = graph.ndata.pop("val_mask")
    test_mask = graph.ndata.pop("test_mask")
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze().to(device)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

    model = GRAND(
        n_in_feats,
        n_hidden,
        n_classes,
        sample,
        order,
        p_drop_node,
        p_drop_input,
        p_drop_hidden,
        bn,
    ).to(device)
    best_model = copy.deepcopy(model)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    cnt_wait = 0
    best_loss = np.inf
    best_acc = 0

    epochs = trange(n_epochs)
    for epoch in epochs:
        model.train()
        logits = model(graph, feats)
        # calculate supervised loss
        loss_sup = 0
        for k in range(sample):
            loss_sup += F.nll_loss(logits[k][train_idx], labels[train_idx])
        loss_sup = loss_sup / sample
        # calculate consistency loss
        loss_consis = model.consis_loss(logits, temp, lam)
        train_loss = loss_sup + loss_consis

        # backward
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model.inference(graph, feats)
            val_loss = F.nll_loss(val_logits[val_idx], labels[val_idx])
            val_preds = val_logits[val_idx].argmax(dim=1)
            val_acc = accuracy_score(labels[val_idx].cpu(), val_preds.cpu())

            epochs.set_description(
                "epoch {:04d} | tloss {:.4f} | vacc {:.4f} | vloss {:.4f}".format(
                    epoch, train_loss.item(), val_acc, val_loss.item(),
                )
            )

            # set early stopping counter
            if val_loss < best_loss:
                cnt_wait = 0
                best_loss = val_loss
                best_model = copy.deepcopy(model)
            else:
                cnt_wait += 1
                if cnt_wait == early_stopping:
                    print("early stopping in epoch {}".format(epoch))
                    break

    best_model.eval()
    test_logits = best_model.inference(graph, feats)
    test_preds = torch.argmax(test_logits[test_idx], dim=1)
    test_acc = accuracy_score(labels[test_idx].cpu(), test_preds.cpu())

    print("GRAND: dataset {} test accuracy {:.4f}".format(dataset_name, test_acc))
    torch.cuda.empty_cache()
