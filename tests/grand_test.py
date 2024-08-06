import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dataset.dataset_from_dgl import dataset_from_dgl
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
        early_stopping,
        n_epochs=2000,
        bn=False,
        wd=5e-4,
        p_drop_node=0.5,
        gpu=0,
        n_hidden=32,
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
    )
    model = model.to(device)
    graph = graph.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_best = np.inf
    acc_best = 0

    for epoch in range(n_epochs):
        model.train()

        loss_sup = 0
        logits = model(graph, feats)

        # calculate supervised loss
        for k in range(sample):
            loss_sup += F.nll_loss(logits[k][train_idx], labels[train_idx])
        loss_sup = loss_sup / sample

        # calculate consistency loss
        loss_consis = model.consis_loss(logits, temp, lam)

        loss_train = loss_sup + loss_consis
        acc_train = torch.sum(
            logits[0][train_idx].argmax(dim=1) == labels[train_idx]
        ).item() / len(train_idx)

        # backward
        opt.zero_grad()
        loss_train.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model.inference(graph, feats)

            loss_val = F.nll_loss(val_logits[val_idx], labels[val_idx])
            acc_val = torch.sum(
                val_logits[val_idx].argmax(dim=1) == labels[val_idx]
            ).item() / len(val_idx)

            # Print out performance
            print(
                "In epoch {}, Train Acc: {:.4f} | Train Loss: {:.4f} ,Val Acc: {:.4f} | Val Loss: {:.4f}".format(
                    epoch,
                    acc_train,
                    loss_train.item(),
                    acc_val,
                    loss_val.item(),
                )
            )

            # set early stopping counter
            if loss_val < loss_best or acc_val > acc_best:
                if loss_val < loss_best:
                    best_epoch = epoch
                    torch.save(model.state_dict(), dataset_name + ".pkl")
                no_improvement = 0
                loss_best = min(loss_val, loss_best)
                acc_best = max(acc_val, acc_best)
            else:
                no_improvement += 1
                if no_improvement == early_stopping:
                    print("Early stopping.")
                    break

    print("Optimization Finished!")

    print("Loading {}th epoch".format(best_epoch))
    model.load_state_dict(torch.load(dataset_name + ".pkl"))

    model.eval()

    test_logits = model.inference(graph, feats)
    test_acc = torch.sum(
        test_logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)

    print("Test Acc: {:.4f}".format(test_acc))
