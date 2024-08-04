import time

import dgl
import torch
import torch.nn.functional as F

from dataset.dataset_from_dgl import dataset_from_dgl
from models import Classifier, DGI


def dgi_test(
        dataset_name,
        gpu=0,
        self_loop=False,
        n_hidden=512,
        n_layers=2,
        p_drop=0.0,
        dgi_lr=1e-3,
        clf_lr=1e-2,
        n_dgi_epochs=300,
        n_clf_epochs=300,
        patience=30,
        weight_decay=0.0,
        verbose=0
):
    dataset = dataset_from_dgl(dataset_name)
    graph = dataset[0]
    feats = torch.FloatTensor(graph.ndata["feat"])
    labels = torch.LongTensor(graph.ndata["label"])
    train_mask = torch.BoolTensor(graph.ndata["train_mask"])
    test_mask = torch.BoolTensor(graph.ndata["test_mask"])
    n_in_feats = feats.shape[1]
    n_classes = dataset.num_classes

    if self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

    if gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(gpu)
        feats = feats.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        test_mask = test_mask.cuda()
        graph = graph.to(gpu)
    dgi = DGI(
        n_in_feats=n_in_feats,
        n_hidden=n_hidden,
        n_layers=n_layers,
        p_drop=p_drop
    )
    if cuda:
        dgi.cuda()
    dgi_optimizer = torch.optim.Adam(
        dgi.parameters(), lr=dgi_lr, weight_decay=weight_decay
    )
    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    mean = 0
    for epoch in range(n_dgi_epochs):
        dgi.train()
        if epoch >= 3:
            t0 = time.time()

        dgi_optimizer.zero_grad()
        loss = dgi(graph, feats)
        loss.backward()
        dgi_optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), "tmp_best_dgi.pkl")
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print("Early stopping!")
            break

        if epoch >= 3 and verbose > 0:
            mean = (mean * (epoch - 3) + (time.time() - t0)) / (epoch - 2)
            print("Epoch {:04d} | Time(s) {:.4f} | Loss {:.4f}".format(epoch, mean, loss.item()))

    # create classifier model
    clf = Classifier(n_hidden, n_classes)
    if cuda:
        clf.cuda()

    classifier_optimizer = torch.optim.Adam(
        clf.parameters(),
        lr=clf_lr,
        weight_decay=weight_decay,
    )

    # train classifier
    print("Loading {}th epoch".format(best_t))
    dgi.load_state_dict(torch.load("tmp_best_dgi.pkl"))
    embeds = dgi.encoder(graph, feats, corrupt=False)
    embeds = embeds.detach()
    mean = 0
    for epoch in range(n_clf_epochs):
        clf.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = clf(embeds)
        loss = F.nll_loss(preds[train_mask], labels[train_mask])
        loss.backward()
        classifier_optimizer.step()

        if epoch >= 3 and verbose > 0:
            mean = (mean * (epoch - 3) + (time.time() - t0)) / (epoch - 2)
            print("Epoch {:04d} | Time(s) {:.4f} | Loss {:.4f}".format(epoch, mean, loss.item()))

    clf.eval()
    with torch.no_grad():
        logits = clf(embeds)[test_mask]
        _, predicted = torch.max(logits, dim=1)
        correct = (predicted == labels[test_mask]).sum().item()
        acc = correct / test_mask.sum().item()

    print("dgi: dataset {} test accuracy {:.4f}".format(dataset_name, acc))
