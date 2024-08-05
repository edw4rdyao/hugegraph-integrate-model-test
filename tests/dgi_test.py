import dgl
import torch
import torch.nn.functional as F
import  copy
from tqdm import trange

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
        patience=40,
        weight_decay=0
):
    if gpu != -1 and torch.cuda.is_available():
        device = "cuda:{}".format(gpu)
    else:
        device = "cpu"

    dataset = dataset_from_dgl(dataset_name)
    graph = dataset[0]
    feats = torch.FloatTensor(graph.ndata["feat"]).to(device)
    labels = torch.LongTensor(graph.ndata["label"]).to(device)
    train_mask = torch.BoolTensor(graph.ndata["train_mask"]).to(device)
    test_mask = torch.BoolTensor(graph.ndata["test_mask"]).to(device)
    n_in_feats = feats.shape[1]
    n_classes = dataset.num_classes

    if self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
    graph = graph.to(device)

    dgi = DGI(
        n_in_feats=n_in_feats,
        n_hidden=n_hidden,
        n_layers=n_layers,
        p_drop=p_drop
    ).to(device)
    best_dgi_model = copy.deepcopy(dgi)

    dgi_optimizer = torch.optim.Adam(
        dgi.parameters(), lr=dgi_lr, weight_decay=weight_decay
    )
    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    epochs = trange(n_dgi_epochs)
    for _ in epochs:
        dgi.train()

        dgi_optimizer.zero_grad()
        loss = dgi(graph, feats)
        loss.backward()
        dgi_optimizer.step()

        if loss < best:
            best = loss
            cnt_wait = 0
            best_dgi_model = copy.deepcopy(dgi)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print("Early stopping!")
            break

        epochs.set_description("Train Loss {:.4f}".format(loss.item()))

    best_dgi_model.eval()
    embeds = best_dgi_model.encoder(graph, feats, corrupt=False)
    embeds = embeds.detach()

    # classifier model
    clf = Classifier(n_hidden, n_classes).to(device)

    classifier_optimizer = torch.optim.Adam(
        clf.parameters(),
        lr=clf_lr,
        weight_decay=weight_decay,
    )
    # train classifier
    epochs = trange(n_clf_epochs)
    for _ in epochs:
        clf.train()

        classifier_optimizer.zero_grad()
        preds = clf(embeds)
        loss = F.nll_loss(preds[train_mask], labels[train_mask])
        loss.backward()
        classifier_optimizer.step()

        epochs.set_description("Train Loss {:.4f}".format(loss.item()))

    clf.eval()
    with torch.no_grad():
        logits = clf(embeds)[test_mask]
        _, predicted = torch.max(logits, dim=1)
        correct = (predicted == labels[test_mask]).sum().item()
        acc = correct / test_mask.sum().item()

    print("DGI: dataset {} test accuracy {:.4f}".format(dataset_name, acc))
