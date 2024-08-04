import functools

import dgl
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

from dataset.dataset_from_dgl import dataset_from_dgl
from models import Grace


def grace_test(
        dataset_name,
        lr,
        n_hidden,
        n_out_feats,
        n_layers,
        act_fn,
        der1,
        der2,
        dfr1,
        dfr2,
        temp,
        epochs,
        wd,
        split='random',
        gpu=0,
        verbose=0
):
    if gpu != -1 and torch.cuda.is_available():
        device = "cuda:{}".format(gpu)
    else:
        device = "cpu"
    act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[act_fn]

    drop_edge_rate_1 = der1
    drop_edge_rate_2 = der2
    drop_feature_rate_1 = dfr1
    drop_feature_rate_2 = dfr2

    dataset = dataset_from_dgl(dataset_name=dataset_name)
    graph = dataset[0]
    feats = graph.ndata.pop("feat")
    labels = graph.ndata.pop("label")
    train_mask = graph.ndata.pop("train_mask")
    test_mask = graph.ndata.pop("test_mask")
    n_in_feats = feats.shape[1]

    model = Grace(
        n_in_feats,
        n_hidden,
        n_out_feats,
        n_layers,
        act_fn,
        temp
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = aug(graph, feats, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, feats, drop_feature_rate_2, drop_edge_rate_2)

        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        feat1 = feat1.to(device)
        feat2 = feat2.to(device)

        loss = model(graph1, graph2, feat1, feat2)
        loss.backward()
        optimizer.step()
        if verbose > 0:
            print(f"Epoch={epoch:04d}, loss={loss.item():.4f}")

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    graph = graph.to(device)
    feats = feats.to(device)
    embeds = model.get_embedding(graph, feats)

    label_classification(
        embeds, labels, train_mask, test_mask, split=split
    )


# Data augmentation on graphs via edge dropping and feature masking
def aug(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.num_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = dgl.add_self_loop(ng)

    return ng, feat


def drop_feature(x, drop_prob):
    drop_mask = (
            torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
            < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
            print_statistics(statistics, f.__name__)
            return statistics

        return wrapper

    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool_)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f"(E) | {function_name}:", end=" ")
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]["mean"]
        std = statistics[key]["std"]
        print(f"{key}={mean:.4f}+-{std:.4f}", end="")
        if i != len(statistics.keys()) - 1:
            print(",", end=" ")
        else:
            print()


@repeat(3)
def label_classification(
        embeddings, y, train_mask, test_mask, split="random", ratio=0.1
):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories="auto").fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)

    X = normalize(X, norm="l2")

    if split == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=1 - ratio
        )
    elif split == "public":
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = Y[train_mask]
        y_test = Y[test_mask]

    logreg = LogisticRegression(solver="liblinear")
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(
        estimator=OneVsRestClassifier(logreg),
        param_grid=dict(estimator__C=c),
        n_jobs=8,
        cv=5,
        verbose=0,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {"F1Mi": micro, "F1Ma": macro}
