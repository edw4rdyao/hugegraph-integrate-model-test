import argparse
import warnings

import dgl
import torch as th
import torch.nn as nn
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

"""
Code adapted from https://github.com/CRIPAC-DIG/GRACE
Linear evaluation on learned node embeddings
"""

import functools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

from models import Grace

warnings.filterwarnings("ignore")


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
    ng = ng.add_self_loop()

    return ng, feat


def drop_feature(x, drop_prob):
    drop_mask = (
            th.empty((x.size(1),), dtype=th.float32, device=x.device).uniform_(0, 1)
            < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def load(name):
    if name == "cora":
        dataset = CoraGraphDataset()
    elif name == "citeseer":
        dataset = CiteseerGraphDataset()
    elif name == "pubmed":
        dataset = PubmedGraphDataset()

    graph = dataset[0]

    train_mask = graph.ndata.pop("train_mask")
    test_mask = graph.ndata.pop("test_mask")

    feat = graph.ndata.pop("feat")
    labels = graph.ndata.pop("label")

    return graph, feat, labels, train_mask, test_mask


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


def count_parameters(model):
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )


def grace_test(test_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="cora")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training periods.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature.")
    parser.add_argument("--act_fn", type=str, default="relu")
    parser.add_argument("--hid_dim", type=int, default=256, help="Hidden layer dim.")
    parser.add_argument("--out_dim", type=int, default=256, help="Output layer dim.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers.")
    parser.add_argument("--der1", type=float, default=0.2, help="Drop edge ratio of the 1st augmentation.")
    parser.add_argument("--der2", type=float, default=0.2, help="Drop edge ratio of the 2nd augmentation.", )
    parser.add_argument("--dfr1", type=float, default=0.2, help="Drop feature ratio of the 1st augmentation.")
    parser.add_argument("--dfr2", type=float, default=0.2, help="Drop feature ratio of the 2nd augmentation.")

    args = parser.parse_args(test_args)

    if args.gpu != -1 and th.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    # Step 1: Load hyperparameters =================================================================== #
    lr = args.lr
    hid_dim = args.hid_dim
    out_dim = args.out_dim

    num_layers = args.num_layers
    act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[args.act_fn]

    drop_edge_rate_1 = args.der1
    drop_edge_rate_2 = args.der2
    drop_feature_rate_1 = args.dfr1
    drop_feature_rate_2 = args.dfr2

    temp = args.temp
    epochs = args.epochs
    wd = args.wd

    # Step 2: Prepare data =================================================================== #
    graph, feat, labels, train_mask, test_mask = load(args.dataname)
    in_dim = feat.shape[1]

    # Step 3: Create model =================================================================== #
    model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
    model = model.to(args.device)
    print(f"# params: {count_parameters(model)}")

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Step 4: Training =======================================================================
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = aug(graph, feat, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, feat, drop_feature_rate_2, drop_edge_rate_2)

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        loss = model(graph1, graph2, feat1, feat2)
        loss.backward()
        optimizer.step()

        print(f"Epoch={epoch:03d}, loss={loss.item():.4f}")

    # Step 5: Linear evaluation ============================================================== #
    print("=== Final ===")

    graph = graph.add_self_loop()
    graph = graph.to(args.device)
    feat = feat.to(args.device)
    embeds = model.get_embedding(graph, feat)

    """Evaluation Embeddings  """
    label_classification(
        embeds, labels, train_mask, test_mask, split=args.split
    )
