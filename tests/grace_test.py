import dgl
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from tqdm import trange

from dataset.dataset_from_dgl import dataset_from_dgl_built
from models import GRACE

drop_edge_rates = {
    "cora": [0.2, 0.4],
    "citeseer": [0.2, 0.0],
    "pubmed": [0.4, 0.1]
}

drop_feat_rates = {
    "cora": [0.3, 0.4],
    "citeseer": [0.3, 0.2],
    "pubmed": [0.0, 0.2]
}


def grace_test(
        dataset_name,
        lr,
        n_hidden,
        n_out_feats,
        act_fn,
        temp,
        n_epochs,
        n_layers=2,
        wd=1e-5,
        gpu=0
):
    if gpu != -1 and torch.cuda.is_available():
        device = "cuda:{}".format(gpu)
    else:
        device = "cpu"
    act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[act_fn]

    drop_edge_rate_1 = drop_edge_rates[dataset_name][0]
    drop_edge_rate_2 = drop_edge_rates[dataset_name][1]
    drop_feat_rate_1 = drop_feat_rates[dataset_name][0]
    drop_feat_rate_2 = drop_feat_rates[dataset_name][1]

    dataset = dataset_from_dgl_built(dataset_name=dataset_name)
    graph = dataset[0]
    feats = torch.FloatTensor(graph.ndata["feat"]).to(device)
    labels = torch.LongTensor(graph.ndata["label"]).to(device)
    n_in_feats = feats.shape[1]

    model = GRACE(
        n_in_feats,
        n_hidden,
        n_out_feats,
        n_layers,
        act_fn,
        temp
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = trange(n_epochs)
    for _ in epochs:
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = generating_views(graph, feats, drop_feat_rate_1, drop_edge_rate_1)
        graph2, feat2 = generating_views(graph, feats, drop_feat_rate_2, drop_edge_rate_2)
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        feat1 = feat1.to(device)
        feat2 = feat2.to(device)

        z1 = model(graph1, feat1)
        z2 = model(graph2, feat2)
        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()

        epochs.set_description("Train Loss {:.4f}".format(loss.item()))

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    graph = graph.to(device)
    feats = feats.to(device)
    embeds = model.get_embedding(graph, feats)
    acc = classification(embeds, labels)
    print("GRACE: dataset {} test accuracy {:.4f}".format(dataset_name, acc))
    torch.cuda.empty_cache()


# Data augmentation on graphs via edge dropping and feature masking
def generating_views(graph, x, feat_drop_rate, edge_mask_rate):
    edge_mask_idx = get_mask_edge_idx(graph, edge_mask_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    new_src = src[edge_mask_idx]
    new_dst = dst[edge_mask_idx]

    new_graph = dgl.graph((new_src, new_dst), num_nodes=graph.num_nodes())
    new_graph = dgl.add_self_loop(new_graph)
    dropped_feat = drop_feat(x, feat_drop_rate)

    return new_graph, dropped_feat


def get_mask_edge_idx(graph, edge_mask_rate):
    E = graph.num_edges()
    mask_rates = torch.FloatTensor(np.ones(E) * edge_mask_rate)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_feat(x, drop_prob):
    drop_mask = torch.rand(x.size(1), dtype=torch.float32, device=x.device) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def classification(embeddings, y):
    X = normalize(embeddings.detach().cpu().numpy(), norm="l2")
    Y = y.detach().cpu().numpy().reshape(-1, 1)
    Y = OneHotEncoder(categories="auto").fit_transform(Y).toarray().astype(np.bool_)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9)

    clf = GridSearchCV(estimator=OneVsRestClassifier(LogisticRegression(solver="liblinear")),
                       param_grid=dict(estimator__C=2.0 ** np.arange(-10, 10)),
                       n_jobs=8, cv=5, verbose=0)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)
    y_pred = np.eye(y_prob.shape[1])[np.argmax(y_prob, axis=1)].astype(np.bool_)
    acc = accuracy_score(y_test, y_pred)
    return acc
