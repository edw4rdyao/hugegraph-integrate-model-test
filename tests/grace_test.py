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

edges_removing_rates = {
    "cora": [0.2, 0.4],
    "citeseer": [0.2, 0.0],
    "pubmed": [0.4, 0.1]
}

feats_masking_rates = {
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
    device = "cuda:{}".format(gpu) if gpu != -1 and torch.cuda.is_available() else "cpu"
    act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[act_fn]

    edges_removing_rate_1 = edges_removing_rates[dataset_name][0]
    edges_removing_rate_2 = edges_removing_rates[dataset_name][1]
    feats_masking_rate_1 = feats_masking_rates[dataset_name][0]
    feats_masking_rate_2 = feats_masking_rates[dataset_name][1]

    dataset = dataset_from_dgl_built(dataset_name=dataset_name)
    graph = dataset[0]
    feats = torch.FloatTensor(graph.ndata["feat"]).to(device)
    labels = torch.LongTensor(graph.ndata["label"]).to(device)
    graph = graph.to(device)
    n_in_feats = feats.shape[1]

    model = GRACE(
        n_in_feats,
        n_hidden,
        n_out_feats,
        n_layers,
        act_fn,
        temp,
        edges_removing_rate_1,
        edges_removing_rate_2,
        feats_masking_rate_1,
        feats_masking_rate_2,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = trange(n_epochs)
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        loss = model(graph, feats)
        loss.backward()
        optimizer.step()

        epochs.set_description("epoch {} | train loss {:.4f}".format(epoch, loss.item()))
        torch.cuda.empty_cache()

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    graph = graph.to(device)
    feats = feats.to(device)
    embeds = model.get_embedding(graph, feats)
    acc = classification(embeds, labels)
    print("GRACE: dataset {} test accuracy {:.4f}".format(dataset_name, acc))


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
