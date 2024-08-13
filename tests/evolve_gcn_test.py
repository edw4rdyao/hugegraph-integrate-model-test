import dgl
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import trange

from dataset.dataset_from_dgl import dataset_from_dgl_download
from models.evolve_gcn import EvolveGCNH, EvolveGCNO


def evolve_gcn_test(
        dataset_name,
        model_name="EvolveGCN-O",
        n_hidden=256,
        n_layers=2,
        lr=0.001,
        n_hist_steps=5,
        loss_class_weight="0.35,0.65",
        eval_class_id=1,
        n_epochs=1000,
        early_stopping=200,
        gpu="0",
):
    device = "cuda:{}".format(gpu) if gpu != -1 and torch.cuda.is_available() else "cpu"
    g, node_mask_by_time, num_classes = dataset_from_dgl_download(dataset_name)

    subgraphs = []
    labeled_node_masks = []
    for i in range(len(node_mask_by_time)):
        # we add self loop edge when we construct full graph, not here
        node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i])
        subgraphs.append(node_subgraph.to(device))
        valid_node_mask = node_subgraph.ndata["label"] >= 0
        labeled_node_masks.append(valid_node_mask)

    if model_name == "EvolveGCN-O":
        model = EvolveGCNO(
            in_feats=int(g.ndata["feat"].shape[1]),
            n_hidden=n_hidden,
            num_layers=n_layers,
        )
    elif model_name == "EvolveGCN-H":
        model = EvolveGCNH(
            in_feats=int(g.ndata["feat"].shape[1]),
            num_layers=n_layers
        )
    else:
        return NotImplementedError("Unsupported model {}".format(model_name))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_class_weight = torch.Tensor(
        [float(w) for w in loss_class_weight.split(",")]
    ).to(device)

    # split train, valid, test(0-30,31-35,36-48)
    # train/valid/test split follow the paper.
    train_max_index = 30
    valid_max_index = 35
    test_max_index = 48
    time_window_size = n_hist_steps

    best_valid_f1 = 0.0
    best_f1_epoch = 0
    best_test_f1 = 0.0

    epochs = trange(n_epochs)
    for epoch in epochs:
        model.train()
        all_preds = []
        all_labels = []
        for i in range(time_window_size, train_max_index + 1):
            graph_list = subgraphs[i - time_window_size: i + 1]
            probs = model(graph_list)[labeled_node_masks[i]]
            labels = subgraphs[i].ndata["label"][labeled_node_masks[i]].long()
            loss = F.cross_entropy(probs, labels, weight=loss_class_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        train_f1 = f1_score(all_labels, all_preds, average="binary", pos_label=eval_class_id)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i in range(train_max_index + 1, valid_max_index + 1):
                graph_list = subgraphs[i - time_window_size: i + 1]
                probs = model(graph_list)
                preds = probs[labeled_node_masks[i]].argmax(dim=1)
                labels = subgraphs[i].ndata["label"][labeled_node_masks[i]].long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        valid_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=eval_class_id)
        epochs.set_description(
            "Train f1 {:.4f} | Val f1 {:.4f}".format(train_f1, valid_f1)
        )

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_f1_epoch = epoch
            all_preds = []
            all_labels = []

            for i in range(valid_max_index + 1, test_max_index + 1):
                graph_list = subgraphs[i - time_window_size: i + 1]
                probs = model(graph_list)
                preds = probs[labeled_node_masks[i]].argmax(dim=1)
                labels = subgraphs[i].ndata["label"][labeled_node_masks[i]].long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            test_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=eval_class_id)
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1

        if epoch - best_f1_epoch >= early_stopping:
            print(
                "Early stopping at epoch {} best Test f1 {:.4f}".format(epoch, best_test_f1)
            )
            break

    print("{}: dataset {} test f1 {:.4f}".format(model_name, dataset_name, best_test_f1))
    torch.cuda.empty_cache()
