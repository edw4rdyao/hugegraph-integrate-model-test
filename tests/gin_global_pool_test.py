import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

from models.gin_global_pool import GIN


def gin_global_pool_test(dataset_name="MUTAG", n_epochs=300, pooling="sum"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load and split dataset
    dataset = GINDataset(dataset_name, self_loop=True)
    labels = [l for _, l in dataset]
    train_idx, val_idx = split_fold10(labels)
    # create dataloader
    train_loader = GraphDataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_idx),
        batch_size=128,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = GraphDataLoader(
        dataset,
        sampler=SubsetRandomSampler(val_idx),
        batch_size=128,
        pin_memory=torch.cuda.is_available(),
    )

    # create GIN model
    in_size = dataset.dim_nfeats
    out_size = dataset.gclasses
    model = GIN(in_size, 16, out_size, pooling=pooling).to(device)

    # model training/validating
    print("Training...")
    train(train_loader, val_loader, device, model, n_epochs=n_epochs)

def split_fold10(labels, fold_idx=0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[fold_idx]
    return train_idx, valid_idx


def evaluate(dataloader, device, model):
    model.eval()
    total = 0
    total_correct = 0
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feat = batched_graph.ndata["attr"]
        total += len(labels)
        logits = model(batched_graph, feat)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
    acc = 1.0 * total_correct / total
    return acc


def train(train_loader, val_loader, device, model, n_epochs):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # training loop
    epochs = trange(n_epochs)
    for epoch in epochs:
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            feat = batched_graph.ndata["attr"]
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_acc = evaluate(train_loader, device, model)
        valid_acc = evaluate(val_loader, device, model)
        epochs.set_description(
            "epoch {:04d} | loss {:.4f} | train_acc {:.4f} | val_acc {:.4f} ".format(
                epoch, total_loss, train_acc, valid_acc
            )
        )


if __name__ == "__main__":
    # choices = ["MUTAG", "PTC", "NCI1", "PROTEINS"]
    # gin_global_pool_test(dataset_name="MUTAG", pooling="sum")
    # gin_global_pool_test(dataset_name="MUTAG", pooling="mean")
    # gin_global_pool_test(dataset_name="MUTAG", pooling="max")
    gin_global_pool_test(dataset_name="MUTAG", pooling="global_attention")
    # gin_global_pool_test(dataset_name="MUTAG", pooling="set2set")

# 调整网络结构，先拼接特征再gp还是先gp再拼接