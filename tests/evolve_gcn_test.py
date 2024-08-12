import dgl
import torch
import torch.nn.functional as F

from dataset.dataset_from_dgl import dataset_from_dgl_download
from models.evolve_gcn import EvolveGCNH, EvolveGCNO


def calculate_measure(tp, fn, fp):
    # avoid nan
    if tp == 0:
        return 0, 0, 0

    p = tp * 1.0 / (tp + fp)
    r = tp * 1.0 / (tp + fn)
    if (p + r) > 0:
        f1 = 2.0 * (p * r) / (p + r)
    else:
        f1 = 0
    return p, r, f1


class Measure(object):
    def __init__(self, num_classes, target_class):
        """

        Args:
            num_classes: number of classes.
            target_class: target class we focus on, used to print info and do early stopping.
        """
        self.num_classes = num_classes
        self.target_class = target_class
        self.true_positives = {}
        self.false_positives = {}
        self.false_negatives = {}
        self.target_best_f1 = 0.0
        self.target_best_f1_epoch = 0
        self.reset_info()

    def reset_info(self):
        """
        reset info after each epoch.
        """
        self.true_positives = {
            cur_class: [] for cur_class in range(self.num_classes)
        }
        self.false_positives = {
            cur_class: [] for cur_class in range(self.num_classes)
        }
        self.false_negatives = {
            cur_class: [] for cur_class in range(self.num_classes)
        }

    def append_measures(self, predictions, labels):
        predicted_classes = predictions.argmax(dim=1)
        for cl in range(self.num_classes):
            cl_indices = labels == cl
            pos = predicted_classes == cl
            hits = predicted_classes[cl_indices] == labels[cl_indices]

            tp = hits.sum()
            fn = hits.size(0) - tp
            fp = pos.sum() - tp

            self.true_positives[cl].append(tp.cpu())
            self.false_negatives[cl].append(fn.cpu())
            self.false_positives[cl].append(fp.cpu())

    def get_each_timestamp_measure(self):
        precisions = []
        recalls = []
        f1s = []
        for i in range(len(self.true_positives[self.target_class])):
            tp = self.true_positives[self.target_class][i]
            fn = self.false_negatives[self.target_class][i]
            fp = self.false_positives[self.target_class][i]

            p, r, f1 = calculate_measure(tp, fn, fp)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        return precisions, recalls, f1s

    def get_total_measure(self):
        tp = sum(self.true_positives[self.target_class])
        fn = sum(self.false_negatives[self.target_class])
        fp = sum(self.false_positives[self.target_class])

        p, r, f1 = calculate_measure(tp, fn, fp)
        return p, r, f1

    def update_best_f1(self, cur_f1, cur_epoch):
        if cur_f1 > self.target_best_f1:
            self.target_best_f1 = cur_f1
            self.target_best_f1_epoch = cur_epoch


def evolve_gcn_test(
        model="EvolveGCN-O",
        n_hidden=256,
        n_layers=2,
        lr=0.001,
        n_hist_steps=5,
        loss_class_weight="0.35,0.65",
        eval_class_id=1,
        n_epochs=1000,
        early_stopping=100,
        gpu="0",
):
    device = "cuda:{}".format(gpu) if gpu != -1 and torch.cuda.is_available() else "cpu"
    g, node_mask_by_time, num_classes = dataset_from_dgl_download("elliptic")

    subgraphs = []
    labeled_node_masks = []
    for i in range(len(node_mask_by_time)):
        # we add self loop edge when we construct full graph, not here
        node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i])
        subgraphs.append(node_subgraph.to(device))
        valid_node_mask = node_subgraph.ndata["label"] >= 0
        labeled_node_masks.append(valid_node_mask)

    if model == "EvolveGCN-O":
        model = EvolveGCNO(
            in_feats=int(g.ndata["feat"].shape[1]),
            n_hidden=n_hidden,
            num_layers=n_layers,
        )
    elif model == "EvolveGCN-H":
        model = EvolveGCNH(
            in_feats=int(g.ndata["feat"].shape[1]),
            num_layers=n_layers
        )
    else:
        return NotImplementedError("Unsupported model {}".format(model))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # split train, valid, test(0-30,31-35,36-48)
    # train/valid/test split follow the paper.
    train_max_index = 30
    valid_max_index = 35
    test_max_index = 48
    time_window_size = n_hist_steps
    loss_class_weight = torch.Tensor([float(w) for w in loss_class_weight.split(",")]).to(device)

    train_measure = Measure(
        num_classes=num_classes, target_class=eval_class_id
    )
    valid_measure = Measure(
        num_classes=num_classes, target_class=eval_class_id
    )
    test_measure = Measure(
        num_classes=num_classes, target_class=eval_class_id
    )

    test_res_f1 = 0
    for epoch in range(n_epochs):
        model.train()
        for i in range(time_window_size, train_max_index + 1):
            graph_list = subgraphs[i - time_window_size: i + 1]
            predictions = model(graph_list)
            predictions = predictions[labeled_node_masks[i]]
            labels = (
                subgraphs[i]
                .ndata["label"][labeled_node_masks[i]]
                .long()
            )
            loss = F.cross_entropy(predictions, labels, weight=loss_class_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_measure.append_measures(predictions, labels)

        # get each epoch measures during training.
        cl_precision, cl_recall, cl_f1 = train_measure.get_total_measure()
        train_measure.update_best_f1(cl_f1, epoch)
        # reset measures for next epoch
        train_measure.reset_info()

        print(
            "Train Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}".format(
                epoch, eval_class_id, cl_precision, cl_recall, cl_f1
            )
        )

        # eval
        model.eval()
        for i in range(train_max_index + 1, valid_max_index + 1):
            graph_list = subgraphs[i - time_window_size: i + 1]
            predictions = model(graph_list)
            predictions = predictions[labeled_node_masks[i]]
            labels = (
                subgraphs[i]
                .ndata["label"][labeled_node_masks[i]]
                .long()
            )

            valid_measure.append_measures(predictions, labels)

        # get each epoch measure during eval.
        cl_precision, cl_recall, cl_f1 = valid_measure.get_total_measure()
        valid_measure.update_best_f1(cl_f1, epoch)
        # reset measures for next epoch
        valid_measure.reset_info()

        print(
            "Eval Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}".format(
                epoch, eval_class_id, cl_precision, cl_recall, cl_f1
            )
        )

        # early stop
        if epoch - valid_measure.target_best_f1_epoch >= early_stopping:
            print(
                "Best eval Epoch {}, Cur Epoch {}".format(
                    valid_measure.target_best_f1_epoch, epoch
                )
            )
            break
        # if cur valid f1 score is best, do test
        if epoch == valid_measure.target_best_f1_epoch:
            for i in range(valid_max_index + 1, test_max_index + 1):
                graph_list = subgraphs[i - time_window_size: i + 1]
                predictions = model(graph_list)
                # get predictions which has label
                predictions = predictions[labeled_node_masks[i]]
                labels = (
                    subgraphs[i]
                    .ndata["label"][labeled_node_masks[i]]
                    .long()
                )

                test_measure.append_measures(predictions, labels)

            # we get each subgraph measure when testing to match fig 4 in EvolveGCN paper.
            (
                cl_precisions,
                cl_recalls,
                cl_f1s,
            ) = test_measure.get_each_timestamp_measure()
            for index, (sub_p, sub_r, sub_f1) in enumerate(
                    zip(cl_precisions, cl_recalls, cl_f1s)
            ):
                print(
                    "  Test | Time {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}".format(
                        valid_max_index + index + 2, sub_p, sub_r, sub_f1
                    )
                )

            # get each epoch measure during test.
            cl_precision, cl_recall, cl_f1 = test_measure.get_total_measure()
            test_measure.update_best_f1(cl_f1, epoch)
            # reset measures for next test
            test_measure.reset_info()

            test_res_f1 = cl_f1

            print(
                "  Test | Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}".format(
                    epoch, eval_class_id, cl_precision, cl_recall, cl_f1
                )
            )

    print(
        "Best test f1 is {}, in Epoch {}".format(
            test_measure.target_best_f1, test_measure.target_best_f1_epoch
        )
    )
    if test_measure.target_best_f1_epoch != valid_measure.target_best_f1_epoch:
        print(
            "The Epoch get best Valid measure not get the best Test measure, "
            "please checkout the test result in Epoch {}, which f1 is {}".format(
                valid_measure.target_best_f1_epoch, test_res_f1
            )
        )
