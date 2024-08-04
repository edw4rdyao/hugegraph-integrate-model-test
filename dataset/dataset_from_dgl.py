from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset


def dataset_from_dgl(dataset_name):
    if dataset_name == "cora":
        dataset = CoraGraphDataset()
    elif dataset_name == "citeseer":
        dataset = CiteseerGraphDataset()
    elif dataset_name == "pubmed":
        dataset = PubmedGraphDataset()
    else:
        raise ValueError("do not support")
    return dataset
