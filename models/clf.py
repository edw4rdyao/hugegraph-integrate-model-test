import torch
from torch import nn


class Classifier(nn.Module):
    r"""
    Classifier module to predict node classes based on embeddings.

    Parameters
    -----------
    n_hidden: int
        Hidden feature size.
    n_classes: int
        Number of classes to predict.
    """

    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        # Initialize the fully connected layer for classification.
        self.fc = nn.Linear(n_hidden, n_classes)
        self.fc.reset_parameters()

    def forward(self, feat):
        # Apply linear transformation and log softmax for classification probabilities.
        feat = self.fc(feat)
        return torch.log_softmax(feat, dim=-1)
