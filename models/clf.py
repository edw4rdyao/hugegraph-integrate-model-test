import torch
from torch import nn


class Classifier(nn.Module):
    r"""
    A simple classifier module for predicting node classes based on embeddings.

    Parameters
    ----------
    n_hidden : int
        The number of hidden units (input feature size) for the classifier.
    n_classes : int
        The number of classes to predict (output feature size).
    """

    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        # Define a fully connected (linear) layer for classification.
        self.fc = nn.Linear(n_hidden, n_classes)
        self.fc.reset_parameters()  # Initialize the parameters of the linear layer

    def forward(self, feat):
        """
        Forward pass to compute class predictions.

        Parameters
        ----------
        feat : torch.Tensor
            Input node embeddings with shape (batch_size, n_hidden).

        Returns
        -------
        torch.Tensor
            Log-softmax probabilities of shape (batch_size, n_classes).
        """
        feat = self.fc(feat)  # Apply the linear transformation
        return torch.log_softmax(feat, dim=-1)  # Apply log-softmax to obtain class probabilities
