import copy
import shutil

import numpy as np
import pytest
import torch
import torch.utils.data as data
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader as dl
from torch.utils.data._utils.collate import default_collate
from torchvision import datasets
from torchvision.transforms import ToTensor

from flamby.datasets.fed_dummy_dataset import Baseline, BaselineLoss, FedDummyDataset
from flamby.strategies.fed_avg import FedAvg
from flamby.strategies.scaffold import Scaffold
from flamby.utils import evaluate_model_on_tests


class NeuralNetwork(nn.Module):
    def __init__(self, lambda_dropout=0.05):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        n_hidden = 128
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def cleanup():
    shutil.rmtree("./data")


@pytest.mark.parametrize("n_clients", [1, 2, 10])
def test_scaffold_integration(n_clients):
    # tests if scaffold is not failing on the MNIST dataset
    # with different number of clients
    # get the data
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    # split to n_clients
    splits = [int(len(training_data) / n_clients)] * n_clients
    splits[-1] = splits[-1] + len(training_data) % n_clients
    training_data = data.random_split(training_data, splits)

    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    train_dataloader = [
        dl(train_data, batch_size=100, shuffle=True) for train_data in training_data
    ]
    test_dataloader = dl(test_data, batch_size=100, shuffle=False)
    loss = nn.CrossEntropyLoss()
    m = NeuralNetwork()
    num_updates = 100
    nrounds = 50
    lr = 0.001
    optimizer_class = torch.optim.Adam

    s = Scaffold(train_dataloader, m, loss, optimizer_class, lr, num_updates, nrounds)
    m = s.run()

    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred.argmax(axis=1))

    res = evaluate_model_on_tests(m[0], [test_dataloader], accuracy)

    print("\nAccuracy client 0:", res["client_test_0"])
    assert res["client_test_0"] > 0.95

    cleanup()


@pytest.mark.parametrize(
    "seed, lr",
    [(42, 0.01), (43, 0.001), (44, 0.0001), (45, 7e-5)],
)
def test_fed_prox_algorithm(seed, lr):
    r"""Scaffold should add a correction term in each of its update step.
    In the first round, this correction step is 0. In each subsequent round,
    the correction depends on previous client models. the global model at each round.

    The implementation provided by the authors show that the wanted behavior
    is to update the weights to be - lr * grad + mu \cdot (var - global_var)

    Parameters
    ----------
    seed : int
        The seed to test.
    lr : float
        Learning rate.
    """

    num_updates = 10
    loss = BaselineLoss()
    torch.manual_seed(seed)

    m1 = Baseline().to(torch.double)
    m2 = copy.deepcopy(m1)

    def collate_fn_double(batch):
        outputs = default_collate(batch)
        return [o.to(torch.double) for o in outputs]

    # Generate data.
    torch.manual_seed(seed)
    training_dataloaders = [
        dl(
            FedDummyDataset(center=0, train=True, pooled=True),
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn_double,
        )
    ]

    # Run SCAFFOLD for 1 round.
    s = Scaffold(
        training_dataloaders,
        m1,
        loss,
        torch.optim.SGD,
        lr,
        num_updates=num_updates,
        nrounds=1,
        log=False,
    )
    m1 = s.run()[0]
    weights_model_after_scaffold = [p.detach().numpy() for p in m1.parameters()]

    # Run FedAvg for 1 round.
    s = FedAvg(
        training_dataloaders,
        m2,
        loss,
        torch.optim.SGD,
        lr,
        num_updates=num_updates,
        nrounds=1,
        log=False,
    )
    m2 = s.run()[0]
    weights_model_after_fedavg = [p.detach().numpy() for p in m2.parameters()]
    # When running only for 1 round, the weights should be the same.
    assert all(
        [
            np.allclose(w1, w2)
            for w1, w2 in zip(weights_model_after_scaffold, weights_model_after_fedavg)
        ]
    )
