import torch

from flamby.datasets.fed_cifar10.dataset import FedCIFAR10

NUM_CLIENTS = 10
BATCH_SIZE = 16
NUM_EPOCHS_POOLED = 10
LR = 1e-3
Optimizer = torch.optim.Adam

FedClass = FedCIFAR10

def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # setting num_updates=100
    n_samples = 7496
    return (n_samples // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
