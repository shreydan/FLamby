import torch

from flamby.datasets.fed_fmnist.dataset import FedFMnist

NUM_CLIENTS = 10
BATCH_SIZE = 64
NUM_EPOCHS_POOLED = 30
LR = 1e-3
Optimizer = torch.optim.Adam

FedClass = FedFMnist

def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # setting num_updates=100
    n_samples = 5002
    return (n_samples // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
