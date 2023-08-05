import torch

from flamby.datasets.fed_covid19.dataset import FedCovid19

NUM_CLIENTS = 7
BATCH_SIZE = 4
NUM_EPOCHS_POOLED = 1
LR = 1e-3
Optimizer = torch.optim.Adam

FedClass = FedCovid19

def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # setting num_updates=100
    n_samples = 655
    return (n_samples // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates # comes out to 2
