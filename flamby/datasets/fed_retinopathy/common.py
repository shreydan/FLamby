import torch

from flamby.datasets.fed_retinopathy.dataset import FedRetinopathy

# 0: aptos 1: eyepacs
NUM_CLIENTS = 2
BATCH_SIZE = 64
NUM_EPOCHS_POOLED = 10
LR = 1e-3
Optimizer = torch.optim.Adam

FedClass = FedRetinopathy

def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # setting num_updates=100
    n_samples = 36647 # train_samples
    return (n_samples // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates # comes out to 28, with bs=64
