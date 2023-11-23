import torch

from flamby.datasets.fed_retinopathy.dataset import FedRetinopathy

# 0-4: aptos 5-9: eyepacs
NUM_CLIENTS = 10
BATCH_SIZE = 64
NUM_EPOCHS_POOLED = 1
LR = 1e-3
Optimizer = torch.optim.Adam

FedClass = FedRetinopathy

def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # setting num_updates=100
    n_samples = 9039 # train_samples
    return (n_samples // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
