from flamby.datasets.fed_retinopathy.common import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    NUM_EPOCHS_POOLED,
    Optimizer,
    get_nb_max_rounds,
    FedClass,
)
from flamby.datasets.fed_retinopathy.dataset import FedRetinopathy
from flamby.datasets.fed_retinopathy.metric import metric
from flamby.datasets.fed_retinopathy.model import Baseline
from flamby.datasets.fed_retinopathy.loss import BaselineLoss