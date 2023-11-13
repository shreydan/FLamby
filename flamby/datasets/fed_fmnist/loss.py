import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class BaselineLoss(_Loss):
    def __init__(self):
        super(BaselineLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = self.cross_entropy(input, target)
        return loss