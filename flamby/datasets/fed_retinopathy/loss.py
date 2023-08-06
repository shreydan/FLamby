import torch.nn as nn
from torch.nn.modules.loss import _Loss

class BaselineLoss(_Loss):
    def __init__(self):
        super(BaselineLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)