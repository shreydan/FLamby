from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        for p in self.model.parameters():
            p.requires_grad=False
        for p in self.model.fc.parameters():
            p.requires_grad=True

    def forward(self,x):
        return self.model(x)