from timm import create_model
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self,num_classes=5):
        super().__init__()
        self.model = create_model('resnet18d',num_classes=num_classes,pretrained=True)
    
    def forward(self,x):
        return self.model(x)