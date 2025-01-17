# import random

# import albumentations
# import torch
# import torch.nn as nn
# from efficientnet_pytorch import EfficientNet

# from flamby.datasets.fed_isic2019 import FedIsic2019


# class Baseline(nn.Module):
#     """Baseline model
#     We use the EfficientNets architecture that many participants in the ISIC
#     competition have identified to work best.
#     See here the [reference paper](https://arxiv.org/abs/1905.11946)
#     Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
#     [pytorch reimplementation of EfficientNets]
#     (https://github.com/lukemelas/EfficientNet-PyTorch).
#     """

#     def __init__(self, pretrained=True, arch_name="efficientnet-b0"):
#         super(Baseline, self).__init__()
#         self.pretrained = pretrained
#         self.base_model = (
#             EfficientNet.from_pretrained(arch_name)
#             if pretrained
#             else EfficientNet.from_name(arch_name)
#         )
#         # self.base_model=torchvision.models.efficientnet_v2_s(pretrained=pretrained)
#         nftrs = self.base_model._fc.in_features
#         print("Number of features output by EfficientNet", nftrs)
#         self.base_model._fc = nn.Linear(nftrs, 8)

#     def forward(self, image):
#         out = self.base_model(image)
#         return out


# if __name__ == "__main__":

#     sz = 200
#     train_aug = albumentations.Compose(
#         [
#             albumentations.RandomScale(0.07),
#             albumentations.Rotate(50),
#             albumentations.RandomBrightnessContrast(0.15, 0.1),
#             albumentations.Flip(p=0.5),
#             albumentations.Affine(shear=0.1),
#             albumentations.RandomCrop(sz, sz),
#             albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
#             albumentations.Normalize(always_apply=True),
#         ]
#     )

#     mydataset = FedIsic2019(0, True, "train", augmentations=train_aug)

#     model = Baseline()

#     for i in range(50):
#         X = torch.unsqueeze(mydataset[i][0], 0)
#         y = torch.unsqueeze(mydataset[i][1], 0)
#         print(X.shape)
#         print(y.shape)
#         print(model(X))



from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

# class Baseline(nn.Module):
#     def __init__(self,num_classes=8):
#         super().__init__()
#         self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#         for p in self.model.parameters():
#             p.requires_grad=False
#         for p in self.model.fc.parameters():
#             p.requires_grad=True

#     def forward(self,x):
#         return self.model(x)


class Baseline(nn.Module):
    def __init__(self, num_classes=8, freeze_up_to_layer=46): #experiment with 45,54,60
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze layers up to the specified layer
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            # print(idx)
            # print(name)
            if idx < freeze_up_to_layer:
                param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
