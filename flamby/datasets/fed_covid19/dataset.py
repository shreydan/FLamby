import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from pathlib import Path


def normalize(img, maxval=255, reshape=False):
    """
    All credits to: https://github.com/mlmed/torchxrayvision
    Permalink: https://github.com/mlmed/torchxrayvision/blob/cd669b6af0279be8b2a6674b7366878a76f75fba/torchxrayvision/utils.py#L45
    Scales images to be roughly [-1024 1024].
    """

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img


class FedCovid19:
    def __init__(self,center=0,train=True,pooled=False):
        self.center = center
        self.split = 'train' if train else 'test'
        self.pooled = pooled
        train_transforms = A.Compose([
            A.RandomScale(0.1),
            A.RandomBrightnessContrast(0.15, 0.1),
            A.HorizontalFlip(p=0.5),
            A.Resize(224, 224),
        ])
        val_transforms = A.Compose([
            A.Resize(224,224)
        ])
        self.to_torch = A.Compose([ToTensorV2()])
        self.tfms = train_transforms if train else val_transforms
        
        self.df_path = Path(__file__).parent / 'splits_main.csv'
        df = pd.read_csv(self.df_path)
        main_cases = df.query(f"split == '{self.split}'").reset_index(drop=True)
        if not self.pooled:
            client_cases = main_cases.query(f"client == {center}").reset_index(drop=True)
        else:
            client_cases = main_cases
        self.cases = self._make_dict(client_cases)
        
    def __len__(self,):
        return len(self.cases)
        
    def _make_dict(self,cases):
        client_dicts = []
        base = Path('/kaggle/working/covid-chestxray-dataset/images').resolve()
        for i in range(len(cases)):
            client_dicts.append({
                'image': base / f"{cases.loc[i,'filename']}",
                'label': int(cases.loc[i,'covid_19'])
            })
        return client_dicts
    
    def __getitem__(self,idx):
        case = self.cases[idx]
        im, label = case['image'], case['label']
        im = np.array(Image.open(im).convert('RGB'))
        aug = self.tfms(image=im)
        im = aug['image']
        im = normalize(im)
        im = self.to_torch(image=im)['image']
        label = torch.tensor([label]).float()
        return im,label
