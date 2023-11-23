import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from pathlib import Path



class FedFmnist:
    def __init__(self,center=0,train=True,pooled=False):
        self.center = center
        self.split = 'train' if train else 'test'
        self.pooled = pooled
        train_transforms = A.Compose([
            #A.RandomScale(0.1),
            #A.RandomBrightnessContrast(0.15, 0.1),
            #A.HorizontalFlip(p=0.5),
            A.Resize(224, 224),
            ToTensorV2()
        ])
        val_transforms = A.Compose([
            A.Resize(224,224),
            ToTensorV2()
        ])
        self.tfms = train_transforms if train else val_transforms
        
        self.df_path = Path(__file__).parent / 'fmnist_splits.csv'
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
        base = Path('/data2/Shreyas/FLamby/flamby/datasets/fed_fmnist').resolve()
        for i in range(len(cases)):
            client_dicts.append({
                'image': base / f"{cases.loc[i,'path']}",
                'label': int(cases.loc[i,'label'])
            })
        return client_dicts
    
    def __getitem__(self,idx):
        case = self.cases[idx]
        im, label = case['image'], case['label']
        im = np.array(Image.open(im).convert('L'))
        aug = self.tfms(image=im)
        im = aug['image'] / 255.
        label = torch.tensor(label).long()
        return im,label
