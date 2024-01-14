import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from CTSlice_Provider import CTSlice_Provider



class CTDataModule(pl.LightningDataModule):
    def __init__(self,data_dir, batch_size,num_view=64,input_size = 256, num_select = -1):
        super().__init__()
        self.input_size = input_size
        self.transform = transforms.Compose([
                                transforms.Resize(self.input_size)
                        ])
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_view = num_view
        self.num__select = num_select

    def setup(self, stage):
        # First stage is 'fit' (or None)
        if stage == "fit" or stage is None:
            # We create a validation split to watch the training.
            self.ct_train = CTSlice_Provider(self.data_dir, num_view=self.num_view, input_size=self.input_size, transform=self.transform,num_select=self.num__select)
            print(len(self.ct_train))
            self.ct_valid = CTSlice_Provider(self.data_dir,test=True, num_view=self.num_view, input_size=self.input_size, transform=self.transform,num_select=self.num__select)
        
        if stage == "test" or stage is None:
            self.ct_test = CTSlice_Provider(self.data_dir, num_view=self.num_view, input_size=self.input_size, transform=self.transform,test=True,num_select=self.num__select)
        
    def train_dataloader(self):
        return DataLoader(self.ct_train, batch_size=self.batch_size, shuffle=True,num_workers=0)
    def val_dataloader(self):
        return DataLoader(self.ct_train, self.batch_size, shuffle=False,num_workers=0)
    def test_dataloader(self):
        return DataLoader(self.ct_test, self.batch_size, shuffle=False,num_workers=0)
    