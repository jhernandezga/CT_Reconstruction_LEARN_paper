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
    def __init__(self,data_dir, batch_size):
        super().__init__()
        self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                 
                        
                        ])
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage):
        # First stage is 'fit' (or None)
        if stage == "fit" or stage is None:
            # We create a validation split to watch the training.
            ct_train_dataset = CTSlice_Provider(self.data_dir)
         
            self.train_size = int(0.8 * len(ct_train_dataset))
            self.valid_size = len(ct_train_dataset) - self.train_size
            self.ct_train, self.ct_valid =  torch.utils.data.random_split(ct_train_dataset, [self.train_size, self.valid_size])

    def train_dataloader(self):
        return DataLoader(self.ct_train, batch_size=self.batch_size, shuffle=True,num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.ct_train, self.batch_size, shuffle=False,num_workers=0)