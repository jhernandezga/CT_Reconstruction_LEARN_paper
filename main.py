import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch
from CTSlice_Provider import CTSlice_Provider
from model import LEARN_pl
from datamodule import CTDataModule
import torch.nn as nn
from model import GradientFunction
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import odl
import numpy as np
from odl.contrib import torch as odl_torch
from pytorch_lightning import seed_everything

#torch.cuda.set_device(1)
#mp.set_start_method('spawn')
#torch.manual_seed(42)
num_view = 64
input_size = 256

path_dir ="AAPM-Mayo-CT-Challenge/"
num_select  =  25   #num image training per patient, select all = -1
#torch.cuda.empty_cache()

n_iterations = 30
batch_size = 16

seed_everything(42, workers=True)
tb_logger = pl.loggers.TensorBoardLogger("LEARN_Training_all")
model = LEARN_pl(n_iterations=n_iterations,num_view=num_view)
dm = CTDataModule(data_dir=path_dir, batch_size=batch_size, num_view=num_view, input_size=input_size, num_select=num_select)

#dm.setup(stage="fit")
#batch = dm.train_dataloader().__iter__().__next__()
trainer = pl.Trainer(accelerator='gpu',max_epochs=50, logger=tb_logger,enable_checkpointing=True)
trainer.fit(model, dm)