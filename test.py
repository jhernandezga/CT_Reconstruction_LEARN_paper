import pytorch_lightning as pl
from model import LEARN_pl
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch
from CTSlice_Provider import CTSlice_Provider
from datamodule import CTDataModule
import torch.nn as nn
from model import GradientFunction
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import odl
import numpy as np
from odl.contrib import torch as odl_torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure 
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from torchmetrics.image import PeakSignalNoiseRatio

#lightning_checkpoint = torch.load("LEARN_Training_all/lightning_logs/version_7/hparams.yaml, map_location=lambda storage, loc: storage")
#hyperparams = lightning_checkpoint["hyper_parameters"]
#print(hyperparams)

num_view = 64
input_size = 256

path_dir ="AAPM-Mayo-CT-Challenge/"
batch_size = 16
#torch.cuda.empty_cache()
transform = transforms.Compose([transforms.Resize(input_size)])
n_iterations = 10

#tb_logger = pl.loggers.TensorBoardLogger("LEARN_Training_all")
model = LEARN_pl.load_from_checkpoint("LEARN_Training_all/lightning_logs/version_11/checkpoints/epoch=27-step=420.ckpt")

dm = CTDataModule(data_dir=path_dir, batch_size=batch_size, num_view=num_view, input_size=input_size)
dm.setup('test')

trainer = pl.Trainer(accelerator='gpu',max_epochs=10,enable_checkpointing=True)
trainer.test(model, dataloaders=dm)
#test_model(model, test_loader)

