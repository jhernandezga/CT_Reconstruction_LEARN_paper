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


#mp.set_start_method('spawn')
#torch.manual_seed(42)

num_view = 96
input_size = 256

def radon_transform(num_view=num_view, start_ang=0, end_ang=2*np.pi, num_detectors=800):
        # the function is used to generate fp, bp, fbp functions
        # the physical parameters is set as MetaInvNet and EPNet
        xx=200
        space=odl.uniform_discr([-xx, -xx], [xx, xx], [input_size,input_size], dtype='float32')
        angles=np.array(num_view).astype(int)
        angle_partition=odl.uniform_partition(start_ang, end_ang, angles)
        detector_partition=odl.uniform_partition(-480, 480, num_detectors)
        geometry=odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
        #geometry=odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
        operator=odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

        #op_norm=odl.operator.power_method_opnorm(operator)
        #op_norm=torch.from_numpy(np.array(op_norm*2*np.pi)).double().cuda()

        op_layer=odl_torch.operator.OperatorModule(operator)
        #op_layer_adjoint=odl_torch.operator.OperatorModule(operator.adjoint)
        fbp=odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9)*np.sqrt(2)
        op_layer_fbp=odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_fbp


path_dir ="AAPM-Mayo-CT-Challenge/"
#torch.cuda.empty_cache()

n_iterations = 10
batch_size = 4

tb_logger = pl.loggers.TensorBoardLogger("LEARN_Training_all")
model = LEARN_pl(n_iterations=n_iterations, radon=radon_transform)
dm = CTDataModule(data_dir=path_dir, batch_size=batch_size, num_view=num_view, input_size=input_size)

#dm.setup(stage="fit")
#batch = dm.train_dataloader().__iter__().__next__()

trainer = pl.Trainer(accelerator='gpu',max_epochs=20, logger=tb_logger,enable_checkpointing=False)
trainer.fit(model, dm)