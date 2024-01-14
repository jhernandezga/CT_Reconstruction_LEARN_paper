import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from torchmetrics.image import PeakSignalNoiseRatio
import odl
import numpy as np
from odl.contrib import torch as odl_torch

class RegularizationBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,kernel_size=5):
        super(RegularizationBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=kernel_size, padding='same')
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=kernel_size, padding='same')
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        self.conv3 = nn.Conv2d(48, out_channels, kernel_size=kernel_size, padding='same')
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class GradientFunction(nn.Module):
    
    def __init__(self):
        super(GradientFunction, self).__init__()
        self.regularitation_term = RegularizationBlock()
        self.alpha = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self,x_t, y, forward_module, backward_module):
        data_fidelity_term = forward_module(x_t) - y
        bp_data_fidelity = backward_module(data_fidelity_term)
        reg_value = self.regularitation_term(x_t)
        gradient = self.alpha * bp_data_fidelity + reg_value
        return gradient
  

class LEARN_pl(pl.LightningModule):
    def __init__(self, n_iterations,num_view):
        super(LEARN_pl, self).__init__()
        self.save_hyperparameters()
        self.gradient_list = nn.ModuleList([GradientFunction() for _ in range(n_iterations)])
        self.initial_lr = 1e-4
        self.final_lr = 1e-5
        self.num_iter = n_iterations
        self.ssim = StructuralSimilarityIndexMeasure()
        self.psnr = PeakSignalNoiseRatio()
        #self.rmse = RootMeanSquaredErrorUsingSlidingWindow()
        #radon_curr, fbp_curr = radon(num_view=num_view)
        radon_curr, fbp_curr = self.radon_transform(num_view=num_view)
        
        self.forward_module = radon_curr
        self.backward_module = fbp_curr
        
        self.grid = None
    
    def forward(self,x_t,y):
        x_t = x_t
        for i in range(self.num_iter):
            x_t = x_t - self.gradient_list[i](x_t, y, self.forward_module, self.backward_module)
        return x_t

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=self.final_lr)
        
        return (
        {   "optimizer":optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }   )

    def training_step(self, train_batch, batch_idx):
        phantom, fbp_u, sino_noisy = train_batch
        x_t = fbp_u
        y = sino_noisy
        initial = torch.rand(y.shape[0],1,256, 256).cuda()
        x_reconstructed = self.forward(x_t, y)        
        loss = nn.functional.mse_loss(phantom, x_reconstructed)

        #self.log('train_loss', loss)
        self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step) 
        return loss

    def validation_step(self, val_batch, batch_idx):
        
        phantom, fbp_u, sino_noisy = val_batch
        x_t = fbp_u
        y = sino_noisy
        initial = torch.rand(y.shape[0],1,256, 256).cuda()
        x_reconstructed = self.forward(x_t, y)
        
        loss = nn.functional.mse_loss(phantom, x_reconstructed)
        ssim_p = self.ssim(x_reconstructed, phantom)
        psnr_p = self.psnr(x_reconstructed, phantom)
        rmse_p = self.rmse(x_reconstructed, phantom)
        
        
        #self.log('val_loss', loss)
        self.logger.experiment.add_scalars('loss', {'validation': loss},self.global_step)
        self.log('val_ssim', ssim_p)
        self.log('val_psnr', psnr_p)
        self.log('val_rmse', rmse_p)
        
        self.grid = torchvision.utils.make_grid(x_reconstructed)
        
    def test_step(self, batch, batch_idx):
        phantom, fbp_u, sino_noisy = batch
        x_t = fbp_u
        y = sino_noisy
        initial = torch.rand(y.shape[0],1,256, 256).cuda()
        x_reconstructed = self.forward(x_t, y)
        
        loss = nn.functional.mse_loss(phantom, x_reconstructed)
        ssim_p = self.ssim(x_reconstructed, phantom)
        psnr_p = self.psnr(x_reconstructed, phantom)
        rmse_p = self.rmse(x_reconstructed, phantom)
        
        test_out = {
            "SSIM": ssim_p,
            "PSNR": psnr_p,
            "RMSE": rmse_p
            }
        
        self.log('test_loss', loss)
        self.log('test_ssim', ssim_p)
        self.log('test_psnr', psnr_p)
        self.log('test_rmse', rmse_p)
        return test_out   
    
    def on_validation_epoch_end(self):
        self.logger.experiment.add_image("generated_images", self.grid, self.current_epoch,)
    
    
    def radon_transform(self, num_view=64, start_ang=0, end_ang=2*np.pi, num_detectors=800):
        # the function is used to generate fp, bp, fbp functions
        # the physical parameters is set as MetaInvNet and EPNet
        xx=200
        space=odl.uniform_discr([-xx, -xx], [xx, xx], [256,256], dtype='float32')
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
    
    def on_validation_epoch_end(self):
        tag = f"generated_images_epoch_{self.current_epoch}"
        self.logger.experiment.add_image(tag, self.grid, self.current_epoch)
        
    def rmse(self,y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    
  
    
    
  