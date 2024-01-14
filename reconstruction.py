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


num_view = 64
input_size = 256

path_dir ="AAPM-Mayo-CT-Challenge/"
#torch.cuda.empty_cache()

transform = transforms.Compose([transforms.Resize(input_size)])
n_iterations = 10


dataset = CTSlice_Provider(path_dir, num_view = num_view, input_size=input_size, transform=transform,test=True)

print(len(dataset))

phantom, fbp_u, sino = dataset[4]

initial = torch.rand(1,256, 256)

model = LEARN_pl.load_from_checkpoint("LEARN_Training_all/lightning_logs/version_7/checkpoints/epoch=9-step=1410.ckpt")
model.eval()
model.to('cpu')
y_hat = model(fbp_u,sino)

ssim = StructuralSimilarityIndexMeasure()
ssim_p = ssim(y_hat.unsqueeze(0), phantom.unsqueeze(0))
print(ssim_p)

# Plotting the images
plt.figure(0)
plt.imshow(initial.permute(1,2,0), cmap='bone')
plt.title('Initialization')
plt.axis('off')

plt.figure(1,figsize=(15,80))
plt.subplot(1, 3, 1)
plt.imshow(sino.permute(1,2,0), cmap='bone')
plt.title('Downsampled\n sinogram')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(y_hat.detach().permute(1,2,0), cmap='bone')
plt.title('Reconstructed Scan SSIM: {:.02f}'.format(ssim_p.detach().item()))
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(phantom.permute(1,2,0).detach(), cmap='bone')
plt.title('Reference Scan\n 2304 views')
plt.axis('off')
plt.show()


plt.figure(0)
plt.imshow(fbp_u.permute(1,2,0), cmap='bone')
plt.title('Initialization')
plt.axis('off')



