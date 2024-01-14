import torch
import numpy as np
import matplotlib.pyplot as plt
from CTSlice_Provider import *

#torch.cuda.empty_cache()
from odl.tomo.backends import (
    ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE, SKIMAGE_AVAILABLE)

print("Astra CUDA available", ASTRA_CUDA_AVAILABLE)

path_dir ="AAPM-Mayo-CT-Challenge/"

dataset = CTSlice_Provider(path_dir,num_view=64, test=True)

print(len(dataset))
phantom, fbp_u, sino_noisy = dataset[10]

phantom = phantom.permute(1, 2, 0)
fbp_u = fbp_u.permute(1, 2, 0)
sino_noisy = sino_noisy.permute(1, 2, 0)

# Scale the tensor between 0 and 255
phantom_scaled = torch.clamp(phantom, 0, 255).numpy()
fbp_scaled = torch.clamp(fbp_u, 0, 255).numpy()
sino_noisy_scaled = torch.clamp(sino_noisy, 0, 255).numpy()

plt.figure(0)
plt.imshow(phantom_scaled, cmap = 'bone')
plt.show()
print(phantom.shape)

plt.figure(1)
plt.imshow(fbp_scaled, cmap = 'bone')
plt.show()
print(phantom.shape)

plt.figure(2)
plt.imshow(sino_noisy_scaled, cmap = 'bone')
plt.show()
print(phantom.shape)

#plt.figure(0)
#plt.plot(phantom.permute(1,2,0))
#plt.show()
