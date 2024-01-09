import odl
from glob import glob
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch
from odl.contrib import torch as odl_torch
import os
base_path = "AAPM-Mayo-CT-Challenge/L333/full_3mm/"
#print(len(slices_path))
slices_path= glob(base_path  + "L333_FD_3_1.CT.*.*.*.*.*.*.*.*.*.IMA")
print(len(slices_path))
print(slices_path[0])

slice_path= "AAPM-Mayo-CT-Challenge/L333/full_3mm/L333_FD_3_1.CT.0001.0001.2015.12.22.20.18.05.702762.358508309.IMA"
dcm=pydicom.read_file(slice_path)
dcm.image=dcm.pixel_array*dcm.RescaleSlope+dcm.RescaleIntercept
data_slice=dcm.image
data_slice=np.array(data_slice).astype(float)
data_slice=(data_slice-np.min(data_slice))/(np.max(data_slice)-np.min(data_slice))


phantom=torch.from_numpy(data_slice).unsqueeze(0).type(torch.FloatTensor)

def _radon_transform(num_view=96, start_ang=0, end_ang=2*np.pi, num_detectors=800):
    # the function is used to generate fp, bp, fbp functions
    # the physical parameters is set as MetaInvNet and EPNet
    xx=200
    space=odl.uniform_discr([-xx, -xx], [xx, xx], [512,512], dtype='float32')
    angles=np.array(num_view).astype(int)
    angle_partition=odl.uniform_partition(start_ang, end_ang, angles)
    detector_partition=odl.uniform_partition(-480, 480, num_detectors)
    geometry=odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
    #geometry=odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
    operator=odl.tomo.RayTransform(space, geometry)

    op_norm=odl.operator.power_method_opnorm(operator)
    op_norm=torch.from_numpy(np.array(op_norm*2*np.pi)).double().cuda()

    op_layer=odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint=odl_torch.operator.OperatorModule(operator.adjoint)
    fbp=odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9)*np.sqrt(2)
    op_layer_fbp=odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

radon_curr, iradon_curr, fbp_curr, op_norm_curr = _radon_transform(num_view=64)

sino=radon_curr(phantom)

poission_level=5e6
gaussian_level=0.05

# add poission noise
intensityI0=poission_level
scale_value=torch.from_numpy(np.array(intensityI0).astype(float))
normalized_sino=torch.exp(-sino/sino.max())
th_data=np.random.poisson(scale_value*normalized_sino)
sino_noisy=-torch.log(torch.from_numpy(th_data)/scale_value)
sino_noisy = sino_noisy*sino.max()

 # add Gaussian noise
noise_std=gaussian_level
noise_std=np.array(noise_std).astype(np.float64)
nx,ny=np.array(64).astype(np.int64),np.array(800).astype(np.int64)
noise = noise_std*np.random.randn(nx,ny)
noise = torch.from_numpy(noise)
sino_noisy = sino_noisy + noise

fbp_u=fbp_curr(sino_noisy)
phantom=phantom#.type(torch.DoubleTensor)
fbp_u=fbp_u#.type(torch.DoubleTensor)
sino_noisy=sino_noisy#.type(torch.DoubleTensor)


print(sino.shape)
print(data_slice.shape)

plt.figure(0)
plt.imshow(sino[0].cpu().numpy(), cmap='bone')

plt.figure(1)
plt.imshow(data_slice, cmap='bone')
plt.show()

plt.figure(2)
plt.imshow(fbp_u[0], cmap='bone')

plt.figure(figsize=(10, 8))
plt.imshow(sino_noisy[0].cpu().numpy(), cmap='bone')