"""
Additional loss functions for HDR

"""

import math
import random
import numpy as np

import os
import torch
import torch.nn as nn
import torch.fft as fft

import pycvvdp 
from utils_color import *
import utils
## TODO: Add more color loss functions

def cvvdp_loss(final_amp, target_amp, metric,disp_photo, dev):
    I_test = final_amp.squeeze()**2 
    I_ref = target_amp.squeeze()**2 
    brightness = 5
    I_test_physical = I_test*brightness
    I_ref_physical = I_ref*brightness
    # loss = metric.loss( I_test_physical, I_ref_physical, dim_order="CHW") + \
    #         0.01 * s_cielab_loss(final_amp, target_amp,disp_photo,dev)
    loss = metric.loss( I_test_physical, I_ref_physical, dim_order="CHW") + 0.0001 * s_cielab_loss(final_amp, target_amp,disp_photo,dev)
    return loss

def cvvdp_loss_video(final_amp, target_amp, metric,disp_photo, dev):
    I_test = final_amp.squeeze()**2 
    I_ref = target_amp.squeeze()**2 
    brightness = 100
    I_test_physical = I_test*brightness
    I_ref_physical = I_ref*brightness
    # loss = metric.loss( I_test_physical, I_ref_physical, dim_order="CHW") + \
    #         0.01 * s_cielab_loss(final_amp, target_amp,disp_photo,dev)
    loss = metric.loss( I_test_physical, I_ref_physical, dim_order="FCHW",frames_per_second=8)
    return loss

def normalize_matrix_to_y_100(matrix):
    x_channel = matrix[0,...]
    y_channel = matrix[1,...]
    z_channel = matrix[2,...]
    
    scale_factors = 1 / y_channel
    
    normalized_x = x_channel * scale_factors
    normalized_y = y_channel * scale_factors
    normalized_z = z_channel * scale_factors
    
    # Stack the normalized channels back into a single matrix
    normalized_matrix = torch.stack((normalized_x, normalized_y, normalized_z), dim=0)
    return normalized_matrix

def cielab_loss(final_amp, target_amp, disp_photo, dev):
    final_inten = final_amp.squeeze()**2 
    target_inten = target_amp.squeeze()**2 

    RGB2XYZ_matrix = [
    [1.2493,0.1147,0.8357],
    [0.4976,1.6037,0.0581],
    [0.0001,0.2151,4.2074]
    ]
    RGB2XYZ = torch.tensor(RGB2XYZ_matrix).to(dev)
    Brightness = 100
    test_xyz = torch.einsum('ij,jab->iab', RGB2XYZ, final_inten*Brightness)
    ref_xyz = torch.einsum('ij,jab->iab', RGB2XYZ, target_inten*Brightness)

    test_lab = xyz2lab(test_xyz, dev)
    ref_lab = xyz2lab(ref_xyz, dev)

    loss_fn = nn.MSELoss()
    loss = loss_fn(test_lab, ref_lab)
    return loss

def CIE94deltaE_loss(final_amp, target_amp, disp_photo, dev):
    ## Loss function for l2 loss in CIELAB color space
    ## CIE94 color difference
    ## https://en.wikipedia.org/wiki/Color_difference
    ## Make sure the input is in [C,H,W,U,V] format

    # disp_photo = pycvvdp.vvdp_display_photo_eotf(L_peak, source_colorspace=source_type)

    ## For test: single channel input, just augment the channels
    final_inten = final_amp.squeeze()**2 
    target_inten = target_amp.squeeze()**2 

    I_test_physical = final_inten*disp_photo.Y_peak
    I_ref_physical = target_inten*disp_photo.Y_peak

    ## RGB to Lab
    RGB2XYZ = torch.tensor(disp_photo.__dict__['rgb2xyz_list']).to(dev)

    test_lab = xyz2lab(torch.einsum('ij,jab->iab', RGB2XYZ, I_test_physical), dev)
    ref_lab = xyz2lab(torch.einsum('ij,jab->iab', RGB2XYZ, I_ref_physical), dev)

    KL = 1
    K1 = 0.045
    K2 = 0.015

    dL = (test_lab[0,...] - ref_lab[0,...])
    da = (test_lab[1,...] - ref_lab[1,...])
    db = (test_lab[2,...] - ref_lab[2,...])

    C1 = torch.sqrt(test_lab[1,...]**2 + test_lab[2,...]**2)
    C2 = torch.sqrt(ref_lab[1,...]**2 + ref_lab[2,...]**2)
    dCab = C1 - C2
    ## Cause NaN value in some cases and putting it zero will cause wrong estimation of loss
    dHab = torch.sqrt(torch.max(da**2 +db**2 - dCab**2, torch.zeros_like(da)+1e-20)) 
    
    
    loss = torch.mean(torch.mean(torch.sqrt((dL/KL)**2 + (dCab/(1+K1*C1))**2 + (dHab/(1+K2*C1))**2), (-2, -1)))
    return loss

def inten_l2_loss(final_amp, target_amp):
    final_inten = final_amp.squeeze()**2
    target_inten = target_amp.squeeze()**2 
                                               
    loss_fn = nn.MSELoss()
    loss = loss_fn(final_inten, target_inten)

    return loss



def create_low_pass_filter(shape, cutoff_ratio):
    H, W = shape
    center_x, center_y = W // 2, H // 2
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    cutoff = cutoff_ratio * (W / 2)
    filter_mask = (distance <= cutoff).float()
    return filter_mask

def apply_filter(input_channel, filter_kernel):
    input_fft = fft.fft2(input_channel)
    input_fft = fft.fftshift(input_fft) 

    filtered_fft = input_fft * filter_kernel

    filtered_output = fft.ifft2(fft.ifftshift(filtered_fft)).real
    return filtered_output

def filtered_CIELAB_loss(final_amp, target_amp,disp_photo,dev):

    final_inten = final_amp.squeeze()**2
    target_inten = target_amp.squeeze()**2

    I_test_physical = final_inten*disp_photo.Y_peak
    I_ref_physical = target_inten*disp_photo.Y_peak

    RGB2XYZ = torch.tensor(disp_photo.__dict__['rgb2xyz_list']).to(dev)
    predicted_XYZ = torch.einsum('ij,jab->iab', RGB2XYZ, I_test_physical)
    target_XYZ  = torch.einsum('ij,jab->iab', RGB2XYZ, I_ref_physical)

    pred_lab = xyz2lab(predicted_XYZ,dev)
    target_lab = xyz2lab(target_XYZ,dev)
    O1_pred = pred_lab[0, :, :]
    O2_pred = pred_lab[1, :, :]
    O3_pred = pred_lab[2, :, :]

    O1_target = target_lab[0, :, :]
    O2_target = target_lab[1, :, :]
    O3_target = target_lab[2, :, :]
   
    H, W = pred_lab.shape[1], pred_lab.shape[2]
    k1 = create_low_pass_filter((H, W), 0.80).to(dev)  
    k2 = create_low_pass_filter((H, W), 0.45).to(dev)  
    k3 = create_low_pass_filter((H, W), 0.35).to(dev)  

    
    O1_pred_filtered = apply_filter(O1_pred, k1)
    O1_target_filtered = apply_filter(O1_target, k1)

    O2_pred_filtered = apply_filter(O2_pred, k2)
    O2_target_filtered = apply_filter(O2_target, k2)

    O3_pred_filtered = apply_filter(O3_pred, k3)
    O3_target_filtered = apply_filter(O3_target, k3)

    
    loss_fn = nn.MSELoss()

    
    loss_O1 = loss_fn(O1_pred_filtered, O1_target_filtered)
    loss_O2 = loss_fn(O2_pred_filtered, O2_target_filtered)
    loss_O3 = loss_fn(O3_pred_filtered, O3_target_filtered)

    
    total_loss = loss_O1 + loss_O2 + loss_O3
    return total_loss

def s_cielab_loss(final_amp, target_amp,disp_photo,dev):
    final_inten = final_amp.squeeze()**2
    target_inten = target_amp.squeeze()**2
    RGB2XYZ_matrix = [
    [1.2493,0.1147,0.8357],
    [0.4976,1.6037,0.0581],
    [0.0001,0.2151,4.2074]
    ]
    # RGB2XYZ_matrix = [
    # [0.4124564,0.3575761,0.1804375],
    # [0.2126729,0.7151522,0.072175],
    # [0.0193339, 0.119192,0.9503041]
    # ]
    RGB2XYZ = torch.tensor(RGB2XYZ_matrix).to(dev)
    brightness = 100
    predicted_XYZ = torch.einsum('ij,jab->iab', RGB2XYZ, final_inten*brightness)
    target_XYZ  = torch.einsum('ij,jab->iab', RGB2XYZ, target_inten*brightness)

    xyz2opp_matrix = [
    [0.2787336, 0.7218031, -0.106552],
    [-0.4487736, 0.2898056, 0.0771569],
    [0.0859513, -0.5899859, 0.5011089]
    ]
    xyz2opp = torch.tensor(xyz2opp_matrix).to(dev)
    opp2xyz = torch.inverse(xyz2opp).to(dev)
    predicted_OPP = torch.einsum('ij,jab->iab', xyz2opp, predicted_XYZ)
    target_OPP  = torch.einsum('ij,jab->iab', xyz2opp, target_XYZ)

    O1_pred = predicted_OPP[0, :, :]
    O2_pred = predicted_OPP[1, :, :]
    O3_pred = predicted_OPP[2, :, :]

    O1_target = target_OPP[0, :, :]
    O2_target = target_OPP[1, :, :]
    O3_target = target_OPP[2, :, :]

    H, W = predicted_OPP.shape[1], predicted_OPP.shape[2]
    k1 = create_low_pass_filter((H, W), 0.8).to(dev)  
    k2 = create_low_pass_filter((H, W), 0.45).to(dev) 
    k3 = create_low_pass_filter((H, W), 0.35).to(dev)  
    
    O1_pred_filtered = apply_filter(O1_pred, k1)
    O1_target_filtered = apply_filter(O1_target, k1)
    O2_pred_filtered = apply_filter(O2_pred, k2)
    O2_target_filtered = apply_filter(O2_target, k2)
    O3_pred_filtered = apply_filter(O3_pred, k3)
    O3_target_filtered = apply_filter(O3_target, k3)

    pred_opp_filtered = torch.stack((O1_pred_filtered, O2_pred_filtered, O3_pred_filtered), dim=0)
    target_opp_filtered = torch.stack((O1_target_filtered, O2_target_filtered, O3_target_filtered), dim=0)
    predicted_XYZ_filtered  = torch.einsum('ij,jab->iab', opp2xyz, pred_opp_filtered)
    target_XYZ_filtered   = torch.einsum('ij,jab->iab', opp2xyz, target_opp_filtered)
    predicted_XYZ_filtered = torch.max(predicted_XYZ_filtered,torch.zeros_like(predicted_XYZ_filtered))
    target_XYZ_filtered = torch.max(target_XYZ_filtered,torch.zeros_like(target_XYZ_filtered))
    pred_lab = xyz2lab(predicted_XYZ_filtered,dev)
    target_lab = xyz2lab(target_XYZ_filtered,dev)
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred_lab, target_lab)
    return loss