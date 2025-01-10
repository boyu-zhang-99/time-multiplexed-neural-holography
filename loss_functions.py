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
# from utils_color import *
import utils

import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import raft_large
from torchvision.transforms.functional import to_tensor, resize
## TODO: Add more color loss functions
model = raft_large(pretrained=True).to("cuda").eval()

def cvvdp_loss(final_amp, target_amp, metric,disp_photo, dev):
    I_test = final_amp.squeeze()**2 
    I_ref = target_amp.squeeze()**2 
    brightness = 100
    I_test_physical = I_test*brightness
    I_ref_physical = I_ref*brightness
    loss = metric.loss( I_test_physical, I_ref_physical, dim_order="CHW") + 0.001 * s_cielab_loss(final_amp, target_amp,disp_photo,dev)
    return loss

def cvvdp_loss_video(final_amp, target_amp, metric,disp_photo, dev):
    I_test = final_amp.squeeze()**2 
    I_ref = target_amp.squeeze()**2 
    brightness = 100
    I_test_physical = I_test*brightness
    I_ref_physical = I_ref*brightness
    #loss = metric.loss( I_test_physical, I_ref_physical, dim_order="FCHW",frames_per_second=30)+ 0.001 * s_cielab_loss_video(final_amp, target_amp,disp_photo,dev)
    loss = metric.loss( I_test_physical, I_ref_physical, dim_order="FCHW",frames_per_second=30)
    return loss

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

def cielab_loss_video(final_amp, target_amp, disp_photo, dev):
    final_inten = final_amp.squeeze()**2 
    target_inten = target_amp.squeeze()**2 

    RGB2XYZ_matrix = [
    [1.2493,0.1147,0.8357],
    [0.4976,1.6037,0.0581],
    [0.0001,0.2151,4.2074]
    ]
    RGB2XYZ = torch.tensor(RGB2XYZ_matrix).to(dev)
    Brightness = 100
    test_xyz = torch.einsum('ij,cjab->ciab', RGB2XYZ, final_inten*Brightness)
    ref_xyz = torch.einsum('ij,cjab->ciab', RGB2XYZ, target_inten*Brightness)

    test_lab = xyz2lab(test_xyz)
    ref_lab = xyz2lab(ref_xyz)

    loss_fn = nn.MSELoss()
    loss = loss_fn(test_lab, ref_lab)

    return loss

def xyz2lab(video_tensor):
    X_n, Y_n, Z_n = 95.047, 100.000, 108.883
    X, Y, Z = video_tensor[:, 0, :, :], video_tensor[:, 1, :, :], video_tensor[:, 2, :, :]
    X = X / X_n
    Y = Y / Y_n
    Z = Z / Z_n
    delta = 6 / 29
    def f(t):
        return torch.where(t > delta ** 3, t ** (1/3), (t / (3 * delta ** 2)) + (4 / 29))
    fX = f(X)
    fY = f(Y)
    fZ = f(Z)
    L = (116 * fY) - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)
    lab_tensor = torch.stack([L, a, b], dim=1)  
    return lab_tensor
def preprocess_image(image, target_size=(520, 960)):
    image = resize(image, target_size)
    return to_tensor(image).unsqueeze(0)

def warp_frame(frame_t1, flow_t_to_t1):
    """
    Warps frame_t1 to the coordinate system of frame_t using the optical flow from t to t+1.

    Args:
        frame_t1 (torch.Tensor): The frame at time t+1 with shape (B, C, H, W).
        flow_t_to_t1 (torch.Tensor): Optical flow from t to t+1 with shape (B, 2, H, W).
                                    The flow is expected to have values in pixels.

    Returns:
        torch.Tensor: Warped frame_t1 in the coordinate system of frame_t with shape (B, C, H, W).
    """
    B, C, H, W = frame_t1.size()

    # Create a mesh grid of pixel coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack((x_coords, y_coords), dim=-1).float().to(frame_t1.device)  # Shape (H, W, 2)
    
    # Normalize grid coordinates to range [-1, 1] for grid_sample
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1
    grid = grid.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # Shape (B, 2, H, W)
    
    # Add optical flow to grid
    flow_grid = grid + flow_t_to_t1
    
    # Rearrange for grid_sample
    flow_grid = flow_grid.permute(0, 2, 3, 1)  # Shape (B, H, W, 2)
    
    # Warp frame_t1 using the flow grid
    warped_frame = F.grid_sample(frame_t1, flow_grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return warped_frame

def l2_loss_video(final_amp, target_amp,target_idx):
    final_inten = final_amp.squeeze()**2
    target_inten = target_amp.squeeze()**2
    loss_fn = nn.MSELoss()
    loss = loss_fn(final_inten, target_inten)

    # for i in range(1,3):
    #     # image1_path = "/home/billzhang/Desktop/time-multiplexed-neural-holography/data/2d/00{}.png".format(i)
    #     # image2_path = "/home/billzhang/Desktop/time-multiplexed-neural-holography/data/2d/00{}.png".format(i+1)
    #     # frame1 = Image.open(image1_path)
    #     # frame2 = Image.open(image2_path)
    #     # frame1_tensor = resize(to_tensor(frame1), [520, 960]).unsqueeze(0).to("cuda")
    #     # frame2_tensor = resize(to_tensor(frame2), [520, 960]).unsqueeze(0).to("cuda")
        
    #     target_tensor_2 = resize(target_amp[i,:,:,:], [800, 1280]).unsqueeze(0)
    #     target_tensor_1 = resize(target_amp[i-1,:,:,:], [800, 1280]).unsqueeze(0)
    #     recon_tensor_2 = resize(final_amp[i,:,:,:], [800, 1280]).unsqueeze(0)
    #     recon_tensor_1 = resize(final_amp[i-1,:,:,:], [800, 1280]).unsqueeze(0)
    #     with torch.no_grad():
    #         flow = model(target_tensor_1, target_tensor_2)[0]
    #     warped_frame = warp_frame(recon_tensor_2, flow)
    #     temporal_diff = warped_frame - recon_tensor_1
    #     temporal_loss = torch.mean(temporal_diff ** 2)
    #     loss = loss + temporal_loss

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

def s_cielab_loss(final_amp, target_amp,disp_photo,dev):
    final_inten = final_amp.squeeze()**2
    target_inten = target_amp.squeeze()**2
    RGB2XYZ_matrix = [
    [1.2493,0.1147,0.8357],
    [0.4976,1.6037,0.0581],
    [0.0001,0.2151,4.2074]
    ]
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

def s_cielab_loss_video(final_amp, target_amp, disp_photo, dev):
    F, C, H, W = final_amp.shape  # Extract dimensions

    final_inten = final_amp ** 2
    target_inten = target_amp ** 2

    RGB2XYZ_matrix = [
        [1.2493, 0.1147, 0.8357],
        [0.4976, 1.6037, 0.0581],
        [0.0001, 0.2151, 4.2074]
    ]
    RGB2XYZ = torch.tensor(RGB2XYZ_matrix).to(dev)
    brightness = 100

    predicted_XYZ = torch.einsum('ij,fjhw->fihw', RGB2XYZ, final_inten * brightness)
    target_XYZ = torch.einsum('ij,fjhw->fihw', RGB2XYZ, target_inten * brightness)

    xyz2opp_matrix = [
        [0.2787336, 0.7218031, -0.106552],
        [-0.4487736, 0.2898056, 0.0771569],
        [0.0859513, -0.5899859, 0.5011089]
    ]
    xyz2opp = torch.tensor(xyz2opp_matrix).to(dev)
    opp2xyz = torch.inverse(xyz2opp).to(dev)

    predicted_OPP = torch.einsum('ij,fjhw->fihw', xyz2opp, predicted_XYZ)
    target_OPP = torch.einsum('ij,fjhw->fihw', xyz2opp, target_XYZ)

    O1_pred = predicted_OPP[:, 0, :, :]
    O2_pred = predicted_OPP[:, 1, :, :]
    O3_pred = predicted_OPP[:, 2, :, :]

    O1_target = target_OPP[:, 0, :, :]
    O2_target = target_OPP[:, 1, :, :]
    O3_target = target_OPP[:, 2, :, :]

    k1 = create_low_pass_filter((H, W), 0.8).to(dev)
    k2 = create_low_pass_filter((H, W), 0.45).to(dev)
    k3 = create_low_pass_filter((H, W), 0.35).to(dev)

    O1_pred_filtered = torch.stack([apply_filter(O1_pred[f], k1) for f in range(F)], dim=0)
    O1_target_filtered = torch.stack([apply_filter(O1_target[f], k1) for f in range(F)], dim=0)
    O2_pred_filtered = torch.stack([apply_filter(O2_pred[f], k2) for f in range(F)], dim=0)
    O2_target_filtered = torch.stack([apply_filter(O2_target[f], k2) for f in range(F)], dim=0)
    O3_pred_filtered = torch.stack([apply_filter(O3_pred[f], k3) for f in range(F)], dim=0)
    O3_target_filtered = torch.stack([apply_filter(O3_target[f], k3) for f in range(F)], dim=0)

    pred_opp_filtered = torch.stack((O1_pred_filtered, O2_pred_filtered, O3_pred_filtered), dim=1)
    target_opp_filtered = torch.stack((O1_target_filtered, O2_target_filtered, O3_target_filtered), dim=1)

    predicted_XYZ_filtered = torch.einsum('ij,fjhw->fihw', opp2xyz, pred_opp_filtered)
    target_XYZ_filtered = torch.einsum('ij,fjhw->fihw', opp2xyz, target_opp_filtered)

    predicted_XYZ_filtered = torch.max(predicted_XYZ_filtered, torch.zeros_like(predicted_XYZ_filtered))
    target_XYZ_filtered = torch.max(target_XYZ_filtered, torch.zeros_like(target_XYZ_filtered))

    pred_lab = xyz2lab(predicted_XYZ_filtered)
    target_lab = xyz2lab(target_XYZ_filtered)

    loss_fn = nn.MSELoss()
    loss = loss_fn(pred_lab, target_lab)



    return loss