"""
Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Time-multiplexed Neural Holography:
A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators
S. Choi*, M. Gopakumar*, Y. Peng, J. Kim, Matthew O'Toole, G. Wetzstein.
SIGGRAPH 2022
-----

$ python main.py --lr=0.01 --num_iters=10000 --num_frames=8 --quan_method=gumbel-softmax

"""
import os
import json
import torch
import imageio
import configargparse
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import utils
import params
import algorithms as algs
import quantization as q
import numpy as np
import image_loader as loaders
from torch.utils.data import DataLoader
import props.prop_model as prop_model
import props.prop_physical as prop_physical
from hw.phase_encodings import phase_encoding
from torchvision.utils import save_image
import cv2

from pprint import pprint

#import wx
#wx.DisableAsserts()
    
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():
    # Command line argument processing / Parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False,
          is_config_file=True, help='Path to config file.')
    params.add_parameters(p, 'eval')
    opt = params.set_configs(p.parse_args())
    params.add_lf_params(opt)
    dev = torch.device('cuda')

    run_id = params.run_id(opt)
    # path to save out optimized phases
    out_path = os.path.join(opt.out_path, run_id)
    print(f'  - out_path: {out_path}')

    # Tensorboard
    summaries_dir = os.path.join(out_path, 'summaries')
    utils.cond_mkdir(summaries_dir)
    writer = SummaryWriter(summaries_dir)

    # Write opt to experiment folder
    utils.write_opt(vars(p.parse_args()), out_path)

    # Propagations
    camera_prop = None
    if opt.citl:
        camera_prop = prop_physical.PhysicalProp(*(params.hw_params(opt)), shutter_speed=opt.shutter_speed).to(dev)
        camera_prop.calibrate_total_laser_energy() # important!
    sim_prop = prop_model.model(opt)
    sim_prop.eval()

    # Look-up table of SLM
    if opt.use_lut:
        lut = q.load_lut(sim_prop, opt)
    else:
        lut = None
    quantization = q.quantization(opt, lut)

    # Algorithm
    algorithm = algs.load_alg(opt.method, mem_eff=opt.mem_eff)

    # Loader
    if ',' in opt.data_path:
        opt.data_path = opt.data_path.split(',')
    img_loader = loaders.TargetLoader(shuffle=opt.random_gen,
                                      vertical_flips=opt.random_gen,
                                      horizontal_flips=opt.random_gen,
                                      is_hdr=opt.hdr,
                                      scale_vd_range=False, **opt)
    target_amp_tensors = []
    target_idx_list = []
    for i, target in enumerate(img_loader):
        batch = opt.batch_size
        target_amp_sub, target_mask, target_idx = target
        target_amp_sub = target_amp_sub.to(dev).detach()
        
        if target_mask is not None:
            target_mask = target_mask.to(dev).detach()
        if len(target_amp_sub.shape) < 4:
            # target_amp_sub = target_amp_sub.unsqueeze(0)
            target_amp_tensors.append(target_amp_sub)
            target_idx_list.append(target_idx)
        print(f'  - run phase optimization for {target_idx}th image ...')

        if (i+1)%batch == 0 :
            target_amp = torch.stack(target_amp_tensors, dim=0)
            print(target_amp.shape)
            

            if opt.random_gen:  # random parameters for dataset generation
                img_files = os.listdir(out_path)
                img_files = [f for f in img_files if f.endswith('.png')]
                if len(img_files) > opt.num_data: # generate enough data
                    break
                print("Num images: ", len(img_files), " (max: ", opt.num_data)
                opt.num_frames, opt.num_iters, opt.init_phase_range, \
                target_range, opt.lr, opt.eval_plane_idx, \
                opt.quan_method, opt.reg_lf_var = utils.random_gen(**opt)
                sim_prop = prop_model.model(opt)
                quantization = q.quantization(opt, lut)
                target_amp *= target_range
                if opt.reg_lf_var > 0.0 and isinstance(sim_prop, prop_model.CNNpropCNN):
                    opt.num_frames = min(opt.num_frames, 4)

            out_path_idx = f'{opt.out_path}_{target_idx}'

            # initial slm phase
            init_phase = utils.init_phase(opt.init_phase_type, target_amp, dev, opt)
            # run algorithm
            results = algorithm(init_phase, target_amp, target_mask, target_idx,
                                forward_prop=sim_prop, camera_prop=camera_prop,
                                writer=writer, quantization=quantization, optimize_s=opt.opt_s,batch_s=opt.batch_size,
                                out_path_idx=out_path_idx, **opt)
                                
            # optimized slm phase
            final_phase = results['final_phase']
            recon_amp = results['recon_amp']
            target_amp = results['target_amp']

            # encoding for SLM & save it out
            if opt.random_gen:
                # decompose it into several 1-bit phases
                for k, final_phase_1bit in enumerate(final_phase):
                    phase_out = phase_encoding(final_phase_1bit.unsqueeze(0), opt.slm_type)
                    phase_out_path = os.path.join(out_path, f'{target_idx}_{opt.num_iters}{k}.png')
                    imageio.imwrite(phase_out_path, phase_out)
            else:
                phase_out_batch = phase_encoding(final_phase, opt.slm_type)
                recon_amp_batch, target_amp_batch = recon_amp.detach().cpu().numpy(), target_amp.detach().cpu().numpy()


                for idx in range(batch):
                    phase_out = phase_out_batch[idx,:,:]
                    recon_amp = recon_amp_batch[idx,:,:,:]
                    target_amp = target_amp_batch[idx,:,:,:]
                    target_idx = target_idx_list[idx]
                    # save final phase and intermediate phases
                    if phase_out is not None:
                        phase_out_path = os.path.join(out_path, f'{target_idx}_phase.png')
                        # imageio.imwrite(phase_out_path, phase_out.transpose(1,2,0))
                        # imageio.imwrite(phase_out_path, phase_out.squeeze(0))

                    if opt.save_images:
                        if not opt.hdr:
                            recon_out_path = os.path.join(out_path, f'{target_idx}_recon.png')
                            target_out_path = os.path.join(out_path, f'{target_idx}_target.png')
                            
                            if opt.channel is None:
                                recon_amp = recon_amp.transpose(1, 2, 0)
                                target_amp = target_amp.transpose(1, 2, 0)

                            recon_out = utils.srgb_lin2gamma(np.clip(recon_amp**2, 0, 1)) # linearize and gamma
                            target_out = utils.srgb_lin2gamma(np.clip(target_amp**2, 0, 1)) # linearize and gamma

                            imageio.imwrite(recon_out_path, (recon_out * 255).astype(np.uint8))
                            # imageio.imwrite(target_out_path, (target_out * 255).astype(np.uint8))
                        else:
                            recon_out_path = os.path.join(out_path, f'{target_idx}_recon.exr')
                            target_out_path = os.path.join(out_path, f'{target_idx}_target.exr')
                            if opt.channel is None:
                                recon_amp = recon_amp.transpose(1, 2, 0)
                                target_amp = target_amp.transpose(1, 2, 0)
                            recon_out = np.clip(recon_amp**2, 0, 1)
                            target_out = np.clip(target_amp**2, 0, 1)
                            cv2.imwrite(recon_out_path, cv2.cvtColor(recon_out/np.percentile(recon_out, 95), cv2.COLOR_RGB2BGR))
                            cv2.imwrite(target_out_path, cv2.cvtColor(target_out/np.percentile(target_out, 95), cv2.COLOR_RGB2BGR))
                            # cv2.imwrite(recon_out_path, cv2.cvtColor(recon_out*4000, cv2.COLOR_RGB2BGR))
                            # cv2.imwrite(target_out_path, cv2.cvtColor(target_out*4000, cv2.COLOR_RGB2BGR))
            target_amp_tensors = []
            target_idx_list = []
    if camera_prop is not None:
        camera_prop.disconnect()

if __name__ == "__main__":
    main()
