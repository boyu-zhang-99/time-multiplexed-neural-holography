#!/bin/bash
python main.py -c=configs_2d.txt --save_images --serial_two_prop_off --num_iters=5000 --lr=0.02 --uniform_nbits=4 --quan_method=nn --opt_s --loss_fnc=amp_l2_loss 
python main.py -c=configs_2d.txt --save_images --serial_two_prop_off --num_iters=5000 --lr=0.02 --uniform_nbits=4 --quan_method=nn --opt_s --loss_fnc=cielab_loss
python main.py -c=configs_2d.txt --save_images --serial_two_prop_off --num_iters=5000 --lr=0.02 --uniform_nbits=4 --quan_method=nn --opt_s --loss_fnc=s_cielab_loss
python main.py -c=configs_2d.txt --save_images --serial_two_prop_off --num_iters=5000 --lr=0.02 --uniform_nbits=4 --quan_method=nn --opt_s --loss_fnc=cvvdp_loss

python main.py -c=configs_2d.txt --save_images --serial_two_prop_off --num_iters=5000 --lr=0.02 --uniform_nbits=4 --quan_method=nn_sigmoid --loss_fnc=s_cielab_loss --prop_dist=0.075 --use_lut
