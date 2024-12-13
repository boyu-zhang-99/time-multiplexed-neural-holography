#!/bin/bash
# python main.py -c=configs_2d.txt --save_images --serial_two_prop_off --num_iters=5000 --lr=0.02 --uniform_nbits=8 --quan_method=nn --opt_s --batch_size=1 --loss_fnc=cvvdp_loss
python main.py -c=configs_2d.txt --save_images --serial_two_prop_off --num_iters=5000 --lr=0.02 --uniform_nbits=8 --quan_method=nn --opt_s --batch_size=1 --loss_fnc=cielab_loss
python main.py -c=configs_2d.txt --save_images --serial_two_prop_off --num_iters=5000 --lr=0.02 --uniform_nbits=8 --quan_method=nn --opt_s --batch_size=1 --loss_fnc=s_cielab_loss
python main.py -c=configs_2d.txt --save_images --serial_two_prop_off --num_iters=5000 --lr=0.02 --uniform_nbits=8 --quan_method=nn --opt_s --batch_size=1 --loss_fnc=amp_l2_loss