#!/bin/bash
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 10 --gpu_id 0 >> stac_multigoal_alpha_02.log &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 0.4 --svgd_kernel_sigma 10 --gpu_id 1 >> stac_multigoal_alpha_04.log &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 0.6 --svgd_kernel_sigma 10 --gpu_id 2 >> stac_multigoal_alpha_06.log &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 0.8 --svgd_kernel_sigma 10 --gpu_id 3 >> stac_multigoal_alpha_08.log 

python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 10 --gpu_id 0 >> stac_multigoal_alpha_02.log &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.4 --svgd_kernel_sigma 10 --gpu_id 1 >> stac_multigoal_alpha_04.log &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.6 --svgd_kernel_sigma 10 --gpu_id 2 >> stac_multigoal_alpha_06.log &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.8 --svgd_kernel_sigma 10 --gpu_id 3 >> stac_multigoal_alpha_08.log 

python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 50 --gpu_id 0 >> stac_multigoal_alpha_02.log &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.4 --svgd_kernel_sigma 50 --gpu_id 1 >> stac_multigoal_alpha_04.log &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.6 --svgd_kernel_sigma 50 --gpu_id 2 >> stac_multigoal_alpha_06.log &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.8 --svgd_kernel_sigma 50 --gpu_id 3 >> stac_multigoal_alpha_08.log 

python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 100 --gpu_id 0 >> stac_multigoal_alpha_02.log &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.4 --svgd_kernel_sigma 100 --gpu_id 1 >> stac_multigoal_alpha_04.log &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.6 --svgd_kernel_sigma 100 --gpu_id 2 >> stac_multigoal_alpha_06.log &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.8 --svgd_kernel_sigma 100 --gpu_id 3 >> stac_multigoal_alpha_08.log 
