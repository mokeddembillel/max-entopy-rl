#!/bin/bash
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 0.01 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 0.01 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 0.01 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 0.01 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 0.01 --gpu_id 3 &

python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 0.1 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 0.1 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 0.1 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 0.1 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 0.1 --gpu_id 3 

python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 0.5 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 0.5 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 0.5 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 0.5 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 0.5 --gpu_id 3 &

python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 0.7 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 0.7 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 0.7 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 0.7 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 0.7 --gpu_id 3 

python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 1.0 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 1.0 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 1.0 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 1.0 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 1.0 --gpu_id 3 &

python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 5.0 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 5.0 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 5.0 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 5.0 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 0 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 5.0 --gpu_id 3 

###########################################################################################

python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 0.01 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 0.01 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 0.01 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 0.01 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 0.01 --gpu_id 3 &

python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 0.1 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 0.1 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 0.1 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 0.1 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 0.1 --gpu_id 3 

python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 0.5 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 0.5 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 0.5 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 0.5 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 0.5 --gpu_id 3 &

python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 0.7 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 0.7 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 0.7 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 0.7 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 0.7 --gpu_id 3 

python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 1.0 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 1.0 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 1.0 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 1.0 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 1.0 --gpu_id 3 &

python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 0.2 --svgd_kernel_sigma 5.0 --gpu_id 0 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 1.0 --svgd_kernel_sigma 5.0 --gpu_id 1 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 3.0 --svgd_kernel_sigma 5.0 --gpu_id 2 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 5.0 --svgd_kernel_sigma 5.0 --gpu_id 3 &
python ./STAC/main.py --svgd_adaptive_lr 1 --svgd_lr 0.1 --alpha 7.0 --svgd_kernel_sigma 5.0 --gpu_id 3 