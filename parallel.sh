#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ./STAC/main.py --env Hopper-v2 --max_experiment_steps 1e6 &
CUDA_VISIBLE_DEVICES=1 python ./STAC/main.py --env HalfCheetah-v2 --max_experiment_steps 3e6 &
CUDA_VISIBLE_DEVICES=2 python ./STAC/main.py --env Ant-v2 --max_experiment_steps 3e6 &
CUDA_VISIBLE_DEVICES=3 python ./STAC/main.py --env Walker2d-v2 --max_experiment_steps 1e6 
