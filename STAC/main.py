import argparse
from envs.multigoal_env import MultiGoalEnv
from core import MaxEntrRL
import random
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from envs.max_entropy_env import MaxEntropyEnv
import numpy as np
import gym
import mujoco_py
from datetime import datetime
from utils import AttrDict
import glob
import timeit


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--env', type=str, default='Multigoal', choices=['Multigoal', 'max-entropy-v0', 'Multigoal', 'Hopper-v2', 'Ant-v2', 'Walker2d-v2', 'Humanoid-v2'])
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--actor', type=str, default='svgd_nonparam', choices=['sac', 'svgd_sql', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'diffusion'])
    ######networks
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l_critic', type=int, default=2)
    parser.add_argument('--l_actor', type=int, default=3)
    parser.add_argument('--critic_activation', type=object, default=torch.nn.ELU)
    parser.add_argument('--actor_activation', type=object, default=torch.nn.ELU)    
    ######RL 
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--replay_size', type=int, default=1e6)
    parser.add_argument('--max_experiment_steps', type=float, default=3e4)
    parser.add_argument('--exploration_steps', type=int, default=10000, help="pure exploration at the beginning of the training")

    parser.add_argument('--num_test_episodes', type=int, default=50)
    parser.add_argument('--stats_steps_freq', type=int, default=400)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=30)
    ######optim 
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=500)
    ######sac
    parser.add_argument('--sac_test_deterministic', type=int, default=1)
    ######sql
    parser.add_argument('--sql_test_deterministic', type=int, default=1)
    ######svgd 
    parser.add_argument('--svgd_particles', type=int, default=10)
    parser.add_argument('--svgd_steps', type=int, default=10)
    parser.add_argument('--svgd_lr', type=float, default=0.01)
    parser.add_argument('--svgd_test_deterministic', type=int, default=0)
    parser.add_argument('--svgd_sigma_p0', type=float, default=0.1)
    parser.add_argument('--svgd_kernel_sigma', type=float, default=None)
    parser.add_argument('--svgd_adaptive_lr', type=int, default=None)
   
    # tensorboard
    parser.add_argument('--tensorboard_path', type=str, default='./runs/')
    parser.add_argument('--evaluation_data_path', type=str, default='./evaluation_data/')
    parser.add_argument('--fig_path', type=str, default='./STAC/multi_goal_plots_/')
    parser.add_argument('--plot', type=int, default=1)

    args = parser.parse_args()  
    args.sac_test_deterministic = bool(args.sac_test_deterministic)
    args.sql_test_deterministic = bool(args.sql_test_deterministic)
    args.svgd_test_deterministic = bool(args.svgd_test_deterministic)
    args.plot = bool(args.plot)
    args.svgd_adaptive_lr = bool(args.svgd_adaptive_lr)
    # print(args.sac_test_deterministic)
    # print(args.svgd_adaptive_lr)
    # import pdb; pdb.set_trace()
    ################# Best parameters for a specific thing #################

    debugging = False 
    if debugging:
        print('############################## DEBUGGING ###################################')
        args.exploration_steps = 0
        # args.exploration_steps = 100
        # args.update_after = 3000
        args.svgd_kernel_sigma = 5.
        args.svgd_adaptive_lr = False
        print('############################################################################')
    


    if args.actor == 'svgd_sql':
        args.lr_critic = 1e-2
        args.alpha = 1.

    if args.actor == 'sac':
        args.critic_activation = torch.nn.ReLU
        args.actor_activation = torch.nn.ReLU
        args.alpha = 0.2
    
    if args.env in ['Hopper-v2', 'Ant-v2', 'Walker2d-v2', 'Humanoid-v2']:
        args.fig_path = './STAC/mujoco_plots_/'
        
    ###########################################################
    # fix the seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # set number of thereads
    torch.set_num_threads(torch.get_num_threads())
    
    # get device
    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # actor arguments
    if (args.actor in ['svgd_nonparam','svgd_p0_pram','svgd_p0_kernel_pram']):
        actor_kwargs=AttrDict(num_svgd_particles=args.svgd_particles, num_svgd_steps=args.svgd_steps, 
            svgd_lr=args.svgd_lr, test_deterministic=args.svgd_test_deterministic, svgd_sigma_p0 = args.svgd_sigma_p0,
            batch_size=args.batch_size,  device=device, hidden_sizes=[args.hid]*args.l_actor, activation=args.actor_activation, 
            kernel_sigma=args.svgd_kernel_sigma, adaptive_lr=args.svgd_adaptive_lr)
    
    elif (args.actor == 'svgd_sql'):
        actor_kwargs=AttrDict(num_svgd_particles=args.svgd_particles, 
            svgd_lr=args.svgd_lr, test_deterministic=args.sql_test_deterministic, 
            batch_size=args.batch_size,  device=device, hidden_sizes=[args.hid]*args.l_actor, activation=args.actor_activation)
    elif (args.actor =='sac'):
        actor_kwargs=AttrDict(hidden_sizes=[args.hid]*args.l_actor, test_deterministic=args.sac_test_deterministic, device=device, activation=args.actor_activation)
    
    # Logging
    #
    project_name =  datetime.now().strftime("%b_%d_%Y_%H_%M_%S") + '_' + args.actor + '_' + args.env + '_alpha_'+str(args.alpha)+'_batch_size_'+str(args.batch_size) + '_lr_critic_' + str(args.lr_critic) + '_lr_actor_' + str(args.lr_actor) +'_activation_'+str(args.actor_activation)[-6:-2] + '_seed_' + str(args.seed)
    
    if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
        project_name += '_svgd_steps_'+str(args.svgd_steps)+'_svgd_particles_'+str(args.svgd_particles)+'_svgd_lr_'+str(args.svgd_lr) + '_svgd_sigma_p0_' + str(args.svgd_sigma_p0) + '_adaptive_' + str(args.svgd_adaptive_lr) + '_svgd_kernel_sigma_' + str(args.svgd_kernel_sigma)


    os.makedirs(args.tensorboard_path + project_name)
    # os.makedirs(args.evaluation_data_path + project_name)

    tb_logger = SummaryWriter(args.tensorboard_path + project_name)

    # RL args
    RL_kwargs = AttrDict(stats_steps_freq=args.stats_steps_freq,gamma=args.gamma,
        alpha=args.alpha,replay_size=int(args.replay_size),exploration_steps=args.exploration_steps,update_after=args.update_after,
        update_every=args.update_every, num_test_episodes=args.num_test_episodes, plot=args.plot, max_steps = args.max_steps, 
        max_experiment_steps=int(args.max_experiment_steps), evaluation_data_path = args.evaluation_data_path + project_name)

    # optim args
    optim_kwargs = AttrDict(polyak=args.polyak,lr_critic=args.lr_critic, lr_actor=args.lr_actor,batch_size=args.batch_size)

    # critic args
    critic_kwargs = AttrDict(hidden_sizes=[args.hid]*args.l_critic, activation=args.critic_activation)

    # stac
    if args.env =='Multigoal':
        env_fn = MultiGoalEnv(max_steps=RL_kwargs.max_steps)
    elif args.env == 'max-entropy-v0':
        env_fn = MaxEntropyEnv(max_steps=RL_kwargs.max_steps)
    else: 
        env_fn = gym.make(args.env)
    
        

    if not os.path.exists(args.fig_path + project_name) and RL_kwargs.plot:
        os.makedirs(args.fig_path + project_name)
    else:
        files = glob.glob(args.fig_path + project_name + '/*')
        [os.remove(file) for file in files]

    stac=MaxEntrRL(env_fn, env=args.env, actor=args.actor, device=device, 
        critic_kwargs=critic_kwargs, actor_kwargs=actor_kwargs,
        RL_kwargs=RL_kwargs, optim_kwargs=optim_kwargs,tb_logger=tb_logger, fig_path=args.fig_path +  project_name)

    start = timeit.default_timer()
    stac.forward()
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    print() 
    print(project_name)