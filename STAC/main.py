import argparse
from envs.multigoal_env import MultiGoalEnv
from core import MaxEntrRL
import random
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from envs.multigoal_env import MultiGoalEnv
from envs.max_entropy_env import MaxEntropyEnv
import numpy as np
import gym
from datetime import datetime
from utils import AttrDict
import glob


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    
    parser.add_argument('--env', type=str, default='Multigoal', choices=['HalfCheetah-v2', 'max-entropy-v0', 'Multigoal'])
    parser.add_argument('--seed', '-s', type=int, default=0)
    #parser.add_argument('--actor', type=str, default='svgd_nonparam', choices=['sac', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'diffusion'])
    parser.add_argument('--actor', type=str, default='svgd_sql', choices=['sac', 'svgd_sql', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'diffusion'])
    ######networks
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l_critic', type=int, default=2)
    parser.add_argument('--l_actor', type=int, default=3)
    ######RL 
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--replay_size', type=int, default=1e6)

    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--exploration_episodes', type=int, default=90)
    #parser.add_argument('--exploration_episodes', type=int, default=200)
    parser.add_argument('--num_test_episodes', type=int, default=50)
    parser.add_argument('--stats_episode_freq', type=int, default=5)
    parser.add_argument('--update_after', type=int, default=3000)
    # parser.add_argument('--update_after', type=int, default=50000)
    #parser.add_argument('--update_every', type=int, default=100)
    parser.add_argument('--update_every', type=int, default=50)
    #parser.add_argument('--max_ep_len', type=int, default=1000)
    # parser.add_argument('--max_ep_len', type=int, default=500)
    ######optim 
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=500)
    ######sac
    parser.add_argument('--sac_test_deterministic', type=bool, default=False)
    ######svgd 
    parser.add_argument('--svgd_particles', type=int, default=10)
    parser.add_argument('--svgd_steps', type=int, default=5)
    parser.add_argument('--svgd_lr', type=float, default=0.1)
    parser.add_argument('--svgd_test_deterministic', type=bool, default=True)
    parser.add_argument('--sql_test_deterministic', type=bool, default=False)
    # tensorboard
    parser.add_argument('--tensorboard_path', type=str, default='./runs/')
    parser.add_argument('--fig_path', type=str, default='./STAC/multi_goal_plots_/')
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--critic_activation', type=object, default=torch.nn.ReLU)
    parser.add_argument('--actor_activation', type=object, default=torch.nn.Tanh)


    args = parser.parse_args()    
    
    ################# Best parameters for SQL #################
    if args.actor == 'svgd_sql':
        args.lr_critic = 1e-2
        args.alpha = 1.
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
            svgd_lr=args.svgd_lr, test_deterministic=args.svgd_test_deterministic, 
            batch_size=args.batch_size,  device=device, hidden_sizes=[args.hid]*args.l_actor, activation=args.actor_activation)
    
    elif (args.actor == 'svgd_sql'):
        actor_kwargs=AttrDict(num_svgd_particles=args.svgd_particles, 
            svgd_lr=args.svgd_lr, test_deterministic=args.sql_test_deterministic, 
            batch_size=args.batch_size,  device=device, hidden_sizes=[args.hid]*args.l_actor, activation=args.actor_activation)
    elif (args.actor =='sac'):
        actor_kwargs=AttrDict(hidden_sizes=[args.hid]*args.l_actor, test_deterministic=args.sac_test_deterministic, device=device, activation=args.actor_activation)
    
    # Logging
    #
    project_name =  datetime.now().strftime("%b_%d_%Y_%H_%M_%S") + '_' + args.actor + '_' + args.env + '_alpha_'+str(args.alpha)+'_batch_size_'+str(args.batch_size) + '_lr_critic_' + str(args.lr_critic) + '_lr_actor_' + str(args.lr_actor) +'_activation_'+str(args.actor_activation)[-6:-2]
    
    if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
        project_name += '_svgd_steps_'+str(args.svgd_steps)+'_svgd_particles_'+str(args.svgd_particles)+'_svgd_lr_'+str(args.svgd_lr)

    
    os.makedirs(args.tensorboard_path + project_name)
    
    tb_logger = SummaryWriter(args.tensorboard_path + project_name)

    # RL args
    RL_kwargs = AttrDict(num_episodes=args.num_episodes,stats_episode_freq=args.stats_episode_freq,gamma=args.gamma,
        alpha=args.alpha,replay_size=int(args.replay_size),exploration_episodes=args.exploration_episodes,update_after=args.update_after,
        update_every=args.update_every, num_test_episodes=args.num_test_episodes, plot=args.plot)

    # optim args
    optim_kwargs = AttrDict(polyak=args.polyak,lr_critic=args.lr_critic, lr_actor=args.lr_actor,batch_size=args.batch_size)
    
    # critic args
    critic_kwargs = AttrDict(hidden_sizes=[args.hid]*args.l_critic, activation=args.critic_activation)

    # stac
    if args.env =='Multigoal':
        env_fn = MultiGoalEnv
    elif args.env == 'max-entropy-v0':
        env_fn = MaxEntropyEnv
    
        

    if not os.path.exists(args.fig_path + project_name) and RL_kwargs.plot:
        os.makedirs(args.fig_path + project_name)
    else:
        files = glob.glob(args.fig_path + project_name + '/*')
        [os.remove(file) for file in files]

    stac=MaxEntrRL(env_fn, env=args.env, actor=args.actor, device=device, 
        critic_kwargs=critic_kwargs, actor_kwargs=actor_kwargs,
        RL_kwargs=RL_kwargs, optim_kwargs=optim_kwargs,tb_logger=tb_logger, fig_path=args.fig_path +  project_name)

    stac.forward()

