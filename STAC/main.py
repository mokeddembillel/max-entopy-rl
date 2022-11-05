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
    parser.add_argument('--env', type=str, default='max-entropy-v0', choices=['Multigoal', 'max-entropy-v0', 'Multigoal', 'Hopper-v2', 'Ant-v2', 'Walker2d-v2', 'Humanoid-v2', 'HalfCheetah-v2'])
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--actor', type=str, default='sac', choices=['sac', 'svgd_sql', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'diffusion'])
    ###### networks
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l_critic', type=int, default=2)
    parser.add_argument('--l_actor', type=int, default=3)
    parser.add_argument('--critic_activation', type=object, default=torch.nn.ELU)
    parser.add_argument('--actor_activation', type=object, default=torch.nn.ELU)    
    ###### RL 
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=5)
    parser.add_argument('--replay_size', type=int, default=1e6)
    parser.add_argument('--load_replay', type=int, default=0)
    parser.add_argument('--replay_path', type=str, default='./STAC/buffers_/')
    parser.add_argument('--max_experiment_steps', type=float, default=3e4)
    parser.add_argument('--exploration_steps', type=int, default=10000, help="pure exploration at the beginning of the training")

    parser.add_argument('--num_test_episodes', type=int, default=20)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=30)
    ###### optim 
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=500)
    ###### sac
    parser.add_argument('--sac_test_deterministic', type=int, default=0)
    ###### sql
    parser.add_argument('--sql_test_deterministic', type=int, default=0)
    ###### svgd 
    parser.add_argument('--svgd_particles', type=int, default=20)
    parser.add_argument('--svgd_steps', type=int, default=10)
    parser.add_argument('--svgd_lr', type=float, default=0.05)
    parser.add_argument('--svgd_test_deterministic', type=int, default=0)
    parser.add_argument('--svgd_sigma_p0', type=float, default=0.1)
    parser.add_argument('--svgd_kernel_sigma', type=float, default=None)
    parser.add_argument('--svgd_adaptive_lr', type=int, default=0)
   
    # tensorboard
    parser.add_argument('--tensorboard_path', type=str, default='./runs/')
    parser.add_argument('--evaluation_data_path', type=str, default='./evaluation_data/')
    parser.add_argument('--fig_path', type=str, default='./STAC/multi_goal_plots_/')
    parser.add_argument('--plot', type=int, default=1)
    parser.add_argument('--plot_format', type=str, default='png', choices=['png', 'jpeg', 'pdf'])
    parser.add_argument('--stats_steps_freq', type=int, default=400) 
    parser.add_argument('--collect_stats_after', type=int, default=400) 
    
    parser.add_argument('--test_time', type=int, default=1) 
    parser.add_argument('--model_path', type=str, default='./evaluation_data/sac')


    ###################################################################################
    parser.add_argument('--debugging', type=int, default=0)
    ###################################################################################
    args = parser.parse_args()  
    args.sac_test_deterministic = bool(args.sac_test_deterministic)
    args.sql_test_deterministic = bool(args.sql_test_deterministic)
    args.svgd_test_deterministic = bool(args.svgd_test_deterministic)
    args.plot = bool(args.plot)
    args.svgd_adaptive_lr = bool(args.svgd_adaptive_lr)
    args.debugging = bool(args.debugging)
    args.load_replay = bool(args.load_replay)
    args.test_time = bool(args.test_time)
    # print(args.sac_test_deterministic)
    # print(args.svgd_adaptive_lr)
    # import pdb; pdb.set_trace()
    ################# Best parameters for a specific thing #################
    
    
    if args.test_time:
        print('############################## TEST TIME ###################################')
        print('############################################################################')
        print('############################################################################')

    if args.actor == 'svgd_sql':
        args.lr_critic = 1e-2
        args.alpha = 1.

    if args.actor == 'sac':
        args.critic_activation = torch.nn.ReLU
        args.actor_activation = torch.nn.ReLU
        # args.alpha = 0.2
    
    if args.env in ['Hopper-v2', 'Ant-v2', 'Walker2d-v2', 'Humanoid-v2', 'HalfCheetah-v2']:
        args.fig_path = './STAC/mujoco_plots_/'
    elif args.env == 'max-entropy-v0':
        args.stats_steps_freq = 1000
        args.max_steps = 500
        # args.num_test_episodes = 2
        args.fig_path = './STAC/max_entropy_plots_/'
    
    if args.debugging:
        print('############################## DEBUGGING ###################################')
        args.exploration_steps = 20000000
        # args.actor = 'svgd_sql'
        # args.svgd_lr = 0.1
        args.max_experiment_steps = 5000000
        # args.exploration_steps = 100
        args.update_after = 10000000
        # args.stats_steps_freq = 11
        # args.num_test_episodes = 1
        # args.max_steps = 500
        # args.collect_stats_after = 0

        args.update_after = 1000000
        args.stats_steps_freq = 11
        args.num_test_episodes = 1
        args.max_steps = 500
        args.collect_stats_after = 2000000

        # args.svgd_kernel_sigma = 5.
        # args.svgd_adaptive_lr = False
        print('############################################################################')
        
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
    # device = torch.device('cpu')

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
    if args.test_time:
        project_name =  datetime.now().strftime("%b_%d_%Y_%H_%M_%S")+ '_initstate_zero' + '_' + args.actor + '_' + args.env + '_test_phase'
    else:
        project_name =  datetime.now().strftime("%b_%d_%Y_%H_%M_%S")+ '_initstate_zero' + '_' + args.actor + '_' + args.env + '_alpha_'+str(args.alpha)+'_batch_size_'+str(args.batch_size) + '_lr_critic_' + str(args.lr_critic) + '_lr_actor_' + str(args.lr_actor) +'_activation_'+str(args.actor_activation)[-6:-2] + '_seed_' + str(args.seed)

    if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
        project_name += '_svgd_steps_'+str(args.svgd_steps)+'_svgd_particles_'+str(args.svgd_particles)+'_svgd_lr_'+str(args.svgd_lr) + '_svgd_sigma_p0_' + str(args.svgd_sigma_p0) + '_adaptive_' + str(args.svgd_adaptive_lr) + '_svgd_kernel_sigma_' + str(args.svgd_kernel_sigma)


    # RL args
    RL_kwargs = AttrDict(stats_steps_freq=args.stats_steps_freq,gamma=args.gamma,
        alpha=args.alpha, replay_size=int(args.replay_size), exploration_steps=args.exploration_steps, update_after=args.update_after,
        update_every=args.update_every, num_test_episodes=args.num_test_episodes, plot=args.plot, max_steps = args.max_steps, 
        max_experiment_steps=int(args.max_experiment_steps), evaluation_data_path = args.evaluation_data_path + project_name, 
        debugging=args.debugging, plot_format=args.plot_format, load_replay= args.load_replay, replay_path=args.replay_path, 
        collect_stats_after=args.collect_stats_after, test_time=args.test_time, model_path=args.model_path)

    # optim args
    optim_kwargs = AttrDict(polyak=args.polyak,lr_critic=args.lr_critic, lr_actor=args.lr_actor,batch_size=args.batch_size)

    # critic args
    critic_kwargs = AttrDict(hidden_sizes=[args.hid]*args.l_critic, activation=args.critic_activation)

    # stac
    if args.env =='Multigoal':
        train_env = MultiGoalEnv(max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MultiGoalEnv(max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    elif args.env == 'max-entropy-v0':
        train_env = MaxEntropyEnv(max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MaxEntropyEnv(max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    else: 
        # Fix max steps here
        train_env = gym.make(args.env)
        test_env = gym.make(args.env)

    if args.test_time:
        tb_logger = None
    else:
        os.makedirs(args.tensorboard_path + project_name)
        os.makedirs(args.evaluation_data_path + project_name)
        os.makedirs(args.replay_path + project_name)
        tb_logger = SummaryWriter(args.tensorboard_path + project_name)
    
    if not os.path.exists(args.fig_path + project_name) and RL_kwargs.plot:
        os.makedirs(args.fig_path + project_name)
    else:
        files = glob.glob(args.fig_path + project_name + '/*')
        [os.remove(file) for file in files]

    if not args.test_time:
        ########################################## Hyper-Parameters ##########################################
        print('########################################## Hyper-Parameters ##########################################')
        print('Debugging: ', args.debugging)
        print('GPU ID: ', args.gpu_id)
        print('Environment: ', args.env)
        print('Algorithm: ', args.actor)
        print('Hidden layer size: ', args.hid)
        print('Critic\'s Number of layers: ', args.l_critic)
        if args.actor not in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            print('Actor\'s Number of layers: ', args.l_actor)
        print('Critic\'s Activation: ', args.critic_activation)
        if args.actor not in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            print('Actor\'s Activation: ', args.actor_activation)
        print('Discount Factor (Gamma): ', args.gamma)
        print('Entropy coefficient (Alpha): ', args.alpha)
        print('Replay Buffer size: ', args.replay_size)
        print('Load Replay Buffer: ', args.load_replay)
        print('Replay Buffer Path: ', args.replay_path)
        print('Experiment\'s steps: ', args.max_experiment_steps)
        print('Initial Exploration steps: ', args.exploration_steps)
        print('Number test episodes: ', args.num_test_episodes)
        print('Start Updating models after step: ', args.update_after)
        print('Update models every: ', args.update_every)
        print('Max Environment steps: ', args.max_steps)
        print('Polyak target update rate: ', args.polyak) 
        print('Critic\'s learning rate: ', args.lr_critic)
        if args.actor not in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            print('Actor\'s learning rate: ', args.lr_actor)
        print('Batch size: ', args.batch_size)
        if args.actor == 'sac':
            print('SAC diterministic action selection: ', args.sac_test_deterministic)
        if args.actor == 'svgd_sql':
            print('SQL diterministic action selection: ', args.sql_test_deterministic)
        if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'svgd_sql']:
            print('Number of particles for SVGD: ', args.svgd_particles)
            print('SVGD learning Rate: ', args.svgd_lr)
        if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            print('SVGD diterministic action selection: ', args.svgd_test_deterministic)
            print('Number of SVGD steps: ', args.svgd_steps)
            print('SVGD initial distribution\'s variance: ', args.svgd_sigma_p0)
            print('SVGD\'s kernel variance: ', args.svgd_kernel_sigma)
            print('SVGD\'s adaptive learning rate: ', args.svgd_adaptive_lr)
        print('Tensorboard path: ', args.debugging)
        print('Evaluation data path: ', args.evaluation_data_path)
        print('Figures path: ', args.fig_path)
        print('Plot results: ', args.plot)
        print('Plot format: ', args.plot_format)
        print('Statistics Collection frequency: ', args.stats_steps_freq)
        print('Collect Statistics after: ', args.collect_stats_after)
        print('Seed: ', args.seed)
        print('Device: ', device)
        print('Project Name: ', project_name)
        print('######################################################################################################')

    stac=MaxEntrRL(train_env, test_env, env=args.env, actor=args.actor, device=device, 
        critic_kwargs=critic_kwargs, actor_kwargs=actor_kwargs,
        RL_kwargs=RL_kwargs, optim_kwargs=optim_kwargs,tb_logger=tb_logger, fig_path=args.fig_path +  project_name)


    if args.test_time:
        stac.test_agent(0)
    else:
        start = timeit.default_timer()
        stac.forward()
        stop = timeit.default_timer()
        print('Time: ', stop - start) 
        print() 
        print(project_name)