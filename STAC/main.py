import argparse
from core import MaxEntrRL
import random
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from envs.max_entropy_env import MaxEntropyEnv
from envs.multigoal_env import MultiGoalEnv
from envs.multigoal_env_obstacles import MultiGoalObstaclesEnv
from envs.multigoal_max_entropy_env import MultiGoalMaxEntropyEnv
from envs.multigoal_max_entropy_env_obstacles import MultiGoalMaxEntropyObstaclesEnv

import numpy as np
import gym
import mujoco_py
from datetime import datetime
from utils import AttrDict
import glob
import timeit
from tqdm import tqdm



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--env', type=str, default='Hopper-v2', choices=['Multigoal', 'max-entropy-v0', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles', 'multigoal-obstacles', 'Hopper-v2', 'Ant-v2', 'Walker2d-v2', 'Humanoid-v2', 'HalfCheetah-v2'])
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--actor', type=str, default='svgd_nonparam', choices=['sac', 'svgd_sql', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'diffusion'])

    ###### networks
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l_critic', type=int, default=2)
    parser.add_argument('--l_actor', type=int, default=3)
    parser.add_argument('--critic_activation', type=object, default=torch.nn.ELU)
    parser.add_argument('--actor_activation', type=object, default=torch.nn.ELU)    

    ###### RL 
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--replay_size', type=int, default=1e6)
    parser.add_argument('--load_replay', type=int, default=0)
    parser.add_argument('--replay_path', type=str, default='./STAC/buffers_/')
    parser.add_argument('--max_experiment_steps', type=float, default=5e4)
    parser.add_argument('--exploration_steps', type=int, default=10000, help="pure exploration at the beginning of the training")

    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=30)

    ###### optim 
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=100)
    
    
    ###### action selection
    parser.add_argument('--train_action_selection', type=str, default='softmax', choices=['softmax', 'max', 'softmax', 'adaptive_softmax', 'softmax_egreedy'])
    parser.add_argument('--test_action_selection', type=str, default='max', choices=['max', 'max', 'softmax', 'adaptive_softmax', 'softmax_egreedy'])
  


    parser.add_argument('--svgd_particles', type=int, default=10)
    parser.add_argument('--svgd_steps', type=int, default=20)
    parser.add_argument('--svgd_lr', type=float, default=0.1)
    parser.add_argument('--svgd_sigma_p0', type=float, default=0.5)
    parser.add_argument('--svgd_kernel_sigma', type=float, default=None)
    parser.add_argument('--kernel_sigma_adaptive', type=int, default=4)
    parser.add_argument('--svgd_adaptive_lr', type=int, default=0)
   
    # tensorboard
    parser.add_argument('--tensorboard_path', type=str, default='./runs/')
    parser.add_argument('--evaluation_data_path', type=str, default='./evaluation_data/')
    parser.add_argument('--fig_path', type=str, default='./STAC/multi_goal_plots_/')
    parser.add_argument('--plot', type=int, default=1)
    parser.add_argument('--plot_format', type=str, default='pdf', choices=['png', 'jpeg', 'pdf', 'svg'])
    parser.add_argument('--stats_steps_freq', type=int, default=400) 
    parser.add_argument('--collect_stats_after', type=int, default=0)
    
    parser.add_argument('--model_path', type=str, default='./evaluation_data/z_after/svgd_nonparam_999999')


    ###################################################################################
    ###################################################################################
    parser.add_argument('--experiment_importance', type=str, default='dbg', choices=['dbg', 'prm', 'scn']) 
    parser.add_argument('--test_time', type=int, default=1)
    parser.add_argument('--all_checkpoints_test', type=int, default=1) 
    parser.add_argument('--debugging', type=int, default=0) 
    ###################################################################################
    ###################################################################################

    
    args = parser.parse_args()  
    args.plot = bool(args.plot)
    args.svgd_adaptive_lr = bool(args.svgd_adaptive_lr)
    args.debugging = bool(args.debugging)
    args.load_replay = bool(args.load_replay)
    args.test_time = bool(args.test_time)
    args.all_check_points_test = bool(args.all_checkpoints_test)
    # import pdb; pdb.set_trace()
    ################# Best parameters for a specific thing #################


    if args.test_time:
        print('############################## TEST TIME ###################################')
        print('############################################################################')
        print('############################################################################')

    if args.actor == 'svgd_sql':
        # args.lr_critic = 1e-2
        # args.alpha = 1.
        args.critic_activation = torch.nn.ReLU
        args.actor_activation = torch.nn.ReLU

    if args.actor in ['sac', 'svgd_p0_pram']:
        # args.critic_activation = torch.nn.Tanh
        # args.actor_activation = torch.nn.Tanh
        args.critic_activation = torch.nn.ReLU
        args.actor_activation = torch.nn.ReLU
        # args.alpha = 0.2
    
    if args.env in ['Hopper-v2', 'Ant-v2', 'Walker2d-v2', 'Humanoid-v2', 'HalfCheetah-v2']:
        args.fig_path = './STAC/mujoco_plots_/'
        args.max_steps = 1000
    elif args.env == 'max-entropy-v0':
        args.stats_steps_freq = 1000
        args.max_steps = 500
        # args.gamma = 1.0
        # args.num_test_episodes = 2
        args.fig_path = './STAC/max_entropy_plots_/'
    
    if args.debugging:
        print('############################## DEBUGGING ###################################')
        args.svgd_steps = 'while'
        args.exploration_steps = 0
        # args.actor = 'svgd_sql'
        args.max_experiment_steps = 30000
        # args.exploration_steps = 100
        args.update_after = 1000
        args.stats_steps_freq = 400
        args.num_test_episodes = 1
        # args.max_steps = 500
        args.collect_stats_after = 0
        # args.entropy_particles = 10
        # args.svgd_particles = 100

        # args.update_after = 1000000
        # args.stats_steps_freq = 400
        # args.num_test_episodes = 1
        # args.max_steps = 500
        # args.collect_stats_after = 0



        # tmp = 400000
        # args.max_experiment_steps =  415000 
        # args.exploration_steps = tmp 
        # args.update_after = tmp 
        # # args.update_after = 400
        # args.collect_stats_after = tmp 
        # args.seed = 0 
        # # args.alpha = 1.0
        # args.alpha = 0.2




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
            svgd_lr=args.svgd_lr, test_action_selection=args.test_action_selection, svgd_sigma_p0 = args.svgd_sigma_p0,
            batch_size=args.batch_size,  device=device, hidden_sizes=[args.hid]*args.l_actor, activation=args.actor_activation, 
            kernel_sigma=args.svgd_kernel_sigma, adaptive_lr=args.svgd_adaptive_lr, adaptive_sig=args.kernel_sigma_adaptive, alpha=args.alpha)
    
    elif (args.actor == 'svgd_sql'):
        actor_kwargs=AttrDict(num_svgd_particles=args.svgd_particles, 
            svgd_lr=args.svgd_lr, test_action_selection=args.test_action_selection, 
            batch_size=args.batch_size,  device=device, hidden_sizes=[args.hid]*args.l_actor, 
            activation=args.actor_activation, kernel_sigma=args.svgd_kernel_sigma, adaptive_sig=args.kernel_sigma_adaptive)
    elif (args.actor =='sac'):
        actor_kwargs=AttrDict(hidden_sizes=[args.hid]*args.l_actor, test_action_selection=args.test_action_selection, device=device, activation=args.actor_activation, batch_size=args.batch_size)
    
    # Logging
    #
    project_name = args.experiment_importance + '_'
    if args.test_time:
        project_name +=  'test_'
    
    # project_name +=  datetime.now().strftime("%b_%d_%Y_%H_%M_%S")+ '_' + args.actor + '_' + args.env + '_alpha_'+str(args.alpha)+'_bs_'+ str(args.batch_size) + '_lr_c_' + str(args.lr_critic) + '_lr_a_' + str(args.lr_actor) +'_act_'+str(args.actor_activation)[-6:-2] + '_seed_' + str(args.seed) + '_'
    project_name +=  datetime.now().strftime("%b_%d_%Y_%H_%M_%S")+ '_' + 'tnas_' + args.train_action_selection + '_ttas_' + args.test_action_selection + '_' + args.actor + '_' + args.env + '_alpha_'+str(args.alpha)+'_bs_'+ str(args.batch_size) + '_gamma_' + str(args.gamma) + '_seed_' + str(args.seed) + '_ntep_' + str(args.num_test_episodes) + '_'

    if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
        project_name += 'ssteps_'+str(args.svgd_steps)+'_sparticles_'+str(args.svgd_particles)+'_slr_'+str(args.svgd_lr) + '_ssigma_p0_' + str(args.svgd_sigma_p0) + '_skernel_sigma_' + str(args.svgd_kernel_sigma) + '_' + str(args.kernel_sigma_adaptive) + '_'
    elif args.actor in ['svgd_sql']:
        project_name += 'sparticles_'+str(args.svgd_particles)+'_slr_'+str(args.svgd_lr) + '_ssigma_p0_' + str(args.svgd_sigma_p0) + '_skernel_sigma_' + str(args.svgd_kernel_sigma) + '_' + str(args.kernel_sigma_adaptive) + '_'

    if args.test_time:
        project_name += 'PID_' + str(os.getpid())
    else:
        project_name += 'exper_' + str(args.max_experiment_steps) + '_explor_' + str(args.exploration_steps) + '_update_' + str(args.update_after) + '_PID_' + str(os.getpid())

    # RL args
    RL_kwargs = AttrDict(stats_steps_freq=args.stats_steps_freq,gamma=args.gamma,
        alpha=args.alpha, replay_size=int(args.replay_size), exploration_steps=args.exploration_steps, update_after=args.update_after,
        update_every=args.update_every, num_test_episodes=args.num_test_episodes, plot=args.plot, max_steps = args.max_steps, 
        max_experiment_steps=int(args.max_experiment_steps), evaluation_data_path = args.evaluation_data_path + project_name, 
        debugging=args.debugging, plot_format=args.plot_format, load_replay= args.load_replay, replay_path=args.replay_path, 
        collect_stats_after=args.collect_stats_after, test_time=args.test_time, all_checkpoints_test=args.all_checkpoints_test, model_path=args.model_path, train_action_selection=args.train_action_selection)

    # optim args
    optim_kwargs = AttrDict(polyak=args.polyak,lr_critic=args.lr_critic, lr_actor=args.lr_actor,batch_size=args.batch_size)

    # critic args
    critic_kwargs = AttrDict(hidden_sizes=[args.hid]*args.l_critic, activation=args.critic_activation)

    # stac
    if args.env =='Multigoal':
        train_env = MultiGoalEnv(env_name='train_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MultiGoalEnv(env_name='test_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    elif args.env == 'multigoal-max-entropy':
        train_env = MultiGoalMaxEntropyEnv(env_name='train_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MultiGoalMaxEntropyEnv(env_name='test_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    elif args.env == 'multigoal-obstacles':
        train_env = MultiGoalObstaclesEnv(env_name='train_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MultiGoalObstaclesEnv(env_name='test_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    elif args.env == 'multigoal-max-entropy-obstacles':
        train_env = MultiGoalMaxEntropyObstaclesEnv(env_name='train_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MultiGoalMaxEntropyObstaclesEnv(env_name='test_env', max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    elif args.env == 'max-entropy-v0':
        train_env = MaxEntropyEnv(max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
        test_env = MaxEntropyEnv(max_steps=RL_kwargs.max_steps, plot_format=args.plot_format)
    else: 
        # Fix max steps here
        train_env = gym.make(args.env)
        test_env = gym.make(args.env)


    if not args.test_time:
        os.makedirs(args.evaluation_data_path + project_name)
        os.makedirs(args.replay_path + project_name)
    os.makedirs(args.tensorboard_path + project_name)
    tb_logger = SummaryWriter(args.tensorboard_path + project_name)
    
    if not os.path.exists(args.fig_path + project_name) and RL_kwargs.plot:
        os.makedirs(args.fig_path + project_name)
    else:
        files = glob.glob(args.fig_path + project_name + '/*')
        [os.remove(file) for file in files]

    ########################################## Hyper-Parameters ##########################################
    print('########################################## Hyper-Parameters ##########################################')
    if not args.test_time:
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

        print('Train action selection: ', args.train_action_selection)
        print('Test action selection: ', args.test_action_selection)

        if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'svgd_sql']:
            print('Number of particles for SVGD: ', args.svgd_particles)
            print('SVGD learning Rate: ', args.svgd_lr)
        if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            print('Number of SVGD steps: ', args.svgd_steps)
            print('SVGD initial distribution\'s variance: ', args.svgd_sigma_p0)
            print('SVGD\'s kernel variance: ', args.svgd_kernel_sigma)
            print('SVGD\'s adaptive learning rate: ', args.svgd_adaptive_lr)
        # print('Number of particles for entropy computation: ', args.entropy_particles)
        print('Tensorboard path: ', args.tensorboard_path)
        print('Evaluation data path: ', args.evaluation_data_path)
        print('Figures path: ', args.fig_path)
        print('Plot results: ', args.plot)
        print('Plot format: ', args.plot_format)
        print('Statistics Collection frequency: ', args.stats_steps_freq)
        print('Collect Statistics after: ', args.collect_stats_after)
        print('Seed: ', args.seed)
        print('Device: ', device)
    print('Project Name: ', project_name)
    print('Experiment Importance: ', args.experiment_importance)
    print('Experiment PID: ', os.getpid())
    print('######################################################################################################')

    if args.all_checkpoints_test:
        project_name = 'prm_Jan_12_2023_13_53_20_tnas_softmax_ttas_max_svgd_nonparam_Hopper-v2_alpha_1.0_bs_100_gamma_0.99_seed_0_ntep_10_ssteps_20_sparticles_10_slr_0.1_ssigma_p0_0.5_sad_lr_False_skernel_sigma_None_4_exper_1000000.0_explor_10000_update_1000_PID_2373730'
        tensorboard_path = './runs/' + 'cpsm2_' + project_name + '/'
        # tensorboard_path = './runs/' + 'dbg_Jan_07_2023_03_26_08_tnas_random_ttas_random_svgd_nonparam_Hopper-v2_alpha_5_bs_100_gamma_0.99_seed_0_ssteps_10_sparticles_10_slr_0.01_ssigma_p0_0.3_sad_lr_False_skernel_sigma_None_4_exper_34432423423_explor_0_update_111111110_PID_1932237' + '/'
        checkpoints_path = './evaluation_data/' + project_name
        RL_kwargs.evaluation_data_path = './evaluation_data/' + 'cpsm2_' + project_name
        try:
            os.makedirs(args.evaluation_data_path + 'cpsm_' + project_name)
        except:
            pass
        tb_logger = SummaryWriter(tensorboard_path)
        for i in tqdm(range(3999, 1000000, 8000)):
            start = timeit.default_timer()
            # RL_kwargs.model_path = checkpoints_path + '/svgd_nonparam_' + str(999999)
            RL_kwargs.model_path = checkpoints_path + '/svgd_nonparam_' + str(i)

            stac=MaxEntrRL(train_env, test_env, env=args.env, actor=args.actor, device=device, 
                critic_kwargs=critic_kwargs, actor_kwargs=actor_kwargs,
                RL_kwargs=RL_kwargs, optim_kwargs=optim_kwargs,tb_logger=tb_logger, fig_path=args.fig_path +  project_name)
            
            stac.RL_kwargs.num_test_episodes = 5
            mean_particles = []
            # config = [[1, 40], [10, 40], [10, 60], [10, 'while'], [20, 40], [20, 60], [20, 'while']]
            # config = [[1, 40], [10, 40], [10, 60]]
            # config = [[20, 40], [20, 60]]
            config = [[10, 'while']]
            # config = [[20, 'while']]
            for p in config:
                stac.ac.pi.num_particles = p[0]
                stac.ac.pi.num_svgd_steps = p[1]
                stac.test_agent(i)
                mean_particles.append(np.mean(list(map(lambda x: x['expected_reward'], stac.debugger.episodes_information))))
                stac.debugger.reset()
            # print({'mean_p'+str(p[0])+'_s'+str(p[1]): mean_particles[c] for c, p in enumerate(config)})
            stac.save_data()
            stac.debugger.tb_logger.add_scalars('Test_EpRet/return_mean_only_particles',  {'mean_p'+str(p[0])+'_s'+str(p[1]): mean_particles[c] for c, p in enumerate(config)}, i)
            
            stop = timeit.default_timer()
            print('Time deriv auto: ', stop - start) 




        # project_name = 'prm_Jan_04_2023_20_57_49_tnd_True_ttd_True_as_2_svgd_nonparam_Hopper-v2_alpha_1.0_bs_100_gamma_0.99_seed_0_ssteps_10_sparticles_10_slr_0.1_ssigma_p0_0.5_sad_lr_False_skernel_sigma_None_4_exper_1000000.0_explor_10000_update_1000_PID_1811987'
        # tensorboard_path = './runs/' + project_name + '/'
        # # tensorboard_path = './runs/' + 'dbg_Jan_07_2023_03_26_08_tnas_random_ttas_random_svgd_nonparam_Hopper-v2_alpha_5_bs_100_gamma_0.99_seed_0_ssteps_10_sparticles_10_slr_0.01_ssigma_p0_0.3_sad_lr_False_skernel_sigma_None_4_exper_34432423423_explor_0_update_111111110_PID_1932237' + '/'
        # checkpoints_path = './evaluation_data/' + project_name
        # tb_logger = SummaryWriter(tensorboard_path)
        # for i in tqdm(range(3999, 999999 + 4000, 4000)):
        #     RL_kwargs.model_path = checkpoints_path + '/svgd_nonparam_' + str(i)

        #     stac=MaxEntrRL(train_env, test_env, env=args.env, actor=args.actor, device=device, 
        #         critic_kwargs=critic_kwargs, actor_kwargs=actor_kwargs,
        #         RL_kwargs=RL_kwargs, optim_kwargs=optim_kwargs,tb_logger=tb_logger, fig_path=args.fig_path +  project_name)
        #     stac.test_agent(i)
        
    else:
    
        stac=MaxEntrRL(train_env, test_env, env=args.env, actor=args.actor, device=device, 
            critic_kwargs=critic_kwargs, actor_kwargs=actor_kwargs,
            RL_kwargs=RL_kwargs, optim_kwargs=optim_kwargs,tb_logger=tb_logger, fig_path=args.fig_path +  project_name)

        start = timeit.default_timer()
        if args.test_time:
            stac.test_agent(0)
            # stac.debugger.entorpy_landscape(args.fig_path +  project_name + '/')
        else:
            stac.forward()
        stop = timeit.default_timer()
        print('Time: ', stop - start) 
        print() 
        print(project_name)




# def step():
#     if direction in current segment is correct:
#         if agent in goal1:
#             do something
#         elif agent in goal2:
#             do something
#         else:
#             find intersections
#             if there are intersections: 
#                 get the closest intersections
#                 next_observation = (intersections[closest_intersection] - current_observation) * 0.90 + current_observation
#                 if next_observation == intersections[closest_intersection]: 
#                     # check becuase with the above line if multiple actions were taken towards the 
#                     # wall it will get closer to the wall. this will prevent it from going outside the wall
#                     next_observation = current_observation
#             get current segment given next_observation
#     else:
#         next_observation = current_observation

#     store transition what ever happens. ( this is what youre asking me to change)













