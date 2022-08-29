import argparse
from envs.multigoal_env import MultiGoalEnv
from spinup_utils.run_utils import setup_logger_kwargs
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from envs.multigoal_env import MultiGoalEnv
import numpy as np
import gym
from datetime import datetime
from core import MaxEntrRL

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Multigoal', choices=['HalfCheetah-v2', 'max-entropy-v0', 'Multigoal'])
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--actor', type=str, default='svgd_nonparam', choices=['sac', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'diffusion'])
    ######networks
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    ######RL 
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--replay_size', type=int, default=1e6)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    ######optim 
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--steps_per_episode', type=int, default=4000)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=500)
    ######sac
    parser.add_argument('--sac_deterministic', type=bool, default=False)
    ######svgd 
    parser.add_argument('--svgd_particles', type=int, default=10)
    parser.add_argument('--svgd_steps', type=int, default=5)
    parser.add_argument('--svgd_lr', type=float, default=0.1)
    parser.add_argument('--svgd_test_deterministic', type=bool, default=False)
    ######logging
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--tensorboard_path', type=str, default='./runs/')
    args = parser.parse_args()    

    # fix the seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # logger
    logger_kwargs = setup_logger_kwargs(args.actor, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    # environment
    if args.env =='Multigoal':
        env_fn = MultiGoalEnv
    else:
        env_fn = lambda : gym.make(args.env)

    # actor arguments
    if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
        actor_kwargs=dict(actor=args.actor, num_svgd_particles=args.svgd_particles, num_svgd_steps=args.svgd_steps, svgd_lr=args.svgd_lr, test_deterministic=args.svgd_test_deterministic)
    elif args.actor in ['sac']:
        actor_kwargs=dict(actor=args.actor, hidden_sizes=[args.hid]*args.l, activation=torch.nn.Identity)

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tensorboard
    project_name = datetime.now().strftime("%b_%d_%Y_%H_%M_%S") +'_'+args.env + '_' + args.actor+'_alpha_'+str(args.alpha)+'_batch_size_'+str(args.batch_size)+'_lr_'+str(args.lr)
    
    if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
        project_name += '_svgd_steps_'+str(args.svgd_steps)+'_svgd_particles_'+str(args.svgd_particles)+'_svgd_lr_'+str(args.svgd_lr)

    tb_logger = SummaryWriter(args.tensorboard_path+project_name)

    # RL args
    RL_kwargs = dict(gamma=args.gamma,alpha=args.alpha,replay_size=args.replay_size,start_steps=args.start_steps,update_after=args.update_after,
        update_every=args.update_every,num_test_episodes=args.num_test_episodes,max_ep_len=args.max_ep_len)

    # optim args
    optim_kwargs = dict(episodes=args.episodes,steps_per_episode=args.steps_per_episode,polyak=args.polyak,lr=args.lr,batch_size=args.batch_size)
    
    stac=MaxEntrRL(env_fn, tb_logger, env=args.env, actor=args.actor, seed=args.seed, device=device, 
        critic_kwargs=dict(hidden_sizes=[args.hid]*args.l, activation=torch.nn.ELU), actor_kwargs= actor_kwargs,
        RL_kwargs=RL_kwargs, optim_kwargs=optim_kwargs, logger_kwargs=logger_kwargs, save_freq = args.save_freq)

