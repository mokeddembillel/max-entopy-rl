import numpy as np 
import torch as torch
import matplotlib.pyplot as plt
from utils import gaussian
from tqdm import tqdm
import matplotlib.colors as mpl_colors
from mpl_toolkits.mplot3d import Axes3D


class Debugger():
    def __init__(self, tb_logger, ac, env_name, train_env, test_env, plot_format, update_after, num_test_episodes, alpha, env_max_steps, max_experiment_steps):
        # Still need some improvements that i will do tomorrow
        self.ac = ac
        self.tb_logger = tb_logger
        self.env_name = env_name
        self.train_env = train_env
        self.test_env = test_env
        self.episodes_information = []
        self.episode_counter = 0
        self.colors = ['red', 'orange', 'purple']
        self.episodes_information_svgd = []
        self.plot_format = plot_format
        self.plot_cumulative_entropy = update_after + 5000
        self.num_test_episodes = num_test_episodes
        self.alpha = alpha
        self.boundary_action_counter = 0 
        self.boundary_all_actions_counter = 0 
        self.action_counter = 0 
        self.env_max_steps = env_max_steps
        self.max_experiment_steps = max_experiment_steps - 1
        
        if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            self.average_cumulative_entropy = np.zeros((self.test_env.num_goals))
            self.cumulative_entropy_coutner = np.zeros((self.test_env.num_goals))

    def collect_data(self, o, a, o2, r, d, log_p, itr, ep_len, robot_pic_rgb=None):
        
        if ep_len == 0:
            self.episodes_information.append({
                'rewards': [],
                'expected_reward': None, 
                'episode_length': None,
                })

        self.episodes_information[-1]['rewards'].append(r)




        if ((ep_len + 1) >= self.env_max_steps) or d: 
            # self.episodes_information[-1]['observations'].append(o2.squeeze())
            self.episodes_information[-1]['expected_reward'] = np.sum(self.episodes_information[-1]['rewards'])
            self.episodes_information[-1]['episode_length'] = ep_len



    def log_to_tensorboard(self, itr):

        # 
        expected_rewards = list(map(lambda x: x['expected_reward'], self.episodes_information))
        episode_length = list(map(lambda x: x['episode_length'], self.episodes_information))

        # import pdb; pdb.set_trace()
        self.tb_logger.add_scalars('Test_EpRet/return_detailed',  {'Mean ': np.mean(expected_rewards), 'Min': np.min(expected_rewards), 'Max': np.max(expected_rewards) }, itr)
        self.tb_logger.add_scalars('Test_EpRet/return_mean_only',  {'Mean ': np.mean(expected_rewards)}, itr)
        self.tb_logger.add_scalar('Test_EpLen', np.mean(episode_length) , itr)
        
    def reset(self,):
        self.episodes_information = []
        self.boundary_action_counter = 0 
        self.boundary_all_actions_counter = 0 
        self.action_counter = 0 



    


    
