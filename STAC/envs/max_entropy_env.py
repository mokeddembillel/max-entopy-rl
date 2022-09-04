import gym
from gym.utils import seeding
import numpy as np
import os
from gym import spaces
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib
import matplotlib.pyplot as plt

class Map:
    
    def __init__(self,):
        
        self.core = Polygon(np.array([[0,-3], [4,-3], [4,3],[3,3],[3,-2],[-3,-2],[-3,0],[-1,0],[-1, 4],[-3.01, 4],[-3.01, 3],
                                      [-2, 3],[-2, 1],[-3, 1],[-3, 3.01],[-3.99, 3.01],[-3.99, 4],[-6, 4],[-6, -1],[-4, -1],[-4, 0],
                                      [-5, 0],[-5, 3],[-4, 3],[-4,-3],[0, -3]]), True)
        self.start = Polygon(np.array([[-0.5,-4], [0.5,-4], [0.5,-3], [-0.5,-3], [-0.5,-4]]), True)
        self.goal_1 = Polygon(np.array([[3,3], [4,3],[4,4],[3,4],[3,3]]), True)
        self.goal_2 = Polygon(np.array([[-3,3], [-4,3],[-4,4],[-3,4],[-3,3]]), True)
        self.patches = [self.core, self.start, self.goal_1, self.goal_2]

    def in_core(self, point):
        return self.core.contains_point(point)
    def in_start(self, point):
        return self.start.contains_point(point)
    def in_goal_1(self, point):
        return self.goal_1.contains_point(point)
    def in_goal_2(self, point):
        return self.goal_2.contains_point(point)


class MaxEntropyEnv(gym.Env):
    
    def __init__(self, writer=None, starting_state=None, max_steps=500):
        module_path = os.path.dirname(__file__)
        
        
        self.observation_bound = 10
        self.action_bound = 1.
        
        self.observation_space = spaces.Box(low=np.array([-self.observation_bound, -self.observation_bound]),
                                            high=np.array([self.observation_bound, self.observation_bound]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-self.action_bound, -self.action_bound]), 
                                       high=np.array([self.action_bound, self.action_bound]), dtype=np.float32)
        
        self.map = Map()
        self.episodes_information = []
        self.max_steps = max_steps
        self.writer = writer
    
    def reset_rendering(self,):
        plt.close()
        self.episodes_information = []
        self.init_figure()
        
    def reset(self, starting_observation='random'):
        if starting_observation == 'random':
            observation = np.random.uniform(low=[-0.5, -4], high=[0.5, -3] ,size=(2,))
        else:
            observation = starting_observation
        self.status = 'seeking_gold'
        self.steps = 0
            
        self.episodes_information.append({'observations':[observation],
                                    'actions': [],
                                    'rewards': [],
                                    'status': None,
                                    'goal': None, 
                                    'mu': [],
                                    'sigma': [],
                                    'svgd_steps': []
                                    })
        self.done = False
        return observation
            
    def step(self, action):
        self.steps += 1
        if self.done:
            raise('Episode already finished. Reset Please!')
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.episodes_information[-1]['actions'].append(action)
        
        next_observation = self.episodes_information[-1]['observations'][-1] + action
        
        
        goal = None
        reward = 0.
        if not self.map.in_core(next_observation) and \
            not self.map.in_start(next_observation):
                
            if self.map.in_goal_1(next_observation):
                self.status = 'succeeded'
                self.done = True
                reward = 1000.
                goal = 1
            elif self.map.in_goal_2(next_observation):
                self.status = 'succeeded'
                self.done = True
                reward = 1000.
                goal = 2
            else:
                next_observation = self.episodes_information[-1]['observations'][-1]
                # goal = None
                # self.status = 'failed'
                # self.done = True
                # reward = -1000.
                # reward = 0.
        else:
            jumping_observation_check = [self.episodes_information[-1]['observations'][-1] + action * i for i in [0.75, 0.66, 0.5, 0.33, 0.25, 0.125]] 
            for obs in jumping_observation_check:
                if not self.map.in_core(obs) and \
                    not self.map.in_start(obs):
                    next_observation = self.episodes_information[-1]['observations'][-1]
                    break
            # reward = -1.
            # reward = 0.

        self.episodes_information[-1]['observations'].append(next_observation)
        self.episodes_information[-1]['rewards'].append(reward)
        
        if self.steps >= self.max_steps:
            # goal = None
            self.status = 'failed'
            self.done = True
            
        if self.done:
            self.episodes_information[-1]['status'] = self.status
            self.episodes_information[-1]['goal'] = goal
        
        return next_observation, reward, self.done, {'status': self.status, 'goal': goal}
                
    def init_figure(self,):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.xlim([-7, 5])
        plt.ylim([-5, 5])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        p = PatchCollection(self.map.patches, cmap=matplotlib.cm.jet, alpha=0.4)
        p.set_facecolor(np.array(['royalblue', 'teal', 'gold', 'gold']))
        # p.set_edgecolor(np.array(['royalblue', 'teal', 'gold', 'gold']))
        p.set_alpha(1.)
        self.ax.add_collection(p)
    
    def render(self, render=True, itr=None, mode="human"):
        self.episodes_information[0]['observations']
        for i in range(len(self.episodes_information)):
            path = np.array(self.episodes_information[i]['observations'])
            if self.episodes_information[i]['status'] == 'failed':
                color = 'red'
            elif self.episodes_information[i]['status'] == 'succeeded':
                color = 'lime'
            else:
                color = 'orange'
            self.ax.plot(path[:, 0], path[:, 1], color)
        if render == 'plot':
            plt.plot()
        elif render == 'draw':
            plt.draw()
            
    def collect_plotting_data(self, ac):

        if ac.pi.actor_name not in ['svgd_nonparam', 'svgd_sql']:
            self.episodes_information[-1]['mu'].append(ac.pi.mu.detach().cpu().numpy())
            self.episodes_information[-1]['sigma'].append(ac.pi.sigma.detach().cpu().numpy())
        if ac.pi.actor_name not in  ['sac', 'svgd_sql']:
            self.episodes_information[-1]['svgd_steps'].append(ac.pi.svgd_steps)
            self.episodes_information[-1]['ac_hess_list'].append(ac.pi.hess_list)
            self.episodes_information[-1]['ac_score_func_list'].append(ac.pi.score_func_list)
            self.episodes_information[-1]['ac_hess_eig_max'].append(ac.pi.hess_eig_max)
            
        
    
    def plot_path(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.xlim([-7, 5])
        plt.ylim([-5, 5])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        p = PatchCollection(self.map.patches, cmap=matplotlib.cm.jet, alpha=0.4)
        p.set_facecolor(np.array(['royalblue', 'teal', 'gold', 'gold']))
        # p.set_edgecolor(np.array(['royalblue', 'teal', 'gold', 'gold']))
        p.set_alpha(1.)
        ax.add_collection(p)    
        path = np.array(self.episodes_information[-2]['observations'])
        # print('path :', self.episodes_information[-2])
        if self.episodes_information[-2]['status'] == 'failed':
            color = 'red'
        elif self.episodes_information[-2]['status'] == 'succeeded':
            color = 'lime'
        else:
            color = 'orange'
        ax.plot(path[:, 0], path[:, 1], color)
        # plt.savefig('./max_entropy_plots_/buffer_1')
        plt.draw()
        
        
    def save_fig(self, path):
        plt.savefig(path)
        plt.close()
    

































