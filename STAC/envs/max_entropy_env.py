import gym
from gym.utils import seeding
import numpy as np
import os
from gym import spaces
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib
import matplotlib.pyplot as plt
import torch

class Map:
    
    def __init__(self,):
        
        # self.core = Polygon(np.array([[0,-3], [4,-3], [4,3],[3,3],[3,-2],[-3,-2],[-3,-1],[-1,-1],[-1, 4],[-3.01, 4],[-3.01, 3],
        #                               [-2, 3],[-2, 0],[-3, 0],[-3, 3.01],[-3.99, 3.01],[-3.99, 4],[-6, 4],[-6, -1],[-4, -1],[-4, 0],
        #                               [-5, 0],[-5, 3],[-4, 3],[-4,-3],[0, -3]]), True)
        # self.core = Polygon(np.array([[0,-3], [4,-3], [4,3],[3,3],[3,-2],[-3,-2],[-3,-1],[-1,-1],[-1, 4],[-3.01, 4],[-3.01, 3],
        #                               [-2, 3],[-2, 0],[-5, 0],[-5, 3],[-3.99, 3],[-3.99, 4],[-6, 4],[-6, -1],[-4, -1],[-4,-3],[0, -3]]), True)
        self.core = Polygon(np.array([[0,-3], [4,-3], [4,0], [2,0],[2,3],[3.01,3],[3.01,4], [1,4],[1,-1],[3,-1],[3,-2],[-3,-2],[-3,-1],[-1,-1],[-1, 4],[-3.01, 4],[-3.01, 3],
                                      [-2, 3],[-2, 0],[-5, 0],[-5, 3],[-3.99, 3],[-3.99, 4],[-6, 4],[-6, -1],[-4, -1],[-4,-3],[0, -3]]), True)


        self.start = Polygon(np.array([[-0.5,-4], [0.5,-4], [0.5,-3], [-0.5,-3], [-0.5,-4]]), True)
        self.goal_1 = Polygon(np.array([[-3,3], [-4,3],[-4,4],[-3,4],[-3,3]]), True)
        self.goal_2 = Polygon(np.array([[3,3], [4,3],[4,4],[3,4],[3,3]]), True)
        # self.core = Polygon(np.array([[0,-3], [-4,-3], [-4,0], [-2,0],[-2,3],[-3.01,3],[-3.01,4], [-1,4],[-1,-1],[-3,-1],[-3,-2],[3,-2],[3,-1],[1,-1],[1, 4],[3.01, 4],[3.01, 3],
        #                               [2, 3],[2, 0],[5, 0],[5, 3],[3.99, 3],[3.99, 4],[6, 4],[6, -1],[4, -1],[4,-3],[0, -3]]), True)
        # self.goal_1 = Polygon(np.array([[3,3], [4,3],[4,4],[3,4],[3,3]]), True)
        # self.goal_2 = Polygon(np.array([[-3,3], [-4,3],[-4,4],[-3,4],[-3,3]]), True)


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
    
    def __init__(self, writer=None, starting_state=None, max_steps=500, plot_format=None):
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
        self.plot_format = plot_format

        # Plotter params, to be cleaned tomorrow. 
        self.entropy_obs_list = [
                [0, -3],
                [-3.5, -0.5], 
                [1.5, 2], 
                [-5.5, 2], 
                [3.5, -0.5],
                [-1.5, 2]
            ]
        self._n_samples = 100
        self.n_plots = len(self.entropy_obs_list)
        self.x_size = (1.25 * self.n_plots + 1)
        self.y_size = 14

        self.failures = np.zeros((2,))

        self.paths = np.zeros((3,))
    
    def reset_rendering(self,):
        plt.close()
        self.episodes_information = []
        self.init_figure()
        
    def reset(self, starting_observation='random'):
        if starting_observation == 'random':
            # observation = np.random.uniform(low=[-0.5, -4], high=[0.5, -3] ,size=(2,))
            observation = np.array([0,-3.5])
        else:
            observation = starting_observation
        self.status = 'seeking_gold'
        self.ep_len = 0
            
        self.episodes_information.append({'observations':[observation],
                                    'actions': [],
                                    'rewards': [],
                                    'status': None,
                                    'goal': None, 
                                    'mu': [],
                                    'sigma': [],
                                    'svgd_steps': [],
                                    'path_taken': None
                                    })
        self.done = False
        return observation
            
    def step(self, action):
        # self.agent_failing = False
        
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
                # reward = 1.
                # reward = 10.
                reward = 40.
                # reward = 0.
                goal = 1

                i = len(self.episodes_information[-1]['observations'])-1
                while i >= 0:
                    if -6 <= self.episodes_information[-1]['observations'][i][0] <= -5:
                        self.paths[0] += 1
                        break
                    elif -2 <= self.episodes_information[-1]['observations'][i][0] <= -1:
                        self.paths[1] += 1
                        break
                    i -= 1

            elif self.map.in_goal_2(next_observation):
                self.status = 'succeeded'
                self.done = True
                # reward = 1.
                # reward = 10.
                reward = 40.
                # reward = 0.
                goal = 2
                self.paths[2] += 1
                
                
            else:
                next_observation = self.episodes_information[-1]['observations'][-1]
                # self.agent_failing = True
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
                    # self.agent_failing = True
                    break
            # obs_tmp = self.episodes_information[-1]['observations'][-1]
            # if -4 <= obs_tmp[0] <= -3 and next_observation[0] <= -4 and next_observation[0] >= -3 and next_observation[1] >= 3:
            #     next_observation = self.episodes_information[-1]['observations'][-1]
            # elif 3 <= obs_tmp[1] <= 4 and -4 <= next_observation[0] <= -3 and next_observation[1] <= 3:
            #     next_observation = self.episodes_information[-1]['observations'][-1]
            # reward = -1.
            # reward = 0.

        self.episodes_information[-1]['observations'].append(next_observation)
        self.episodes_information[-1]['rewards'].append(reward)

        # if not self.agent_failing:
        self.ep_len += 1

        if self.ep_len == self.max_steps:
            # goal = None
            self.status = 'failed'
            self.done = False
            if next_observation[0] <= -1:
                self.failures[0] += 1
            elif next_observation[0] >= 1:
                self.failures[1] += 1
            
        if self.ep_len == self.max_steps or self.done:
            self.episodes_information[-1]['status'] = self.status
            self.episodes_information[-1]['goal'] = goal
        
        return next_observation, reward, self.done, {'status': self.status, 'goal': goal}
                
    # def init_figure(self,):
    #     self.fig, self.ax = plt.subplots(figsize=(10, 10))
    #     plt.xlim([-7, 5])
    #     plt.ylim([-5, 5])
    #     self.ax.set_xlabel('x')
    #     self.ax.set_ylabel('y')
    #     p = PatchCollection(self.map.patches, cmap=matplotlib.cm.jet, alpha=0.4)
    #     p.set_facecolor(np.array(['royalblue', 'teal', 'gold', 'gold']))
    #     # p.set_edgecolor(np.array(['royalblue', 'teal', 'gold', 'gold']))
    #     p.set_alpha(1.)
    #     self.ax.add_collection(p)
    
    
    def render(self, fig_path, plot, render='plot', itr=None, mode="human", ac=None, goals=None):
        if plot:
            self._ax_lst[0].set_title(str(goals))
            self.episodes_information[0]['observations']
            for i in range(len(self.episodes_information)):
                path = np.array(self.episodes_information[i]['observations'])
                if self.episodes_information[i]['status'] == 'failed':
                    color = 'red'
                elif self.episodes_information[i]['status'] == 'succeeded':
                    color = 'lime'
                else:
                    color = 'orange'
                self._ax_lst[0].plot(path[:, 0], path[:, 1], color, zorder=1)

            obs_colors = np.array([[255,99,71], [255,140,0], [0,100,0], [138,43,226], [160,82,45], [80,82,45]])
            self.entropy_obs_list = np.array(self.entropy_obs_list)
            self.entropy_list = []
            if ac.pi.actor == 'sac':
                self.mean_list_x = []
                self.sigma_list_x = []
                self.mean_list_y = []
                self.sigma_list_y = []
            # get actions 
            for i in range(len(self.entropy_obs_list)):
                o = torch.as_tensor(self.entropy_obs_list[i], dtype=torch.float32).to(ac.pi.device).view(-1,1,self.observation_space.shape[0]).repeat(1,ac.pi.num_particles,1).view(-1,self.observation_space.shape[0])
                a, log_p = ac(o, deterministic=ac.pi.test_deterministic, with_logprob=True, all_particles=False)
                self.entropy_list.append(round(-log_p.detach().item(), 2))
                if ac.pi.actor == 'sac':
                    self.mean_list_x.append(ac.pi.mu[0,0].detach().cpu().item())
                    self.sigma_list_x.append(ac.pi.sigma[0,0].detach().cpu().item())
                    self.mean_list_y.append(ac.pi.mu[0,1].detach().cpu().item())
                    self.sigma_list_y.append(ac.pi.sigma[0,1].detach().cpu().item())
            self._ax_lst[0].scatter(self.entropy_obs_list[:, 0], self.entropy_obs_list[:, 1], c=list(obs_colors/255.0), marker='*', s=100, zorder=2)
            for i in range(len(self.entropy_obs_list)):
                self._ax_lst[0].annotate(str(self.entropy_list[i]), (self.entropy_obs_list[i,0] + 0.1, self.entropy_obs_list[i,1]), fontsize=12, color=[0,0,0], zorder=2)
            
            self._plot_level_curves( ac)
            self._plot_action_samples(ac)

            if render == 'plot':
                plt.plot()
            elif render == 'draw':
                plt.draw()
            plt.savefig(fig_path+ '/env_' + str(itr) + '.' + self.plot_format)
            plt.close()
            
    def init_figure(self,grid_size=(5,3)):
        self._ax_lst = []
        self.fig_env = plt.figure(figsize=(self.x_size, self.y_size), constrained_layout=True) 
        self._ax_lst.append(plt.subplot2grid(grid_size, (0,0), colspan=3, rowspan=3))
        self._ax_lst[0].set_xlim([-7, 5])
        # self._ax_lst[0].set_xlim([-5, 7])
        self._ax_lst[0].set_ylim([-5, 5])
        self._ax_lst[0].set_xlabel('x')
        self._ax_lst[0].set_ylabel('y')
        p = PatchCollection(self.map.patches, cmap=matplotlib.cm.jet, alpha=0.4)
        p.set_facecolor(np.array(['royalblue', 'teal', 'gold', 'gold']))
        # p.set_edgecolor(np.array(['royalblue', 'teal', 'gold', 'gold']))
        p.set_alpha(1.)
        self._ax_lst[0].add_collection(p)
        for i in range(self.n_plots):
            ax = plt.subplot2grid(grid_size, (3 + i//3,i%3))
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            ax.grid(True)
            self._ax_lst.append(ax)
            self._line_objects = list()
        # plt.savefig('STAC/max_entropy_plots_' + '/env_.' + self.plot_format)
        # print('done')

    def _plot_level_curves(self, ac):
        # Create mesh grid.
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        xgrid, ygrid = np.meshgrid(xs, ys)
        a = np.concatenate((np.expand_dims(xgrid.ravel(), -1), np.expand_dims(ygrid.ravel(), -1)), -1)
        a = torch.from_numpy(a.astype(np.float32)).to(ac.pi.device)
        for i in range(len(self.entropy_obs_list)):
            o = torch.Tensor(self.entropy_obs_list[i]).repeat([a.shape[0],1]).to(ac.pi.device)
            with torch.no_grad():
                qs = ac.q1(o.to(ac.pi.device), a).cpu().detach().numpy()
            qs = qs.reshape(xgrid.shape)
            cs = self._ax_lst[i+1].contour(xgrid, ygrid, qs, 20)
            self._line_objects += cs.collections
            self._line_objects += self._ax_lst[i+1].clabel(
                cs, inline=1, fontsize=10, fmt='%.2f')

    def _plot_action_samples(self, ac):
        for i in range(len(self.entropy_obs_list)):
            if ac.pi.actor == 'svgd_nonparam':
                o = torch.as_tensor(self.entropy_obs_list[i], dtype=torch.float32).repeat([self._n_samples*ac.pi.num_particles,1]).to(ac.pi.device)
            else:
                o = torch.as_tensor(self.entropy_obs_list[i], dtype=torch.float32).repeat([self._n_samples,1]).to(ac.pi.device)
            actions, _ = ac(o, deterministic=ac.pi.test_deterministic, with_logprob=False)
            actions = actions.cpu().detach().numpy().squeeze()
            x, y = actions[:, 0], actions[:, 1]
            self._ax_lst[i+1].title.set_text(str(self.entropy_obs_list[i]))
            self._line_objects += self._ax_lst[i+1].plot(x, y, 'b*')
    
    # def plot_path(self):
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     plt.xlim([-7, 5])
    #     plt.ylim([-5, 5])
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     p = PatchCollection(self.map.patches, cmap=matplotlib.cm.jet, alpha=0.4)
    #     p.set_facecolor(np.array(['royalblue', 'teal', 'gold', 'gold']))
    #     # p.set_edgecolor(np.array(['royalblue', 'teal', 'gold', 'gold']))
    #     p.set_alpha(1.)
    #     ax.add_collection(p)    
    #     path = np.array(self.episodes_information[-2]['observations'])
    #     # print('path :', self.episodes_information[-2])
    #     if self.episodes_information[-2]['status'] == 'failed':
    #         color = 'red'
    #     elif self.episodes_information[-2]['status'] == 'succeeded':
    #         color = 'lime'
    #     else:
    #         color = 'orange'
    #     ax.plot(path[:, 0], path[:, 1], color)
    #     # plt.savefig('./max_entropy_plots_/buffer_1')
    #     plt.draw()
        
        
    

































