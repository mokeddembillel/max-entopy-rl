from gym.utils import EzPickle
from gym import spaces
from gym import Env
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import torch

class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * np.random.normal(size=self.s_dim)
        return state_next


class MultiGoalMaxEntropyEnv(Env, EzPickle): 
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """
    def __init__(self, goal_reward=10, actuation_cost_coeff=30.0, distance_cost_coeff=1.0, init_sigma=0.05, max_steps=None, plot_format=None):
        EzPickle.__init__(**locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.max_steps = max_steps
        # goal
        # self.goal_positions = np.array(((5, 0),(-5, 0),(0, 5),(0, -5)), dtype=np.float32)
        self.goal_positions = np.array(((5, 0),(-4, 3),(-4, -3)), dtype=np.float32)
        # self.goal_positions = np.array(((5, 0),(-4.582575, 2),(-4.582575, -2)), dtype=np.float32)
        
        self.num_goals = len(self.goal_positions)
        self.goal_threshold = 0.5 #1.0
        self.goal_reward = goal_reward
        # reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        # plotting
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        
        # logging
        self.episode_observations = [] 
        self.ep_len = 0
       
        # Plotter params, to be cleaned tomorrow. 
        # self._obs_lst = [[0,0],[-2.5,-2.5],[2.5,2.5]]
        self.entropy_obs_names = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
        self._obs_lst = np.array([
                [-3.5, -2.5],
                [-3.5, 0],
                [-3.5, 2.5],
                [-2, -1.5],
                [-2, 1.5],
                [-1, 0],
                [0, 0],
                [1, 0],
                [2, 0],
                [4, 0],
            ])
        self._n_samples = 100
        self.n_plots = len(self._obs_lst)
        self.x_size = (1 * self.n_plots + 1)
        self.y_size = 27
        self.agent_failure = None
        self.plot_format = plot_format

    def reset(self, init_state=None):
        if init_state:
            unclipped_observation = init_state
        else: 
            # unclipped_observation = (self.init_mu + self.init_sigma * np.random.normal(size=self.dynamics.s_dim)) ############################
            unclipped_observation = self.init_mu

        self.observation = np.clip(unclipped_observation, self.observation_space.low, self.observation_space.high)
        self.ep_len = 0
        return self.observation

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            dtype=np.float32,
            shape=(2,))

    @property
    def action_space(self):
        return spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim, ),
            dtype=np.float32)


    def step(self, action):
        # compute action
        action = action.ravel()
        action = np.clip(action, self.action_space.low, self.action_space.high).ravel()

        # compute observation
        self.observation = self.dynamics.forward(self.observation, action)
        self.observation = np.clip(self.observation, self.observation_space.low,self.observation_space.high)
        self.ep_len += 1

        # compute reward
        reward = self.compute_reward(self.observation, action)
        # reward = 0.0 ############# Need to be removed

        # compute done
        distance_to_goals = [np.linalg.norm(self.observation - goal_position)for goal_position in self.goal_positions]
        min_dist_index = np.argmin(distance_to_goals)
        done = distance_to_goals[min_dist_index] < self.goal_threshold
        
        # reward at the goal
        if done:
            reward += self.goal_reward

        if done or self.ep_len == self.max_steps:
            self.episode_observations.append(self.observation)
        
        return self.observation, reward, done, None
    

    def compute_reward(self, observation, action): 
        # penalize the L2 norm of acceleration
        # noinspection PyTypeChecker
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = self.observation
        # cur_position = np.array([0, 0])

        # noinspection PyTypeChecker
        goal_cost = self.distance_cost_coeff * np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward
    
    def reset_rendering(self):
        self.episode_observations = []
        self.number_of_hits_mode = np.zeros(self.num_goals)
        
    
    def render(self, itr, fig_path, plot, ac=None, paths=None):
        positions = np.stack(self.episode_observations)
        if plot:
            self._init_plot(self.x_size, self.y_size)
            self._ax_lst[0].plot(positions[:, 0], positions[:, 1], '+b')

            self.entropy_list = []
            for i in range(len(self._obs_lst)):
                o = torch.as_tensor(self._obs_lst[i], dtype=torch.float32).to(ac.pi.device).view(-1,1,self.observation_space.shape[0]).repeat(1,ac.pi.num_particles,1).view(-1,self.observation_space.shape[0])
                a, log_p = ac(o, deterministic=ac.pi.test_deterministic, with_logprob=True, all_particles=False)
                self.entropy_list.append(round(-log_p.detach().item(), 2))
            
            for i in range(len(self._obs_lst)):
                self._ax_lst[0].scatter(self._obs_lst[i, 0], self._obs_lst[i, 1], c='black', marker='$' + self.entropy_obs_names[i] + '$', s=100, zorder=2)
            # self._ax_lst[0].scatter(self._obs_lst[:, 0], self._obs_lst[:, 1], c=list(obs_colors/255.0), marker='a', s=100, zorder=2)
            for i in range(len(self._obs_lst)):
                self._ax_lst[0].annotate(str(self.entropy_list[i]), (self._obs_lst[i,0] + 0.1, self._obs_lst[i,1]), fontsize=12, color=[0,0,0], zorder=2)

            self._plot_level_curves(self._obs_lst, ac)
            self._plot_action_samples(ac)
            plt.plot()
            plt.savefig(fig_path+ '/env_' + str(itr) + '.' + self.plot_format)   
            plt.close()

        modes_dist = (((positions).reshape(-1,1,2) - np.expand_dims(self.goal_positions,0))**2).sum(-1)
        ind = modes_dist[np.where(modes_dist.min(-1)<1)[0]].argmin(-1)
        self.number_of_hits_mode[ind]+=1
    
    
    def _init_plot(self, x_size, y_size, grid_size=(7,3), debugging=False):
        self._ax_lst = []
        ###### Setup the environment plot ######
        self.fig_env = plt.figure(figsize=(x_size, y_size), constrained_layout=True) 
        self._ax_lst.append(plt.subplot2grid(grid_size, (0,0), colspan=3, rowspan=3))
        self._ax_lst[0].axis('equal')
        self._ax_lst[0].set_xlim(self.xlim)
        self._ax_lst[0].set_ylim(self.xlim)
        self._ax_lst[0].set_title('Multigoal Environment')
        self._ax_lst[0].set_xlabel('x')
        self._ax_lst[0].set_ylabel('y')
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, 0.01),
            np.arange(y_min, y_max, 0.01)
        )
        goal_costs = np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)
        costs = goal_costs
        contours = self._ax_lst[0].contour(X, Y, costs, 20)
        self._ax_lst[0].clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        self._ax_lst[0].set_xlim([x_min, x_max])
        self._ax_lst[0].set_ylim([y_min, y_max])
        self._ax_lst[0].plot(self.goal_positions[:, 0], self.goal_positions[:, 1], 'ro')
        if not debugging:
            ###### Setup Q Contours plot ######
            for i in range(self.n_plots):
                ax = plt.subplot2grid(grid_size, (3 + i//3,i%3))
                ax.set_xlim((-1, 1))
                ax.set_ylim((-1, 1))
                ax.grid(True)
                self._ax_lst.append(ax)
            self._line_objects = list()
        return self._ax_lst[0]

    def _plot_level_curves(self, _obs_lst, ac):
        # Create mesh grid.
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        xgrid, ygrid = np.meshgrid(xs, ys)
        a = np.concatenate((np.expand_dims(xgrid.ravel(), -1), np.expand_dims(ygrid.ravel(), -1)), -1)
        a = torch.from_numpy(a.astype(np.float32)).to(ac.pi.device)
        for i in range(len(_obs_lst)):
            o = torch.Tensor(_obs_lst[i]).repeat([a.shape[0],1]).to(ac.pi.device)
            with torch.no_grad():
                qs = ac.q1(o.to(ac.pi.device), a).cpu().detach().numpy()
            qs = qs.reshape(xgrid.shape)
            cs = self._ax_lst[i+1].contour(xgrid, ygrid, qs, 20)
            self._line_objects += cs.collections
            self._line_objects += self._ax_lst[i+1].clabel(
                cs, inline=1, fontsize=10, fmt='%.2f')

    def _plot_action_samples(self, ac):
        for i in range(len(self._obs_lst)):
            if ac.pi.actor == 'svgd_nonparam':
                o = torch.as_tensor(self._obs_lst[i], dtype=torch.float32).repeat([self._n_samples*ac.pi.num_particles,1]).to(ac.pi.device)
            else:
                o = torch.as_tensor(self._obs_lst[i], dtype=torch.float32).repeat([self._n_samples,1]).to(ac.pi.device)
            actions, _ = ac(o, deterministic=ac.pi.test_deterministic, with_logprob=False)
            actions = actions.cpu().detach().numpy().squeeze()
            x, y = actions[:, 0], actions[:, 1]
            self._ax_lst[i+1].title.set_text(str(self.entropy_obs_names[i] + '_' + str(self.entropy_list[i])))
            self._line_objects += self._ax_lst[i+1].plot(x, y, 'b*')
            
