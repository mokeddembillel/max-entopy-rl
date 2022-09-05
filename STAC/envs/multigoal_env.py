from gym.utils import EzPickle
from gym import spaces
from gym import Env
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import os

def gaussian(x, mu, sig):
    out = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    out = np.tanh(out)
    return out 

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


class MultiGoalEnv(Env, EzPickle): 
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """
    def __init__(self, writer=None, actor=None, goal_reward=10, actuation_cost_coeff=30.0,distance_cost_coeff=1.0,
                 init_sigma=0.05, max_steps=30):
        EzPickle.__init__(**locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.max_steps = max_steps
        # goal
        self.goal_positions = np.array(((5, 0),(-5, 0),(0, 5),(0, -5)), dtype=np.float32)
        self.num_goals = len(self.goal_positions)
        self.goal_threshold = 0.05 #1.0
        self.goal_reward = goal_reward
        # reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        # plotting
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        self._ax = None
        self._env_lines = []
        # logging
        self.writer = writer
        self.actor = actor
        self.episodes_information = []

        # remove the content of the fig folder
        figs = glob.glob('./STAC/multi_goal_plots_/*')
        [os.remove(fig) for fig in figs]
    
    def reset(self, init_state=None):
        if init_state:
            unclipped_observation = init_state
        else: 
            unclipped_observation = (self.init_mu + self.init_sigma * np.random.normal(size=self.dynamics.s_dim))

        self.observation = np.clip(unclipped_observation, self.observation_space.low, self.observation_space.high)
        
        self.episodes_information.append({'observations':[self.observation],
                            'actions': [],
                            'rewards': [],
                            'status': None,
                            'goal': None, 
                            'mu': [],
                            'sigma': [],
                            'q_hess' : [],
                            'q_score': [],
                            })
        
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
        self.episodes_information[-1]['actions'].append(action)

        # compute observation
        observation = self.dynamics.forward(self.observation, action)
        observation = np.clip(observation, self.observation_space.low,self.observation_space.high)
        self.episodes_information[-1]['observations'].append(observation)

        # compute reward
        reward = self.compute_reward(observation, action)

        # compute done
        dist_to_goal = np.amin([np.linalg.norm(observation - goal_position)for goal_position in self.goal_positions])
        done = dist_to_goal < self.goal_threshold
        
        # reward at the goal
        if done:
            reward += self.goal_reward
        
        self.episodes_information[-1]['rewards'].append(reward)

        self.observation = np.copy(observation)

        return observation, reward, done, {'pos': observation}
    

    def compute_reward(self, observation, action): 
        # penalize the L2 norm of acceleration
        # noinspection PyTypeChecker
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = observation

        # noinspection PyTypeChecker
        goal_cost = self.distance_cost_coeff * np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward
    

    def _plot_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))

        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )

        goal_costs = np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)

        costs = goal_costs

        contours = ax.contour(X, Y, costs, 20)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goal_positions[:, 0],
                       self.goal_positions[:, 1], 'ro')
        
        return [contours, goal]
    

    def _init_plot(self):
        fig_env = plt.figure(figsize=(7, 7)) 
        self._ax = fig_env.add_subplot(111)
        self._ax.axis('equal')

        self._env_lines = []
        self._ax.set_xlim((-7, 7))
        self._ax.set_ylim((-7, 7))

        self._ax.set_title('Multigoal Environment')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')

        self._plot_position_cost(self._ax)

    
    def reset_rendering(self,):
        self.episodes_information = []
        
    
    def render(self, itr):
        """Rendering the past rollouts of the environment.""" 
        self._init_plot()

        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = []
        
        # compute the number of runs reaching the goals
        number_of_hits_mode = np.zeros(self.num_goals)
        q_score, q_score_min, q_score_max = [], [], []
        q_hess, q_hess_min, q_hess_max = [], [], []
        
        for _, path in enumerate(self.episodes_information):
            
            positions = np.stack(path['observations'])

            self._env_lines += self._ax.plot(positions[-1, 0], positions[-1, 1], '+b')
            #compute the number of modes
            modes_dist = ((positions[-1].reshape(-1,2)-self.goal_positions)**2).sum(-1)

            if modes_dist.min()<1:
                number_of_hits_mode[modes_dist.argmin()]+=1 
            
            ###
            q_score_ = np.stack(path['q_score'])
            q_score.append(q_score_.mean())
            q_score_min.append(q_score_.min())
            q_score_max.append(q_score_.max())

            q_hess_ = np.stack(path['q_hess'])
            q_hess.append(q_hess_.mean())
            q_hess_min.append(q_hess_.min())
            q_hess_max.append(q_hess_.max())
        
        
        plt.savefig('./STAC/multi_goal_plots_/'+ str(itr)+".pdf")   
        plt.close()

        # logging
        #  log the number of hits across episodes for each of the modes
        self.writer.add_scalar('modes/num_modes',(number_of_hits_mode>0).sum(), itr)
        self.writer.add_scalar('modes/total_number_of_hits_mode',number_of_hits_mode.sum(), itr)
        
        for ind in range(self.num_goals):
            self.writer.add_scalar('modes/prob_mod_'+str(ind),number_of_hits_mode[ind]/number_of_hits_mode.sum(), itr)
        
        self.writer.add_scalars('smoothness/q_score',  {'Mean ': np.mean(q_score), 'Min': np.mean(q_score_min), 'Max': np.mean(q_score_max)  }, itr)
        self.writer.add_scalars('smoothness/q_hess', {'Mean ': np.mean(q_hess), 'Min': np.mean(q_hess_min), 'Max': np.mean(q_hess_max)  }, itr)
        

        
    def plot_policy(self, itr):
        """Rendering the past rollouts of the environment.""" 
        
        self._init_plot()
        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = []
        
        path = self.episodes_information[0]
            
        positions = np.stack(path['observations'])

        self._env_lines += self._ax.plot(positions[:, 0], positions[:, 1], '+b')

        for i in range(len(positions)):
            self._ax.annotate(str(i), (positions[i,0], positions[i,1]), fontsize=6)

        for i in range(len(positions)-1):

            if self.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
                mu = path['mu'][i][0]
                std = path['sigma'][i][0]
            else:
                mu = 0
                std = 1

            x_values = np.linspace(positions[i]+mu+self.action_space.low, positions[i]+mu+self.action_space.high , 30) 
            plt.plot(x_values[:,0] , gaussian(x_values, positions[i]+mu, std)[:,0] )

        plt.savefig('./STAC/multi_goal_plots_/path_vis_'+ str(itr)+".pdf")   
        plt.close()
            
    
    def collect_data_for_logging(self, ac, o, a):

        if self.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            self.episodes_information[-1]['mu'].append(ac.pi.mu.detach().cpu().numpy())
            self.episodes_information[-1]['sigma'].append(ac.pi.sigma.detach().cpu().numpy())
        

        if self.actor in  ['sac', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            grad_q_ = torch.autograd.grad(ac.q1(o,a), a, retain_graph=True, create_graph=True)[0].squeeze()
            hess_q = ((torch.abs(torch.autograd.grad(grad_q_[0], a, retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_q_[1], a, retain_graph=True)[0])).sum()/4)
            self.episodes_information[-1]['q_score'].append(torch.abs(grad_q_).mean().detach().cpu().numpy())
            self.episodes_information[-1]['q_hess'].append(hess_q.detach().cpu().numpy())
