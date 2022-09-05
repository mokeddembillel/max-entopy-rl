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
                            'q_hess_eig_max': [],
                            'q_particles_var': [],
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
        
        for _, path in enumerate(self.episodes_information):
            
            positions = np.stack(path['observations'])

            self._env_lines += self._ax.plot(positions[-1, 0], positions[-1, 1], '+b')
            #compute the number of modes
            modes_dist = ((positions[-1].reshape(-1,2)-self.goal_positions)**2).sum(-1)

            if modes_dist.min()<1:
                number_of_hits_mode[modes_dist.argmin()]+=1 
        
        plt.savefig('./STAC/multi_goal_plots_/'+ str(itr)+".pdf")   
        plt.close()

        # logging
        #  log the number of hits across episodes for each of the modes
        self.writer.add_scalar('modes/num_modes',(number_of_hits_mode>0).sum(), itr)
        self.writer.add_scalar('modes/total_number_of_hits_mode',number_of_hits_mode.sum(), itr)
        
        for ind in range(self.num_goals):
            self.writer.add_scalar('modes/prob_mod_'+str(ind),number_of_hits_mode[ind]/number_of_hits_mode.sum(), itr)
        
        ############ no hess variables for sac for now ###########
        # ac_hess_list = torch.flatten(list(map(lambda x: x['ac_hess_list'], self.episodes_information)))
        # ac_score_func_list = torch.flatten(list(map(lambda x: x['ac_score_func_list'], self.episodes_information)))
        # ac_hess_eig_max = torch.flatten(list(map(lambda x: x['ac_hess_eig_max'], self.episodes_information)))

        # # 
        # self.writer.add_scalar('smoothness/ac_score/mean', torch.abs(ac_score_func_list).mean() , itr)
        # self.writer.add_scalar('smoothness/ac_score/std', torch.abs(ac_score_func_list).std() , itr)
        # self.writer.add_scalar('smoothness/hess/mean', torch.abs(ac_hess_list).mean() , itr)
        # self.writer.add_scalar('smoothness/hess/std', torch.abs(ac_hess_list).std() , itr)
        # self.writer.add_scalar('smoothness/hess/max_eigen_val/mean', ac_hess_eig_max.mean() , itr)
        # self.writer.add_scalar('smoothness/hess/max_eigen_val/std', ac_hess_eig_max.std() , itr)

        
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
            
    
    def collect_data_for_logging(self, ac):
        if self.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            self.episodes_information[-1]['mu'].append(ac.pi.mu.detach().cpu().numpy())
            self.episodes_information[-1]['sigma'].append(ac.pi.sigma.detach().cpu().numpy())
        

        if self.actor in  ['sac', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            self.episodes_information[-1]['q_hess'].append(ac.pi.hess_list)
            self.episodes_information[-1]['q_score'].append(ac.pi.score_func_list)
            self.episodes_information[-1]['q_hess_eig_max'].append(ac.pi.hess_eig_max)
            self.episodes_information[-1]['q_particles_var'].append(ac.pi.hess_eig_max)
        
        
    
    def debugging_metrics(self, itr, ac, num_svgd_particles):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        init_state = torch.tensor([0.0,0.0]).to(device)
        a_up = torch.tensor([0.0,0.7]).to(device)
        a_down = torch.tensor([0.0,-0.7]).to(device)
        a_left = torch.tensor([-0.7,0.0]).to(device)
        a_right = torch.tensor([0.7,0.0]).to(device)

        q_up = ac.q1(init_state,a_up).detach()
        q_down = ac.q1(init_state,a_down).detach()
        q_left = ac.q1(init_state,a_left).detach()
        q_right = ac.q1(init_state,a_right).detach()
        self.writer.add_scalars('init_state/q_val',{'q_up': q_up, 'q_down':q_down, 'q_left':q_left, 'q_right':q_right}, itr)

        def compute_q(init_s, action):
            return ac.q1(init_s,action)


        # add curvature and gradient
        a_up_ = a_up.requires_grad_(True)
        init_state_ = init_state.requires_grad_(True)
        grad_up_ = torch.autograd.grad(ac.q1(init_state_,a_up_), a_up_,retain_graph=True, create_graph=True)[0]
        grad_up = torch.abs(grad_up_).mean()
        hess_up = ((torch.abs(torch.autograd.grad(grad_up_[0],a_up_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_up_[1],a_up_,retain_graph=True)[0])).sum()/4).detach()
        #hess_up = torch.autograd.functional.hessian(compute_q, (init_state_,a_up_) )
        #hess_up = torch.abs(hess_up[1][1]).mean().detach()
        ###
        a_down_ = a_down.requires_grad_(True)
        grad_down_ = torch.autograd.grad(ac.q1(init_state_,a_down_), a_down_,retain_graph=True, create_graph=True)[0]
        grad_down = torch.abs(grad_down_).mean()
        hess_down = ((torch.abs(torch.autograd.grad(grad_down_[0],a_down_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_down_[1],a_down_,retain_graph=True)[0])).sum()/4).detach()
        #hess_down = torch.autograd.functional.hessian(compute_q, (init_state_,a_down_) )
        #hess_down = torch.abs(hess_down[1][1]).mean().detach()
        ###
        a_left_ = a_left.requires_grad_(True)
        grad_left_ = torch.autograd.grad(ac.q1(init_state_,a_left_), a_left_,retain_graph=True, create_graph=True)[0]
        grad_left = torch.abs(grad_left_).mean()
        hess_left = ((torch.abs(torch.autograd.grad(grad_left_[0],a_left_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_left_[1],a_left_,retain_graph=True)[0])).sum()/4).detach()
        #hess_left = torch.autograd.functional.hessian(compute_q, (init_state_,a_left_) )
        #hess_left = torch.abs(hess_left[1][1]).mean().detach()
        ###
        a_right_ = a_right.requires_grad_(True)
        grad_right_ = torch.autograd.grad(ac.q1(init_state_,a_right_), a_right_,retain_graph=True, create_graph=True)[0]
        grad_right = torch.abs(grad_right_).mean()
        hess_right = ((torch.abs(torch.autograd.grad(grad_right_[0],a_right_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_right_[1],a_right_,retain_graph=True)[0])).sum()/4).detach()
        #hess_right = torch.autograd.functional.hessian(compute_q, (init_state_,a_right_) )
        #hess_right = torch.abs(hess_right[1][1]).mean().detach()
        
        #import pdb; pdb.set_trace()  
        self.writer.add_scalars('init_state/hessian',{'hess_up': hess_up, 'hess_down':hess_down, 'hess_left':hess_left, 'hess_right':hess_right}, itr)
        self.writer.add_scalars('init_state/grad',{'grad_up': grad_up, 'grad_down':grad_down, 'grad_left':grad_left, 'grad_right':grad_right}, itr)
        
        
        # compute the variance of running svgd
        num_samples = 100

        s_up = a_up.view(-1,1,a_up.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_up.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_up),2)).to(device)
        a_svgd_up, _, _ = ac.pi(s_up, a_rand.detach()) 
        q_svgd_up = ac.q1(s_up,a_svgd_up).detach()
        q_svgd_up_var = torch.var(q_svgd_up)
        
        s_down = a_down.view(-1,1,a_down.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_down.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_down),2)).to(device)
        a_svgd_down, _, _ = ac.pi(s_down, a_rand.detach()) 
        q_svgd_down = ac.q1(s_down,a_svgd_down).detach()
        q_svgd_down_var = torch.var(q_svgd_down)
        
        s_left = a_left.view(-1,1,a_left.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_left.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_left),2)).to(device)
        a_svgd_left, _, _ = ac.pi(s_left, a_rand.detach()) 
        q_svgd_left = ac.q1(s_left,a_svgd_left).detach()
        q_svgd_left_var = torch.var(q_svgd_left)
        
        s_right = a_left.view(-1,1,a_right.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_right.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_right),2)).to(device)
        a_svgd_right, _, _ = ac.pi(s_right, a_rand.detach()) 
        q_svgd_right = ac.q1(s_right,a_svgd_right).detach()
        q_svgd_right_var = torch.var(q_svgd_right)

        #if itr==2000:
        #    import pdb; pdb.set_trace()
        self.writer.add_scalars('init_state/q_var',{'q_up': q_svgd_up_var, 'q_down':q_svgd_down_var, 'q_left':q_svgd_left_var, 'q_right':q_svgd_right_var}, itr)



