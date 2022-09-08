import numpy as np 
import torch as torch
import matplotlib.pyplot as plt
from utils import gaussian

class Debugger():
    def __init__(self, tb_logger, ac, env):
        # Still need some improvements that i will do tomorrow
        self.ac = ac
        self.tb_logger = tb_logger
        self.env = env
        self.episodes_information = []
            

    def collect_data(self, o, a, o2, r, d, info):
        
        if self.env.ep_len == 1:
            self.episodes_information.append({
                'observations':[],
                #'actions': [],
                'rewards': [],
                #'status': None,
                #'goal': None, 
                'mu': [],
                'sigma': [],
                'q_hess' : [],
                'q_score': [],
                'expected_reward': None, 
                'episode_length': None,
                'q_score_start': None, 
                'q_score_mid': None, 
                'q_score_end': None, 
                'q_hess_start': None, 
                'q_hess_mid': None, 
                'q_hess_end': None, 
                })
        self.episodes_information[-1]['observations'].append(o.detach().cpu().numpy().squeeze())
        #self.episodes_information[phase][-1]['actions'].append(a.detach().cpu().numpy().squeeze())
        self.episodes_information[-1]['rewards'].append(r)

        q1_value = self.ac.q1(o,a)
        q2_value = self.ac.q2(o,a)
        
        self.episodes_information[-1]['q1_values'] = q1_value.detach().cpu().numpy()
        self.episodes_information[-1]['q2_values'] = q2_value.detach().cpu().numpy()


        if self.ac.pi.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            self.episodes_information[-1]['mu'].append(self.ac.pi.mu.detach().cpu().numpy())
            self.episodes_information[-1]['sigma'].append(self.ac.pi.sigma.detach().cpu().numpy())
        
        
        grad_q_ = torch.autograd.grad(torch.min(q1_value, q2_value), a, retain_graph=True, create_graph=True)[0].squeeze()
        hess_q = ((torch.abs(torch.autograd.grad(grad_q_[0], a, retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_q_[1], a, retain_graph=True)[0])).sum()/4)
        self.episodes_information[-1]['q_score'].append(torch.abs(grad_q_).mean().detach().cpu().item())
        self.episodes_information[-1]['q_hess'].append(hess_q.detach().cpu().item())
        
        if d: 
            self.episodes_information[-1]['observations'].append(o2.squeeze())
            self.episodes_information[-1]['expected_reward'] = np.sum(self.episodes_information[-1]['rewards'])
            self.episodes_information[-1]['episode_length'] = self.env.ep_len
            if self.env.ep_len >= 5:
                self.episodes_information[-1]['q_score_start'] = np.mean(self.episodes_information[-1]['q_score'][:5])
                self.episodes_information[-1]['q_hess_start'] = np.mean(self.episodes_information[-1]['q_hess'][:5])
            if self.env.ep_len >= 17:
                self.episodes_information[-1]['q_score_mid'] = np.mean(self.episodes_information[-1]['q_score'][12:17])
                self.episodes_information[-1]['q_hess_mid'] = np.mean(self.episodes_information[-1]['q_hess'][12:17])
            if self.env.ep_len >= 30:
                self.episodes_information[-1]['q_score_end'] = np.mean(self.episodes_information[-1]['q_score'][25:self.env.ep_len])
                self.episodes_information[-1]['q_hess_end'] = np.mean(self.episodes_information[-1]['q_hess'][25:self.env.ep_len])

    def _init_plot(self):
        self.fig_env = plt.figure(figsize=(7, 7)) 
        self.ax = self.fig_env.add_subplot(111)
        self.ax.axis('equal')
        self.ax.set_xlim((-7, 7))
        self.ax.set_ylim((-7, 7))
        self.ax.set_title('Multigoal Environment')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        x_min, x_max = tuple(1.1 * np.array(self.env.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.env.ylim))

        X, Y = np.meshgrid(
            np.arange(x_min, x_max, 0.01),
            np.arange(y_min, y_max, 0.01)
        )

        goal_costs = np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.env.goal_positions
        ], axis=0)

        costs = goal_costs
        contours = self.ax.contour(X, Y, costs, 20)
        self.ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        self.ax.set_xlim([x_min, x_max])
        self.ax.set_ylim([y_min, y_max])
        self.ax.plot(self.env.goal_positions[:, 0], self.env.goal_positions[:, 1], 'ro')
        #goal = ax.plot(self.goal_positions[:, 0], self.goal_positions[:, 1], 'ro')

    def plot_policy(self, itr, fig_path, plot):
        if plot:
            self._init_plot()
            path = self.episodes_information[0]
            positions = np.stack(path['observations'])
            self.ax.plot(positions[:, 0], positions[:, 1], '+b')
            for i in range(len(positions)):
                self.ax.annotate(str(i), (positions[i,0], positions[i,1]), fontsize=6)
            for i in range(len(positions)-1):
                if self.ac.pi.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
                    mu = path['mu'][i][0]
                    std = path['sigma'][i][0]
                else:
                    mu = 0
                    std = 1

                x_values = np.linspace(positions[i] + mu + self.env.action_space.low, positions[i] + mu + self.env.action_space.high , 30)
                plt.plot(x_values[:,0] , gaussian(x_values, positions[i]+mu, std)[:,0])
            plt.savefig(fig_path + '/path_vis_'+ str(itr)+".pdf")   
            plt.close()


    def add_scalar(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_scalar(tb_path, value, itr)
    
    def add_scalars(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_scalars(tb_path, value, itr)

    def add_histogram(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_histogram(tb_path, value, itr)


    def log_to_tensorboard(self, itr):
        
        self.tb_logger.add_scalar('modes/num_modes',(self.env.number_of_hits_mode>0).sum(), itr)
        self.tb_logger.add_scalar('modes/total_number_of_hits_mode',self.env.number_of_hits_mode.sum(), itr)
        
        for ind in range(self.env.num_goals):
            self.tb_logger.add_scalar('modes/prob_mod_'+str(ind),self.env.number_of_hits_mode[ind]/self.env.number_of_hits_mode.sum() if self.env.number_of_hits_mode.sum() != 0 else 0.0, itr)
        

        q_score_ = list(map(lambda x: np.stack(x['q_score']), self.episodes_information))
        q_score_mean = list(map(lambda x: x.mean(), q_score_))
        q_score_min = list(map(lambda x: x.min(), q_score_))
        q_score_max = list(map(lambda x: x.max(), q_score_))

        q_hess_ = list(map(lambda x: np.stack(x['q_hess']), self.episodes_information))
        q_hess_mean = list(map(lambda x: x.mean(), q_hess_))
        q_hess_min = list(map(lambda x: x.min(), q_hess_))
        q_hess_max = list(map(lambda x: x.max(), q_hess_))
        
        self.tb_logger.add_scalars('smoothness/q_score',  {'Mean ': np.mean(q_score_mean), 'Min': np.mean(q_score_min), 'Max': np.mean(q_score_max)  }, itr)
        self.tb_logger.add_scalars('smoothness/q_hess', {'Mean ': np.mean(q_hess_mean), 'Min': np.mean(q_hess_min), 'Max': np.mean(q_hess_max)  }, itr)
        
        q_score_averaged = []
        q_hess_averaged = []

        for i in ['_start', '_mid', '_end']:
            q_score_i = np.array(list(map(lambda x: x['q_score' + i], self.episodes_information)))
            q_score_averaged.append(np.mean(q_score_i[q_score_i != np.array(None)]))

            q_hess_i = np.array(list(map(lambda x: x['q_hess' + i], self.episodes_information)))
            q_hess_averaged.append(np.mean(q_hess_i[q_hess_i != np.array(None)]))

        self.tb_logger.add_scalars('smoothness/q_score_averaged',  {'Start ': q_score_averaged[0], 'Mid': q_score_averaged[1], 'End': q_score_averaged[2] }, itr)
        self.tb_logger.add_scalars('smoothness/q_hess_averaged', {'Start ': q_hess_averaged[0], 'Mid': q_hess_averaged[1], 'End': q_hess_averaged[2] }, itr)

        expected_rewards = list(map(lambda x: x['expected_reward'], self.episodes_information))
        episode_length = list(map(lambda x: x['episode_length'], self.episodes_information))

        self.tb_logger.add_scalars('test_episode_return',  {'Mean ': np.mean(expected_rewards), 'Min': np.min(expected_rewards), 'Max': np.max(expected_rewards) }, itr)
        self.tb_logger.add_scalar('test_episode_length', np.mean(episode_length) , itr)
        
        self.episodes_information = []

    


    






