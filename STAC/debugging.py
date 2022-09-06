import numpy as np 
import torch as torch
import matplotlib.pyplot as plt
from utils import gaussian

class Debugger():
    def __init__(self, writer, ac, env):
        # Still need some improvements that i will do tomorrow
        self.ac = ac
        self.writer = writer
        self.env = env
        self.episodes_information = []
            

    def collect_data(self, o, a, r, d, info): # if we can pass ac from the beginning it would be better. I still didn't experiment with that
        
        if self.env.ep_len == 1:
            self.episodes_information.append({
                'observations':[],
                #'actions': [],
                #'rewards': [],
                #'status': None,
                #'goal': None, 
                'mu': [],
                'sigma': [],
                'q_hess' : [],
                'q_score': [],
                })
        self.episodes_information[-1]['observations'].append(o.detach().cpu().numpy().squeeze())
        #self.episodes_information[phase][-1]['actions'].append(a.detach().cpu().numpy().squeeze())
        #self.episodes_information[phase][-1]['rewards'].append(r)

        if self.ac.pi.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            self.episodes_information[-1]['mu'].append(self.ac.pi.mu.detach().cpu().numpy())
            self.episodes_information[-1]['sigma'].append(self.ac.pi.sigma.detach().cpu().numpy())
        
        
        grad_q_ = torch.autograd.grad(self.ac.q1(o,a), a, retain_graph=True, create_graph=True)[0].squeeze()
        hess_q = ((torch.abs(torch.autograd.grad(grad_q_[0], a, retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_q_[1], a, retain_graph=True)[0])).sum()/4)
        self.episodes_information[-1]['q_score'].append(torch.abs(grad_q_).mean().detach().cpu().numpy())
        self.episodes_information[-1]['q_hess'].append(hess_q.detach().cpu().numpy())
        
            


    def plot_policy(self, itr, fig_path):
        
        ax = self.env._init_plot()
        
        path = self.episodes_information[0]
            
        positions = np.stack(path['observations'])

        ax.plot(positions[:, 0], positions[:, 1], '+b')

        for i in range(len(positions)):
            ax.annotate(str(i), (positions[i,0], positions[i,1]), fontsize=6)

        for i in range(len(positions)-1):

            if self.ac.pi.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
                mu = path['mu'][i][0]
                std = path['sigma'][i][0]
            else:
                mu = 0
                std = 1

            x_values = np.linspace(positions[i]+mu+self.env.action_space.low, positions[i]+mu+self.env.action_space.high , 30) 
            plt.plot(x_values[:,0] , gaussian(x_values, positions[i]+mu, std)[:,0] )

        plt.savefig(fig_path + 'path_vis_'+ str(itr)+".pdf")   
        plt.close()

    def log_to_tensorboard_from_core(self, tb_path=None, value=None, operation=None):
        pass
        # every call to tensorboard from core with a call to this function
        # I will do it tomorrwo

    def log_to_tensorboard(self, itr):
        

        q_score_ = list(map(lambda x: np.stack(x['q_score']), self.episodes_information))
        q_score_mean = list(map(lambda x: x.mean(), q_score_))
        q_score_min = list(map(lambda x: x.min(), q_score_))
        q_score_max = list(map(lambda x: x.max(), q_score_))

        q_hess_ = list(map(lambda x: np.stack(x['q_hess']), self.episodes_information))
        q_hess_mean = list(map(lambda x: x.mean(), q_hess_))
        q_hess_min = list(map(lambda x: x.min(), q_hess_))
        q_hess_max = list(map(lambda x: x.max(), q_hess_))
        
        self.writer.add_scalar('modes/num_modes',(self.env.number_of_hits_mode>0).sum(), itr)
        self.writer.add_scalar('modes/total_number_of_hits_mode',self.env.number_of_hits_mode.sum(), itr)
        
        for ind in range(self.env.num_goals):
            self.writer.add_scalar('modes/prob_mod_'+str(ind),self.env.number_of_hits_mode[ind]/self.env.number_of_hits_mode.sum(), itr)
        
        self.writer.add_scalars('smoothness/q_score',  {'Mean ': np.mean(q_score_mean), 'Min': np.mean(q_score_min), 'Max': np.mean(q_score_max)  }, itr)
        self.writer.add_scalars('smoothness/q_hess', {'Mean ': np.mean(q_hess_mean), 'Min': np.mean(q_hess_min), 'Max': np.mean(q_hess_max)  }, itr)

        self.episodes_information = []

    


    






