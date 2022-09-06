import numpy as np 
import torch as torch
import matplotlib.pyplot as plt
from utils import gaussian

class Debugging():
    def __init__(self, fig_path):
        
        self.fig_path = fig_path

        self.episodes_information = {}
        for phase in ['train', 'test']:
            self.episodes_information[phase] = [{
                'observations':[],
                'actions': [],
                'rewards': [],
                'status': None,
                'goal': None, 
                'mu': [],
                'sigma': [],
                'q_hess' : [],
                'q_score': [],
                }]
            

    def collect_data(self, ac, o, a, r, d, info, phase=None): # if we can pass ac from the beginning it would be better. I still didn't experiment with that
        self.episodes_information[phase][-1]['observations'].append(o)
        self.episodes_information[phase][-1]['actions'].append(a)
        self.episodes_information[phase][-1]['rewards'].append(r)

        if self.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            self.episodes_information[phase][-1]['mu'].append(ac.pi.mu.detach().cpu().numpy())
            self.episodes_information[phase][-1]['sigma'].append(ac.pi.sigma.detach().cpu().numpy())
        
        if self.actor in  ['sac', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'svgd_sql']:
            grad_q_ = torch.autograd.grad(ac.q1(o,a), a, retain_graph=True, create_graph=True)[0].squeeze()
            hess_q = ((torch.abs(torch.autograd.grad(grad_q_[0], a, retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_q_[1], a, retain_graph=True)[0])).sum()/4)
            self.episodes_information[phase][-1]['q_score'].append(torch.abs(grad_q_).mean().detach().cpu().numpy())
            self.episodes_information[phase][-1]['q_hess'].append(hess_q.detach().cpu().numpy())
        
        if d:
            self.episodes_information[phase][-1]['goal'] = info['goal']
            self.episodes_information[phase][-1]['status'] = info['status']
            self.episodes_informations[phase].append({
            'observations':[],
            'actions': [],
            'rewards': [],
            'status': None,
            'goal': None, 
            'mu': [],
            'sigma': [],
            'q_hess' : [],
            'q_score': [],
            })


    def plot_path_with_gaussian(self, itr, phase=None):
        
        self._init_plot()
        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = []
        
        path = self.episodes_information[phase][0]
            
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

        plt.savefig(self.fig_path + 'path_vis_'+ str(itr)+".pdf")   
        plt.close()


    def log_data_to_tensorboard(self, itr, phase=None):

        q_score_ = np.array(list(map(lambda x: x['q_score'], self.episodes_information[phase])))
        q_score_mean = q_score_.mean(axis=1)
        q_score_min = q_score_.min(axis=1)
        q_score_max = q_score_.max(axis=1)


        q_hess_ = np.array(list(map(lambda x: x['q_hess'], self.episodes_information[phase])))
        q_hess_mean = q_hess_.mean(axis=1)
        q_hess_min = q_hess_.min(axis=1)
        q_hess_max = q_hess_.max(axis=1)

        number_of_hits_mode = np.array(list(map(lambda x: x['goal'], self.episodes_information[phase]))).sum(axis=0)

        self.writer.add_scalar('modes/num_modes',(number_of_hits_mode>0).sum(), itr)
        self.writer.add_scalar('modes/total_number_of_hits_mode',number_of_hits_mode.sum(), itr)
        
        for ind in range(self.num_goals):
            self.writer.add_scalar('modes/prob_mod_'+str(ind),number_of_hits_mode[ind]/number_of_hits_mode.sum(), itr)
        
        self.writer.add_scalars('smoothness/q_score',  {'Mean ': np.mean(q_score_mean), 'Min': np.mean(q_score_min), 'Max': np.mean(q_score_max)  }, itr)
        self.writer.add_scalars('smoothness/q_hess', {'Mean ': np.mean(q_hess_mean), 'Min': np.mean(q_hess_min), 'Max': np.mean(q_hess_max)  }, itr)

    


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


        self.writer.add_scalars('init_state/q_var',{'q_up': q_svgd_up_var, 'q_down':q_svgd_down_var, 'q_left':q_svgd_left_var, 'q_right':q_svgd_right_var}, itr)





    
 

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










