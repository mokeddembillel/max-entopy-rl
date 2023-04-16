# -*- coding: utf-8 -*-
"""Copy of Entropy_toy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bqVXC6JgiFEicRptOpTy-kB8G9Mr6tNX # this is an old version 

# Imports
"""
import math
import torch
import numpy as np
from torch import autograd
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from altair_saver import save as alt_save
import pickle
import argparse

# alt.data_transformers.enable('default', max_rows=None)
alt.data_transformers.enable('default')
alt.data_transformers.disable_max_rows()

from utils import get_density_chart, get_particles_chart, GMMDist

"""# Global Variables"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
const = 1

"""# Kernels"""

# RBF Kernel 
class RBF:
    def __init__(self, sigma=None):
        #self.sigma = sigma
        self.sigma = sigma #5.0#10.0 #sigma

    def forward(self, input_1, input_2):
        _, out_dim1 = input_1.size()[-2:]
        _, out_dim2 = input_2.size()[-2:]
        num_particles = input_2.size()[-2]
        assert out_dim1 == out_dim2
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)
        
        h = None
        # Get median.
        if self.sigma is None:
            median_sq = torch.median(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1)[0]
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(num_particles + 1.))
            sigma = const * torch.sqrt(h)
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma**2) 
        #gamma = gamma * 0.1
        
        kappa = (-gamma * dist_sq).exp() 
        
        #kappa_grad = -2. * (diff * gamma) * kappa
        #kappa = kappa - torch.eye(len(kappa)).unsqueeze(-1).to(device) 
        kappa_grad = -2. * (diff * gamma) * kappa

        return kappa.squeeze(), diff, h, kappa_grad, gamma

"""# Optimizer"""
class Optim():
    def __init__(self, lr=None):
        self.v_dx = 0
        self.lr = lr
        ####
        self.beta_2 = 0.999
        
    
    def step(self,x, dx, dq, adaptive_lr=None): 
        dx = dx.view(x.size())
        if not adaptive_lr:
            self.lr_coeff = self.lr
        ###
        else:
            self.v_dx = self.beta_2 * self.v_dx + (1-self.beta_2) * dq**2
            v_x_hat = self.v_dx/(1-self.beta_2)

            self.lr_coeff = self.lr * 1/(torch.sqrt(v_x_hat)+1e-8)
            self.lr_coeff = self.lr_coeff.mean()
        print("######################## ", self.lr_coeff)
        
        ### 
        x = x + self.lr_coeff * dx 
        return x

"""# Entropy Toy Class"""

class Entropy_toy():
    def __init__(self, P, init_dist, K, optimizer, num_particles, particles_dim, with_logprob):
        self.P = P
        self.init_dist = init_dist
        self.optim = optimizer
        self.num_particles = num_particles
        self.particles_dim = particles_dim
        self.with_logprob = with_logprob
        self.K = K

        # svgd 
        mu_ld_noise = torch.zeros((self.particles_dim,)) 
        sigma_ld_noise = torch.eye(self.particles_dim) * 0.05
        self.init_dist_ld = torch.distributions.MultivariateNormal(mu_ld_noise,covariance_matrix=sigma_ld_noise)
        self.identity_mat = torch.eye(self.particles_dim).to(device)

        
        # entropy
        self.logp_line1 = 0
        self.logp_line2 = 0
        self.logp_line3 = 0
        self.logp_line4 = 0
    
    def SVGD(self,X):
        # Stein Variational Gradient Descent
        X = X.requires_grad_(True)
        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X)[0].reshape(X.size())
        # print('************ score_func ', (score_func**2).sum(-1).mean() )

        self.score_func = score_func.reshape(X.size())
        self.K_XX, self.K_diff, self.K_h, self.K_grad, self.K_gamma = self.K.forward(X, X)        
        self.num_particles =  X.size(0)

        self.phi_term1 = self.K_XX.matmul(score_func) / X.size(0)
        self.phi_term2 = self.K_grad.sum(0) / X.size(0)
        
        phi = self.phi_term1 + self.phi_term2

        phi_entropy = (self.K_XX-torch.eye(X.size(0)).to(device)).matmul(score_func) / X.size(0) + self.phi_term2
        #phi_entropy = self.phi_term1 + self.phi_term2
        #phi_entropy = self.phi_term1
        # phi_entropy = phi_entropy * (torch.norm(phi_entropy, dim=1).view(-1,1) > 0.0001).int() 

        #print('kernel: ',self.K_XX[0])
        
        return phi, phi_entropy

    def LD(self,X, with_noise=None):
        # Langevin Dynamics
        X = X.requires_grad_(True)
        log_prob = self.P.log_prob(X).sum()
        ld = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0].reshape((self.num_particles, self.particles_dim))
        if with_noise:
            ld = ld + 2* (np.sqrt(self.optim.lr) * self.init_dist_ld.sample((self.num_particles,))).to(device) / self.optim.lr
        return ld

    def compute_entropy(self, phi, X):
        grad_phi =[]
        for i in range(len(X)):
            grad_phi_tmp = []
            for j in range(X.size(1)):
                grad_ = autograd.grad(phi[i][j], X, retain_graph=True)[0][i].detach()
                grad_phi_tmp.append(grad_)
            grad_phi.append(torch.stack(grad_phi_tmp))

        self.grad_phi = torch.stack(grad_phi) 
        self.logp_line1 = self.logp_line1 - torch.log(torch.abs(torch.det(self.identity_mat + self.optim.lr_coeff * self.grad_phi)))
        
        grad_phi_trace = torch.stack( [torch.trace(grad_phi[i]) for i in range(len(grad_phi))] ) 
        self.logp_line2 = self.logp_line2 - self.optim.lr_coeff * grad_phi_trace
        
        ######################################################################################################
        '''
        line3_term1 = []
        for i in range(len(self.score_func)):
            mat_1 = 0
            for j in range(len(self.score_func)):
                mat_1 += torch.trace(self.score_func[j].unsqueeze(-1).matmul( self.K_grad[i,j].unsqueeze(0)))
                #print(i," ",j, " ", torch.trace(self.score_func[j].unsqueeze(-1).matmul( self.K_grad[j,i].unsqueeze(0))) )
            #print('_______________________________')
            line3_term1.append(mat_1/len(self.score_func))
            
        line3_term1 = torch.stack(line3_term1)

        
        line3_term2 = []
        for i in range(len(self.score_func)):
            mat_2 = 0
            for j in range(len(self.score_func)):
                mat_2 += torch.trace( (-2 * self.K_gamma) * ( self.K_diff[j,i].unsqueeze(-1).matmul( self.K_grad[i,j].unsqueeze(0) )- torch.eye(2).to(device)* (self.K_XX-torch.eye(len(self.K_XX)).to(device))[j,i])    )
                #print(i," ",j, " ", torch.trace(self.score_func[j].unsqueeze(-1).matmul( self.K_grad[j,i].unsqueeze(0))) )
            line3_term2.append(mat_2/len(self.score_func))
        
        line3_term2 = torch.stack(line3_term2)
        self.logp_line3 = self.logp_line3 - self.optim.lr_coeff * (line3_term1 + line3_term2)
        #self.logp_line3 = self.logp_line3 - self.optim.lr_coeff *  line3_term1
        #self.logp_line3 = self.logp_line3 - self.optim.lr_coeff *  line3_term2
        
        #line3_term1 = (self.K_grad * self.score_func.unsqueeze(1)).sum(-1).mean(0)
        #line3_term2 = -2 * self.K_gamma * ((self.K_grad * self.K_diff).sum(-1) - X.size(1) * (self.K_XX-torch.eye(X.size(0)).to(device)) ).mean(0)
        #self.logp_line3 = self.logp_line4 - self.optim.lr_coeff * (line4_term1 + line4_term2)
        '''
        ######################################################################################################
        # line4_term1_b = (self.K_grad * self.score_func.unsqueeze(1)).sum(-1).mean(0)
        line4_term1 = (self.K_grad * self.score_func.unsqueeze(0)).sum(-1).mean(1)
        # line4_term1_a = (self.K_grad.permute(1,0,2) * self.score_func.unsqueeze(1)).sum(-1).mean(0)
        line4_term2 = -2 * self.K_gamma * (( self.K_grad.permute(1,0,2) * self.K_diff).sum(-1) - X.size(1) * (self.K_XX-torch.eye(len(self.K_XX)).to(device)) ).mean(0)
        
        
        self.logp_line4 = self.logp_line4 - self.optim.lr_coeff * (line4_term1 + line4_term2)
        #self.logp_line4 = self.logp_line4 - self.optim.lr_coeff * line4_term1
        #self.logp_line4 = self.logp_line4 - self.optim.lr_coeff * line4_term2


    def step(self, X, itr, alg=None, adaptive_lr=None):
        if alg == 'svgd':
            phi_X, phi_X_entropy = self.SVGD(X) 
        elif alg == 'ld':
            phi_X = self.LD(X, with_noise=False) 
            phi_X_entropy = phi_X
        
        # print('Phi :', phi_X[0])
        X_new = self.optim.step(X, phi_X, self.score_func.mean(0), adaptive_lr=adaptive_lr) 

        if self.with_logprob: 
            self.compute_entropy(phi_X_entropy, X)
        
        X = X_new.detach() 
        
        return X, phi_X


def run_experiment(exper_params, gauss, gauss_chart, expers_data, particles_dim, steps, lr_adaptive):
    
    num_particles = int(exper_params[0])
    lr = exper_params[1]
    kernel_sigma = exper_params[2]
    init_dist_mu = exper_params[3]
    init_dist_sigma = exper_params[4]
    target_dist_sigma = exper_params[5]
    gmm = exper_params[6]


    mu = torch.zeros((particles_dim,)) + init_dist_mu
    sigma = torch.eye(particles_dim) * init_dist_sigma

    init_dist = torch.distributions.MultivariateNormal(mu.to(device), covariance_matrix=sigma.to(device))
    X_init = init_dist.sample((num_particles,))

    init_chart = gauss_chart + get_particles_chart(X_init.cpu().numpy())
    

    experiment = Entropy_toy(gauss, init_dist, RBF(sigma=kernel_sigma), Optim(lr), num_particles=num_particles, particles_dim=particles_dim, with_logprob=True) 

    
    charts, line_1, line_2, line_4 = main_loop('svgd', experiment, gauss_chart, X_init, steps,lr_adaptive)


    """# Run Lengevin Dynamics"""
    #charts = main_loop('ld', X_init.clone(), steps=200)
    #charts[-1]
    plt.plot(np.arange(len(line_1)),line_1, c="r", label="line_1")
    plt.plot(np.arange(len(line_2)),line_2, c="b", label="line_2")
    plt.title('Comparison between the entropy of Line 1 and Line 2')
    plt.xlabel('Training iterations')
    plt.ylabel('Entropy')
    plt.legend()
    plt.savefig('./ToyExperiments/figs/line_1_line_2.png')
    plt.close()

    expers_data['parameters'][-1]['num_particles'] = num_particles
    expers_data['parameters'][-1]['lr'] = lr
    expers_data['parameters'][-1]['kernel_sigma'] = kernel_sigma
    expers_data['parameters'][-1]['init_dist_mu'] = init_dist_mu
    expers_data['parameters'][-1]['init_dist_sigma'] = init_dist_sigma
    expers_data['parameters'][-1]['target_dist_sigma'] = target_dist_sigma
    expers_data['parameters'][-1]['gmm'] = gmm
    expers_data['results'][-1]['init_chart'] = init_chart
    expers_data['results'][-1]['charts'] = charts
    expers_data['results'][-1]['line_1'] = line_1
    expers_data['results'][-1]['line_2'] = line_2
    expers_data['results'][-1]['line_4'] = line_4


def main_loop(alg, experiment, gauss_chart, X_init, steps, adaptive_lr):
    X = X_init.clone()
    print('steps ', steps)

    charts = []
    line_1 = []
    line_2 = []
    line_3 = []
    line_4 = []
    X_svgd_ = []

    for t in range(steps):
        print(t)
        X, _ = experiment.step(X, t, alg, adaptive_lr)
        X_svgd_.append(X.clone())
        
        line_1.append( -(experiment.init_dist.log_prob(X_init) + experiment.logp_line1).mean().item())
        line_2.append( -(experiment.init_dist.log_prob(X_init) + experiment.logp_line2).mean().item())
        # line_3.append( -(init_dist.log_prob(X_init) + experiment.logp_line3).mean().item() )
        line_4.append( -(experiment.init_dist.log_prob(X_init) + experiment.logp_line4).mean().item())
        print(t, ' entropy svgd (line 1): ',  line_1[-1])
        print(t, ' entropy svgd (line 2): ',  line_2[-1])
        #print(t, ' entropy svgd (line 3): ',  line_3[-1])
        print(t, ' entropy svgd (line 4): ',  line_4[-1])
        #print('sampler logprob ', experiment.logp_line1.mean()) 
        #X_svgd_ = torch.stack(X_svgd_)
        # Plotting the results 
        
        if (t%100)==0: 
            chart = gauss_chart + get_particles_chart(X.detach().cpu().numpy(), device=device)
            alt_save(chart, "./ToyExperiments/figs/gmm_"+str(gmm)+"_"+str(t)+".png")  
            charts.append(chart)
        
        #chart_ = gauss_chart + get_particles_chart(X.detach().cpu().numpy(), X_svgd_.detach().cpu().numpy())
        print()
        # print('entropy gt: ', gauss.entropy().item())  
        print('entropy gt (logp): ', - gauss.log_prob(X).mean())  
        print('entropy svgd/LD (line 1): ',  -(experiment.init_dist.log_prob(X_init) + experiment.logp_line1).mean().item())
        print()
        print('init_dist_entr_GT ', experiment.init_dist.log_prob(X_init).mean()) 
        print('sampler logprob ', experiment.logp_line1.mean()) 
    return charts, line_1, line_2, line_4


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_particles', type=float, default=200)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--lr_adaptive', type=float, default=True)
    parser.add_argument('--kernel_sigma', type=float, default=5)
    parser.add_argument('--init_dist_mu', type=float, default=None)
    parser.add_argument('--init_dist_sigma', type=float, default=None)
    parser.add_argument('--targer_dist_sigma', type=float, default=1.)
    parser.add_argument('--target_dist_sigma', type=float, default=None)
    parser.add_argument('--gmm', type=float, default=2)

    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()    


    expers_data = {
        'parameters':[],
        'results':[]
    }

    particles_dim = 2
    steps = args.steps
    gmm = args.gmm

    if (gmm == 1):
        args.init_dist_mu = 4 
        args.init_dist_sigma = 0.2 #6
        args.target_dist_sigma = 1.0
        gauss = torch.distributions.MultivariateNormal(torch.Tensor([0.0,0.0]).to(device),covariance_matrix= args.target_dist_sigma * torch.Tensor([[1.0,0.0],[0.0,1.0]]).to(device))
    else:
        args.init_dist_mu = 0
        args.init_dist_sigma = 0.2 #6
        gauss = GMMDist(dim=2, n_gmm=gmm, device=device)

    exper_params = [args.num_particles, args.lr, args.kernel_sigma, args.init_dist_mu, args.init_dist_sigma, args.target_dist_sigma, args.gmm]

    gauss_chart = get_density_chart(gauss, d=7.0, step=0.1, device=device) 

    # expers_data['entropy_gt'] = gauss.entropy().item()

    
    print('Experiment run : num_particles--> ', exper_params[0],' lr--> ', exper_params[1],' kernel_sigma--> ', exper_params[2],' init_dist_mu--> ', exper_params[3], ' init_dist_sigma--> ', exper_params[4], ' target_dist_sigma--> ', exper_params[5], ' gmm--> ', exper_params[6])
    expers_data['parameters'].append({})
    expers_data['results'].append({})


    # print()
    run_experiment(exper_params, gauss, gauss_chart, expers_data, particles_dim, steps, args.lr_adaptive)

    pickle.dump(expers_data, open('ToyExperiments/exper_result_files/results_experiment_num_particles_'+ str(exper_params[0])+ '_lr_'+ str(exper_params[1])+ '_kernel_sigma_'+ str(exper_params[2])+ '_init_dist_mu_'+ str(exper_params[3])+ '_init_dist_sigma_'+ str(exper_params[4])+ '_target_dist_sigma_' + str(exper_params[5]) + '_gmm_' + str(exper_params[6]) + '_.pkl', "wb"))
    
    print('Finished. Results Data Saved !, results_experiment_num_particles_', str(exper_params[0]), '_lr_', str(exper_params[1]), '_kernel_sigma_', str(exper_params[2]), '_init_dist_mu_', str(exper_params[3]), '_init_dist_sigma_', str(exper_params[4]), '_target_dist_sigma_' , str(exper_params[5]) , '_gmm_' , str(exper_params[6]), '_.pkl')




