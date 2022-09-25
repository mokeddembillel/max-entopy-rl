import torch
import numpy as np
from torch import autograd
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
import argparse

alt.data_transformers.enable('default', max_rows=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


def get_density_chart(P, d=7.0, step=0.1):
    xv, yv = torch.meshgrid([torch.arange(-d, d, step), torch.arange(-d, d, step)])
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
    p_xy = P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu()

    df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
    df = pd.DataFrame({
        'x': df[:, :, 0].ravel(),
        'y': df[:, :, 1].ravel(),
        'p': df[:, :, 2].ravel(),})

    chart = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    color=alt.Color('p:Q', scale=alt.Scale(scheme='viridis')),
    tooltip=['x','y','p'])

    return chart


def get_particles_chart(X, X_svgd=None):
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],})

    chart = alt.Chart(df).mark_circle(color='red').encode(x='x:Q',y='y:Q')

    if X_svgd is not None:
        #import pdb; pdb.set_trace()
        for i in range(np.shape(X_svgd)[1]):
            df_trajectory = pd.DataFrame({'x': X_svgd[:,i,0],'y': X_svgd[:,i,1],})
            chart += alt.Chart(df_trajectory).mark_line().mark_circle(color='green').encode(x='x:Q',y='y:Q')

    return chart

    from zipfile import ZIP_DEFLATED


class RBF:
    def __init__(self, sigma=None, zeta=1.):
        self.sigma = sigma
        self.zeta = zeta

    def forward(self, input_1, input_2):
        _, out_dim1 = input_1.size()[-2:]
        _, out_dim2 = input_2.size()[-2:]
        num_particles = input_2.size()[-2]
        assert out_dim1 == out_dim2
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)
        
        # Get median.
        if self.sigma is None:
            median_sq = torch.median(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1)[0]
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(num_particles + 1.))
            sigma = torch.sqrt(h)
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma**2) 


        kappa = (-gamma * dist_sq).exp() 
        
        kappa_grad = -2. * (diff * gamma) * kappa
        return kappa.squeeze(), diff, kappa_grad, gamma


class Optim():
    def __init__(self, lr=None):
        self.m_dx, self.v_dx = 0, 0
        self.lr = lr
    
    def step(self,x, dx): 
        dx = dx.view(x.size())
        x = x + self.lr * dx 
        return x

class Entropy_toy():
    def __init__(self, P, init_dist, K, optimizer, num_particles, particles_dim, with_logprob):
        self.P = P
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

        # experiment variables 
        self.init_dist = init_dist

    
    def SVGD(self,X):
        # Stein Variational Gradient Descent
        X = X.requires_grad_(True)
        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X)[0].reshape(X.size())
        self.score_func = score_func.reshape(X.size())
        self.K_XX, self.K_diff, self.K_grad, self.K_gamma = self.K.forward(X, X)        
        self.num_particles=  X.size(0)
        self.phi_term1 = self.K_XX.matmul(score_func) / X.size(0)
        self.phi_term2 = self.K_grad.sum(0) / X.size(0)
        phi = self.phi_term1 + self.phi_term2

        phi_entropy = (self.K_XX-torch.eye(X.size(0)).to(device)).matmul(score_func) / X.size(0) + self.phi_term2
        
        phi_entropy = phi_entropy * (torch.norm(phi_entropy, dim=1).view(-1,1) > 0.001).int() 

        return phi, phi_entropy

    def LD(self,X, with_noise=None):
        # Langevin Dynamics
        X = X.requires_grad_(True)
        log_prob = self.P.log_prob(X).sum()
        ld = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0].reshape((self.num_particles, self.particles_dim))
        if with_noise:
            ld = ld + (np.sqrt(self.optim.lr) * self.init_dist_ld.sample((self.num_particles,))).to(device) / self.optim.lr
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
        self.logp_line1 = self.logp_line1 - torch.log(torch.abs(torch.det(self.identity_mat + self.optim.lr * self.grad_phi)))

        
        # print('______________')
        # print('self.grad_phi (mean): ', self.grad_phi.mean())
        # print('self.grad_phi (min): ', self.grad_phi.min())
        # print('self.grad_phi (max): ', self.grad_phi.max())
        
        #print(' ')
        # print('self.logp_line1: ', - torch.log(torch.abs(torch.det(self.identity_mat + self.optim.lr * self.grad_phi))).mean() )
        
    def step(self, X, itr, alg=None):
        if alg == 'svgd':
            phi_X, phi_X_entropy = self.SVGD(X) 
        elif alg == 'ld':
            phi_X = self.LD(X, with_noise=True) 
        # print('Phi :', phi_X[0])
        X_new = self.optim.step(X, phi_X) 
        if self.with_logprob: 
            self.compute_entropy(phi_X_entropy, X)
        X = X_new.detach() 
        return X, phi_X 


# Main_loop
charts = []
def main_loop(alg, experiment, gauss_chart, X_init, steps):
    X = X_init.clone()
    
    entropy_svgd_all = []
    X_svgd_=[]
    for t in range(steps): 
        X, phi_X = experiment.step(X, t, alg)
        X_svgd_.append(X.clone())

        #if (t%10)==0: 
        #chart_ = gauss_chart + get_particles_chart(X.detach().cpu().numpy(), torch.stack(X_svgd_).detach().cpu().numpy())
        charts.append(gauss_chart + get_particles_chart(X.detach().cpu().numpy()))
        entropy_svgd_all.append(-(experiment.init_dist.log_prob(X_init) + experiment.logp_line1).mean().item())
        # print(t, ' entropy svgd (line 1): ',  entropy_svgd_all[-1])
        # print('init_dist_entr_GT ', experiment.init_dist.log_prob(X_init).mean()) 
        #print('sampler logprob ', experiment.logp_line1.mean()) 
        # print('_______')

    X_svgd_ = torch.stack(X_svgd_)

    # Plotting the results 
    chart = gauss_chart + get_particles_chart(X.detach().cpu().numpy())
    chart_ = gauss_chart + get_particles_chart(X.detach().cpu().numpy(), X_svgd_.detach().cpu().numpy())

    entropy_svgd_final = -(experiment.init_dist.log_prob(X_init) + experiment.logp_line1).mean().item()
    log_prob_svgd_final = experiment.logp_line1.mean().item()

    # print()
    print('entropy gt: ', experiment.P.entropy().item())  
    # print('entropy gt (logp): ', - experiment.P.log_prob(X).mean().item())  
    print('entropy svgd/LD (line 1): ',  entropy_svgd_final)
    # print()
    # print('init_dist_entr_GT ', experiment.init_dist.log_prob(X_init).mean().item()) 
    # print('sampler logprob ', log_prob_svgd_final) 
    return charts, entropy_svgd_final, entropy_svgd_all, log_prob_svgd_final


def run_experiment(exper_params, gauss, gauss_chart, expers_data, particles_dim, steps):
    
    num_particles = int(exper_params[0])
    lr = exper_params[1]
    kernel_sigma = exper_params[2]
    kernel_zeta = exper_params[3]
    init_dist_mu = exper_params[4]
    init_dist_sigma = exper_params[5]


    mu = torch.zeros((particles_dim,)) + init_dist_mu
    sigma = torch.eye(particles_dim) * init_dist_sigma

    init_dist = torch.distributions.MultivariateNormal(mu.to(device), covariance_matrix=sigma.to(device))
    X_init = init_dist.sample((num_particles,))

    init_chart = gauss_chart + get_particles_chart(X_init.cpu().numpy())
    

    experiment = Entropy_toy(gauss, init_dist, RBF(sigma=kernel_sigma, zeta=kernel_zeta), Optim(lr), num_particles=num_particles, particles_dim=particles_dim, with_logprob=True) 
    
    
    charts, entropy_svgd_final, entropy_svgd_all, log_prob_svgd_final = main_loop('svgd', experiment, gauss_chart, X_init, steps)


    expers_data['parameters'][-1]['num_particles'] = num_particles
    expers_data['parameters'][-1]['lr'] = lr
    expers_data['parameters'][-1]['kernel_sigma'] = kernel_sigma
    expers_data['parameters'][-1]['kernel_zeta'] = kernel_zeta
    expers_data['parameters'][-1]['init_dist_mu'] = init_dist_mu
    expers_data['parameters'][-1]['init_dist_sigma'] = init_dist_sigma
    expers_data['results'][-1]['init_chart'] = init_chart
    expers_data['results'][-1]['entropy_svgd_final'] = entropy_svgd_final
    expers_data['results'][-1]['log_prob_svgd_final'] = log_prob_svgd_final
    expers_data['results'][-1]['entropy_svgd_all'] = entropy_svgd_all
    expers_data['results'][-1]['charts'] = charts

def run_all_experiments():
    exper_num_particles = [5, 20, 200, 500]
    exper_lr = [0.1, 0.5]
    exper_kernel_sigmas = [0.01, 0.1, 1., 10., 100.]
    exper_kernel_zetas = [0.1, 1., 5, 10]
    exper_init_dist_mus = [0.,4.]
    exper_init_dist_sigmas = [6., 0.2]
    # exper_num_particles = [5]
    # exper_lr = [0.1, 0.5]
    # exper_kernel_sigmas = [0.01]
    # exper_kernel_zetas = [0.1]
    # exper_init_dist_mus = [0.]
    # exper_init_dist_sigmas = [6.]

    expers_params = np.array(np.meshgrid(exper_num_particles, exper_lr, exper_kernel_sigmas, exper_kernel_zetas, exper_init_dist_mus, exper_init_dist_sigmas)).T.reshape(-1,6)

    expers_data = {
        'parameters':[],
        'results':[]
    }

    particles_dim = 2
    steps = 500


    gauss = torch.distributions.MultivariateNormal(torch.Tensor([-0.6871,0.8010]).to(device),
        covariance_matrix=5 * torch.Tensor([[0.2260,0.1652],[0.1652,0.6779]]).to(device))
    gauss_chart = get_density_chart(gauss, d=7.0, step=0.1) 

    expers_data['entropy_gt'] = gauss.entropy().item()

    print('Number of experiments: ', expers_params.shape[0])
    for i in range(expers_params.shape[0]):
        print('Experiment num--> ', str(i+1), '/', expers_params.shape[0], ' num_particles--> ', expers_params[i][0],' lr--> ', expers_params[i][1],' kernel_sigma--> ', expers_params[i][2],' kernel_zeta--> ', expers_params[i][3],' init_dist_mu--> ', expers_params[i][4], ' init_dist_sigma--> ', expers_params[i][5])
        expers_data['parameters'].append({})
        expers_data['results'].append({})

        run_experiment(expers_params[i], gauss, gauss_chart, expers_data, particles_dim, steps)
        pickle.dump(expers_data, open('all_results.pkl', "wb"))
        print('Results Data Saved for iteration: ', i)
    
    print('Finished. Results Data Saved !')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_particles', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--kernel_sigma', type=float, default=None)
    parser.add_argument('--kernel_zeta', type=float, default=None)
    parser.add_argument('--init_dist_mu', type=float, default=None)
    parser.add_argument('--init_dist_sigma', type=float, default=None)
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()    

    exper_params = [args.num_particles, args.lr, args.kernel_sigma, args.kernel_zeta, args.init_dist_mu, args.init_dist_sigma]

    expers_data = {
        'parameters':[],
        'results':[]
    }

    particles_dim = 2
    steps = args.steps


    gauss = torch.distributions.MultivariateNormal(torch.Tensor([-0.6871,0.8010]).to(device),
        covariance_matrix=5 * torch.Tensor([[0.2260,0.1652],[0.1652,0.6779]]).to(device))
    gauss_chart = get_density_chart(gauss, d=7.0, step=0.1) 

    expers_data['entropy_gt'] = gauss.entropy().item()

    
    print('Experiment run : num_particles--> ', exper_params[0],' lr--> ', exper_params[1],' kernel_sigma--> ', exper_params[2],' kernel_zeta--> ', exper_params[3],' init_dist_mu--> ', exper_params[4], ' init_dist_sigma--> ', exper_params[5])
    expers_data['parameters'].append({})
    expers_data['results'].append({})


    # print()
    run_experiment(exper_params, gauss, gauss_chart, expers_data, particles_dim, steps)
    pickle.dump(expers_data, open('exper_result_files/results_experiment_num_particles_'+ str(exper_params[0])+ '_lr_'+ str(exper_params[1])+ '_kernel_sigma_'+ str(exper_params[2])+ '_kernel_zeta_'+ str(exper_params[3])+ '_init_dist_mu_'+ str(exper_params[4])+ '_init_dist_sigma_'+ str(exper_params[5])+ '_.pkl', "wb"))
    
    print('Finished. Results Data Saved !, results_experiment_num_particles_', str(exper_params[0]), '_lr_', str(exper_params[1]), '_kernel_sigma_', str(exper_params[2]), '_kernel_zeta_', str(exper_params[3]), '_init_dist_mu_', str(exper_params[4]), '_init_dist_sigma_', str(exper_params[5]), '_.pkl')


