import math
import torch
import numpy as np
import altair as alt
import pandas as pd
import pickle


def get_density_chart(P, d=7.0, step=0.1, device=None):
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


def get_particles_chart(X, X_svgd=None, device=None):
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


class GMMDist(object):
    def __init__(self, dim, n_gmm, device):
        def _compute_mu(i):
            return 4.0 * torch.Tensor([[torch.tensor(i * math.pi / (n_gmm//2)).sin(),torch.tensor(i * math.pi / (n_gmm//2)).cos()]])
        
        self.mix_probs = 0.25 * torch.ones(n_gmm).to(device)
        # self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        # self.mix_probs = torch.tensor([0.1, 0.1, 0.8])
        # self.means = torch.stack([5 * torch.ones(dim), torch.zeros(dim), -torch.ones(dim) * 5], dim=0)
        self.means = torch.cat([_compute_mu(i) for i in range(n_gmm)], dim=0).to(device)
        #self.means = torch.stack([5 * torch.ones(dim).to(device), -torch.ones(dim).to(device) * 5], dim=0)
        self.sigma = 1.0
        self.std = torch.stack([torch.ones(dim).to(device) * self.sigma for i in range(len(self.mix_probs))], dim=0)

    def sample(self, n):
        n = n[0]
        mix_idx = torch.multinomial(self.mix_probs, n, replacement=True)
        means = self.means[mix_idx]
        stds = self.std[mix_idx]
        return torch.randn_like(means) * stds + means

    def log_prob(self, samples):
        logps = []
        for i in range(len(self.mix_probs)):
            logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * self.sigma ** 2) - 0.5 * np.log(
                2 * np.pi * self.sigma ** 2)) + self.mix_probs[i].log())
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp





# def run_all_experiments():
#     exper_num_particles = [5, 20, 200, 500]
#     exper_lr = [0.1, 0.5]
#     exper_kernel_sigmas = [0.01, 0.1, 1., 10., 100.]
#     exper_kernel_zetas = [0.1, 1., 5, 10]
#     exper_init_dist_mus = [0.,4.]
#     exper_init_dist_sigmas = [6., 0.2]
#     # exper_num_particles = [5]
#     # exper_lr = [0.1, 0.5]
#     # exper_kernel_sigmas = [0.01]
#     # exper_kernel_zetas = [0.1]
#     # exper_init_dist_mus = [0.]
#     # exper_init_dist_sigmas = [6.]

#     expers_params = np.array(np.meshgrid(exper_num_particles, exper_lr, exper_kernel_sigmas, exper_kernel_zetas, exper_init_dist_mus, exper_init_dist_sigmas)).T.reshape(-1,6)

#     expers_data = {
#         'parameters':[],
#         'results':[]
#     }

#     particles_dim = 2
#     steps = 500


#     gauss = torch.distributions.MultivariateNormal(torch.Tensor([-0.6871,0.8010]).to(device),
#         covariance_matrix=5 * torch.Tensor([[0.2260,0.1652],[0.1652,0.6779]]).to(device))
#     gauss_chart = get_density_chart(gauss, d=7.0, step=0.1) 

#     expers_data['entropy_gt'] = gauss.entropy().item()

#     print('Number of experiments: ', expers_params.shape[0])
#     for i in range(expers_params.shape[0]):
#         print('Experiment num--> ', str(i+1), '/', expers_params.shape[0], ' num_particles--> ', expers_params[i][0],' lr--> ', expers_params[i][1],' kernel_sigma--> ', expers_params[i][2],' kernel_zeta--> ', expers_params[i][3],' init_dist_mu--> ', expers_params[i][4], ' init_dist_sigma--> ', expers_params[i][5])
#         expers_data['parameters'].append({})
#         expers_data['results'].append({})

#         run_experiment(expers_params[i], gauss, gauss_chart, expers_data, particles_dim, steps)
#         pickle.dump(expers_data, open('all_results.pkl', "wb"))
#         print('Results Data Saved for iteration: ', i)
    
#     print('Finished. Results Data Saved !')











