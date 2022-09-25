import torch
import numpy as np
from torch import autograd
import altair as alt
from altair_saver import save 
import matplotlib.pyplot as plt
import pandas as pd
import glob, os
from tqdm import tqdm
alt.data_transformers.enable('default', max_rows=None)


device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
const = 1
save_folder_path = './ToyExperiments/figs/'

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

class RBF:
    def __init__(self, sigma=None):
        self.sigma = sigma

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
            sigma = const * torch.sqrt(h)
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma**2) 
        kappa = (-gamma * dist_sq).exp() 
        
        kappa_grad = -2. * (diff * gamma) * kappa
        return kappa.squeeze(), diff, h, kappa_grad, gamma

class AdamOptim():
    def __init__(self, lr=None):
        self.m_dx, self.v_dx = 0, 0
        self.lr = lr
    
    def step(self,x, dx): 
        dx = dx.view(x.size())
        x = x + self.lr * dx 
        return x

class Entropy_toy():
    def __init__(self, P, K, optimizer, num_particles, particles_dim, with_logprob):
        self.P = P
        self.optim = optimizer
        self.num_particles = num_particles
        self.particles_dim = particles_dim
        self.with_logprob = with_logprob
        self.K = K

        mu_ld_noise = torch.zeros((self.particles_dim,)) 
        sigma_ld_noise = torch.eye(self.particles_dim) * 0.1
        self.init_dist_ld = torch.distributions.MultivariateNormal(mu_ld_noise,covariance_matrix=sigma_ld_noise)
        self.identity_mat = torch.eye(self.particles_dim).to(device)



    def SVGD(self,X):
        X = X.requires_grad_(True)
        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X)[0].reshape(X.size())
        self.score_func = score_func.reshape(X.size())
        self.K_XX, self.K_diff, self.K_h, self.K_grad, self.K_gamma = self.K.forward(X, X)        
        self.num_particles=  X.size(0)
        self.phi_term1 = self.K_XX.matmul(score_func) / X.size(0)
        self.phi_term2 = self.K_grad.sum(0) / X.size(0)
        phi = self.phi_term1 + self.phi_term2
        return phi

    def LD(self,X, with_noise=None):
        X = X.requires_grad_(True)
        log_prob = self.P.log_prob(X).sum()
        ld = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0].reshape((self.num_particles, self.particles_dim))
        if with_noise:
            ld = ld + (np.sqrt(self.optim.lr) * self.init_dist_ld.sample((self.num_particles,))).to(device) / self.optim.lr
        return ld

    def compute_entropy(self, phi, X):
        grad_phi =[]
        for i in range(len(X)):
            #############################
            grad_phi_tmp = []
            for j in range(X.size(1)):
                grad_ = autograd.grad(phi[i][j], X, retain_graph=True)[0][i].detach()
                grad_phi_tmp.append(grad_)
                
            grad_phi_tmp = torch.stack(grad_phi_tmp)
            ##########################
            # grad_phi.append(torch.autograd.functional.hessian(lambda i : self.P.log_prob(i).sum(), X[i]).detach())
            grad_phi.append(grad_phi_tmp)
        
        self.grad_phi = torch.stack(grad_phi) 
        self.logp_line1 = -torch.log(torch.abs(torch.det(self.identity_mat + self.optim.lr * self.grad_phi)))

    # def compute_entropy(self, phi, X):
    #     grad_phi =[]
    #     for i in range(len(X)):
    #         grad_phi_tmp = []
    #         for j in range(X.size(1)):
    #             grad_ = autograd.grad(phi[i][j], X, retain_graph=True)[0][i].detach()
    #             grad_phi_tmp.append(grad_)
    #         grad_phi.append(torch.stack(grad_phi_tmp))
    #     self.grad_phi = torch.stack(grad_phi) 
    #     self.logp_line1 = -torch.log(torch.abs(torch.det(self.identity_mat + self.optim.lr * self.grad_phi)))

    def step(self, X, itr):
        # phi_X = self.SVGD(X) 
        phi_X = self.LD(X, with_noise=True) 
        # print('Phi :', phi_X[0])
        X_new = self.optim.step(X, phi_X) 
        if self.with_logprob: 
            self.compute_entropy(phi_X, X)
        X = X_new.detach()
        return X, phi_X 

################################# Experiment: Unimodal Gaussian ###############################

files = glob.glob(save_folder_path + '*')
[os.remove(file) for file in files]

lr = 0.5
dim = 2
n = 3

# Initial distribution of SVGD
mu = torch.zeros((dim,)) + 4
sigma_ = 1.
sigma = torch.eye(dim) * sigma_

init_dist = torch.distributions.MultivariateNormal(mu,covariance_matrix=sigma)
X_init = init_dist.sample((n,))

gauss = torch.distributions.MultivariateNormal(torch.Tensor([-0.6871,0.8010]).to(device),
    covariance_matrix=5 * torch.Tensor([[0.2260,0.1652],[0.1652,0.6779]]).to(device))

# init_dist_entr_gt = init_dist.entropy()
# init_dist_entr_by_hand = -init_dist.log_prob(X_init)

gauss_chart = get_density_chart(gauss, d=7.0, step=0.1) 
chart = gauss_chart + get_particles_chart(X_init.cpu().numpy())
save(chart,save_folder_path + 'unimodal_gaussian.pdf') 

X = X_init.clone()
experiment = Entropy_toy(gauss, RBF(), AdamOptim(lr), num_particles=n, particles_dim=dim, with_logprob=True) 

X_svgd_=[]
for t in tqdm(range(1000)): 
    X, phi_X = experiment.step(X, t)
    X_svgd_.append(X.clone())

    if (t%100)==0: 
        chart = gauss_chart + get_particles_chart(X.detach().cpu().numpy())
        chart_ = gauss_chart + get_particles_chart(X.detach().cpu().numpy(), torch.stack(X_svgd_).detach().cpu().numpy())
        save(alt.hconcat(chart, chart_),save_folder_path + 'unimodal_gaussian_svgd_sig_'+str(const)+ '_iteration ' + str(t) +'.pdf') 


X_svgd_ = torch.stack(X_svgd_)
chart = gauss_chart + get_particles_chart(X.detach().cpu().numpy())
chart_ = gauss_chart + get_particles_chart(X.detach().cpu().numpy(), X_svgd_.detach().cpu().numpy())
save(alt.hconcat(chart, chart_),save_folder_path + 'unimodal_gaussian_svgd_sig_'+str(const)+'.pdf') 

print('Stein at convergence: ', phi_X.mean() )
print('entropy gt: ', gauss.entropy() )  
print('logp_gt: ', gauss.log_prob(X).mean() )
print('entropy svgd/LD (line 1): ',  -(init_dist.log_prob(X_init) + experiment.logp_line1).mean())


