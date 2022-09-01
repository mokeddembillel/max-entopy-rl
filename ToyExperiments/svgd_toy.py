import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import altair as alt
from altair_saver import save 


alt.data_transformers.enable('default', max_rows=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

const = 10
############################## Drawing Utilities ##################################

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


def get_particles_chart(X):
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],})

    chart = alt.Chart(df).mark_circle(color='red').encode(x='x:Q',y='y:Q')
    return chart


########################### RBF Kernel ###########################

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
        return kappa.squeeze(), diff, h, kappa_grad


################################# SVGD Sampler ###############################

class SVGD:
    def __init__(self, P, with_logprob=False):
        self.P = P
        self.K = RBF()
        self.logp = 0
        self.with_logprob = with_logprob
        self.act_limit = 1
        self.svgd_lr = 0.1

    def phi(self, X):
        X = X.requires_grad_(True)
        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X)[0].reshape(X.size())
        
        K_XX, K_diff, K_h, K_grad = self.K.forward(X, X.detach())        
        phi = (K_XX.detach().matmul(score_func) - K_grad.sum(1)) / X.size(0)
        phi = phi.detach()
        
        if self.with_logprob:
            #import pdb; pdb.set_trace()
            tmp1 = (K_grad * score_func.unsqueeze(1)).sum(-1).mean(-1).mean(-1)
            tmp2 = (-(2/K_h) * K_diff *  K_grad).sum(-1).mean(-1).mean(-1)
            tmp3 = ((1/K_h.squeeze(-1)) * self.act_limit * K_XX).mean(-1).mean(-1)
            self.logp -= self.svgd_lr*(tmp1 + tmp2 + tmp3)
        
        return phi 
    
    def svgd_optim(self, X, dX): 
        dX = dX.view(X.size())
        X = X + self.svgd_lr * dX
        return X
    
    def step(self, X):
        X = self.svgd_optim(X, self.phi(X))
        X = torch.clamp(X, -self.act_limit, self.act_limit).detach()
        return X 
        
        
        

################################# Experiment: Unimodal Gaussian ###############################
gauss = torch.distributions.MultivariateNormal(torch.Tensor([-0.6871,0.8010]).to(device),
    covariance_matrix=5 * torch.Tensor([[0.2260,0.1652],[0.1652,0.6779]]).to(device))

num_particles = 100
X_init = (3 * torch.randn(num_particles, *gauss.event_shape)).to(device)

gauss_chart = get_density_chart(gauss, d=7.0, step=0.1)

chart = gauss_chart + get_particles_chart(X_init.cpu().numpy())
save(chart,'./ToyExperiments/figs/unimodal_gaussian.pdf') 


X = X_init.clone() 
svgd = SVGD(gauss, with_logprob=True)

for t in range(1000):
    X = svgd.step(X)

chart = gauss_chart + get_particles_chart(X.detach().cpu().numpy())
save(chart,'./ToyExperiments/figs/unimodal_gaussian_k2_sig_'+str(const)+'.pdf') 

print('entropy gt: ', gauss.entropy() ) 
print('entropy svgd: ', svgd.logp ) 


################################# Experiment: Mixture of 2 Gaussians ###############################

class MoG(torch.distributions.Distribution):
    def __init__(self, loc, covariance_matrix):
        self.num_components = loc.size(0)
        self.loc = loc
        self.covariance_matrix = covariance_matrix

        self.dists = [
          torch.distributions.MultivariateNormal(mu, covariance_matrix=sigma)
          for mu, sigma in zip(loc, covariance_matrix)
        ]
        super(MoG, self).__init__(torch.Size([]), torch.Size([loc.size(-1)]))

    def log_prob(self, value):
        return torch.cat([p.log_prob(value).unsqueeze(-1) for p in self.dists], dim=-1).logsumexp(dim=-1)

    


class MoG2(MoG):
    def __init__(self, device=None):
        loc = torch.Tensor([[-5.0, 0.0], [5.0, 0.0]]).to(device)
        cov = torch.Tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(2, 1, 1).to(device)
        super(MoG2, self).__init__(loc, cov)
    
mog2 = MoG2(device=device)

num_particles = 100
X_init = (5 * torch.randn(num_particles, *mog2.event_shape)).to(device)

mog2_chart = get_density_chart(mog2, d=7.0, step=0.1)

X = X_init.clone()
svgd = SVGD(mog2, with_logprob=False)

for t in range(1000):
    X = svgd.step(X)

chart = (mog2_chart + get_particles_chart(X_init.cpu().numpy())) | (mog2_chart + get_particles_chart(X.detach().cpu().numpy()))
save(chart,'./ToyExperiments/figs/MoG2_k2_svgd2_sig_'+str(const)+'.pdf') 

print('entropy gt: ', -mog2.log_prob(X).mean() ) 
print('entropy svgd: ', svgd.logp ) 


################################# Experiment: Mixture of 6 Gaussians ###############################

class MoG6(MoG):
    def __init__(self, device=None):
        def _compute_mu(i):
            return 5.0 * torch.Tensor([[torch.tensor(i * math.pi / 3.0).sin(),torch.tensor(i * math.pi / 3.0).cos()]])

        loc = torch.cat([_compute_mu(i) for i in range(1, 7)], dim=0).to(device)
        cov = torch.Tensor([0.5, 0.5]).diag().unsqueeze(0).to(device).repeat(6, 1, 1) 

        super(MoG6, self).__init__(loc, cov)

mog6 = MoG6(device=device)

num_particles = 100
X_init = (5 * torch.randn(num_particles, *mog6.event_shape)).to(device)

mog6_chart = get_density_chart(mog6, d=7.0, step=0.1)

X = X_init.clone()
svgd = SVGD(mog6, with_logprob=False)

for t in range(1000):
    X = svgd.step(X, t)

chart = (mog6_chart + get_particles_chart(X_init.cpu().numpy())) | (mog6_chart + get_particles_chart(X.detach().cpu().numpy()))
save(chart,'./ToyExperiments/figs/MoG6_k2_svgd2_sig_'+str(const)+'.pdf')  

print('entropy gt: ', -mog6.log_prob(X).mean() ) 
print('entropy svgd: ', svgd.logp ) 


