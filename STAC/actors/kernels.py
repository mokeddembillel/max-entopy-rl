import torch
import numpy as np
from networks import mlp

class RBF(torch.nn.Module):
    def __init__(self, parametrized=False, act_dim=None, hidden_sizes=None, activation=torch.nn.ReLU, num_particles=None, sigma=None):
        super(RBF, self).__init__()
        self.parametrized = parametrized
        self.num_particles = num_particles
        self.sigma = sigma

        if parametrized:
            self.log_std_layer = mlp([num_particles*num_particles] + list(hidden_sizes) + [act_dim] , activation)
            self.log_std_min = 2
            self.log_std_max = -20
            

    def forward(self, input_1, input_2,  h_min=1e-3):
        _, out_dim1 = input_1.size()[-2:]
        _, out_dim2 = input_2.size()[-2:]
        num_particles = input_2.size()[-2]
        assert out_dim1 == out_dim2
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)
        
        if self.sigma is not None:
            sigma = self.sigma
        elif self.parametrized == False:
            # Get median.
            median_sq = torch.median(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1)[0]
            median_sq = median_sq.reshape(-1,1,1,1)
            h = median_sq / (2 * np.log(num_particles + 1.))
            sigma = torch.sqrt(h)
        else:
            log_std = self.log_std_layer(dist_sq)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            sigma = torch.exp(log_std)
        
        gamma = 1.0 / (1e-8 + 2 * sigma**2) 
        kappa = (-gamma * dist_sq).exp()
        kappa_grad = -2. * (diff * gamma) * kappa
        
        return kappa.squeeze(-1), diff, gamma, kappa_grad
