import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, input_1, input_2,  h_min=1e-3):
        k_fix, out_dim1 = input_1.size()[-2:]
        k_upd, out_dim2 = input_2.size()[-2:]
        num_particles = input_2.size()[-2]
        assert out_dim1 == out_dim2
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)
        
        # Get median.
        median_sq = torch.median(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1)[0]
        median_sq = median_sq.reshape(-1,1,1,1)
        
        h = median_sq / (2 * np.log(num_particles + 1.))
        sigma = torch.sqrt(h)
        gamma = 1.0 / (1e-8 + 2 * sigma**2) 
        
        kappa = (-gamma * dist_sq).exp()
        kappa_grad = -2. * (diff * gamma) * kappa
        return kappa.squeeze(-1), diff, gamma, kappa_grad

class ActorSvgd():
    def __init__(self, actor, num_svgd_particles, num_svgd_steps, svgd_lr, test_deterministic):
        self.num_particles = num_svgd_particles
        self.num_svgd_steps = num_svgd_steps
        self.svgd_lr = svgd_lr
        self.test_deterministic = test_deterministic
        self.Kernel = RBF()

    def svgd_optim(self, x, dx): 
        dx = dx.view(x.size())
        x = x + self.svgd_lr * dx
        return x

    def sampler(self, obs, a, with_logprob=True, debug=False, itr=None):
        logp = 0

        def phi(X):
            nonlocal logp
            X = X.requires_grad_(True)
            log_prob = self.q1(obs, X)
            score_func = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0]
            
            X = X.reshape(-1, self.num_particles, self.act_dim)
            score_func = score_func.reshape(X.size())
            K_XX, K_diff, K_gamma, K_grad = self.Kernel(X, X.detach())
            phi = (K_XX.detach().matmul(score_func) - K_grad.sum(2)) / self.num_particles 
            
            # compute the entropy
            if with_logprob:
                #import pdb; pdb.set_trace()
                line_4 = (K_grad * score_func.reshape(-1,1,self.num_particles,self.act_dim)).sum(-1).mean(-1)
                line_5 = -2 * K_gamma.view(-1,1) * ((-K_grad * K_diff).sum(-1) - self.act_dim * K_XX).mean(-1)
                logp -= self.svgd_lr*(line_4+line_5)
            
            return phi 
        
        for t in range(self.num_svgd_steps):
            a = self.svgd_optim(a, phi(a))
            a = torch.clamp(a, -self.act_limit, self.act_limit).detach()

        return a, logp, phi(a)  

    def act(self, obs):
        obs = obs.view(-1,1,obs.size()[-1]).repeat(1,self.num_svgd_particles,1).view(-1,obs.size()[-1])

        # sample from a Gaussian
        a = torch.normal(0, 1, size=(len(obs) * self.num_svgd_particles, self.act_dim)).to(self.device)

        # entropy of a gaussian
        logp0_a2 = 0

        # svgd
        a2, logq_a2, phi_a2 = self.svgd_sampler(obs, a.detach()) 
        a2 = a2.detach()
        
        # compute the entropy 
        logp_a2 = logp0_a2.detach() + logq_a2.detach().mean(-1)

        return a2, logp_a2
        


class ActorSvgdNonParam(ActorSvgd):
    def __init__(self):
        ActorSvgd.__init__(self)
    def act(self, obs):
        return ActorSvgd.act(self, obs)


class ActorSvgdP0Param(ActorSvgd):
    def __init__(self):
        ActorSvgd.__init__(self)
    def act(self, obs):
        return ActorSvgd.act(self, obs)


class ActorSvgdP0KernelParam(ActorSvgd):
    def __init__(self):
        ActorSvgd.__init__(self)
    def act(self, obs):
        return ActorSvgd.act(self, obs)


