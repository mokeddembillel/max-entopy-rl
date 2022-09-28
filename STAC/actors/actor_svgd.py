import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from networks import MLPSquashedGaussian
from torch.distributions import Normal, Categorical
from actors.kernels import RBF

class ActorSvgd(torch.nn.Module):
    def __init__(self, actor, obs_dim, act_dim, act_limit, num_svgd_particles, svgd_sigma_p0, num_svgd_steps, svgd_lr, test_deterministic, batch_size, 
    device, hidden_sizes, q1, q2, activation=torch.nn.ReLU, kernel_sigma=None, adaptive_lr=True):
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.num_particles = num_svgd_particles
        self.num_svgd_steps = num_svgd_steps
        self.svgd_lr = svgd_lr
        self.device = device
        self.q1 = q1
        self.q2 = q2
        self.sigma_p0 = svgd_sigma_p0

        self.test_deterministic = test_deterministic
        self.batch_size = batch_size

        #optimizer parameters
        self.adaptive_lr = adaptive_lr
        self.beta = 0.999
        self.v_dx = 0

        if actor == "svgd_nonparam":
            self.a0 = torch.normal(0, self.sigma_p0, size=(5 * batch_size * num_svgd_particles, self.act_dim)).to(self.device)
        else:
            self.p0 = MLPSquashedGaussian(obs_dim, act_dim, hidden_sizes, activation)

        if actor == "svgd_p0_kernel_pram":
            self.Kernel = RBF(True, act_dim, hidden_sizes, sigma=kernel_sigma)
        else:
            self.Kernel = RBF(num_particles=self.num_particles, sigma=kernel_sigma)
        
        # identity
        self.identity = torch.eye(self.num_particles).to(self.device)
        

    def svgd_optim(self, x, dx, dq): 
        dx = dx.view(x.size())

        if self.adaptive_lr and dq.mean()>1.0:
            self.v_dx = self.beta * self.v_dx + (1-self.beta) * dq**2
            v_x_hat = self.v_dx/(1-self.beta)
            self.svgd_lr = self.svgd_lr * 1/(torch.sqrt(v_x_hat)+1e-8)
            self.svgd_lr = self.svgd_lr.mean()
        
        # print('self.svgd_lr', self.svgd_lr)
        x = x + self.svgd_lr * dx
        return x

    def sampler(self, obs, a, with_logprob=True):
        logp = 0

        def phi(X):
            nonlocal logp
            X.requires_grad_(True)            
            log_prob1 = self.q1(obs, X)
            log_prob2 = self.q2(obs, X)
            log_prob = torch.min(log_prob1, log_prob2)
            score_func = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0]
            
            X = X.reshape(-1, self.num_particles, self.act_dim)
            score_func = score_func.reshape(X.size())
            K_XX, K_diff, K_gamma, K_grad = self.Kernel(X, X)
            phi = (K_XX.matmul(score_func) + K_grad.sum(1)) / self.num_particles 
            
            # compute the entropy
            if with_logprob:
                term1 = (K_grad * score_func.unsqueeze(1)).sum(-1).mean(2)
                term2 = -2 * K_gamma * ( (K_grad.permute(0,2,1,3) * K_diff).sum(-1) - self.num_particles * (K_XX-self.identity) ).mean(1)
                #import pdb; pdb.set_trace()#
                #tmp1 = (K_grad * score_func.reshape(-1,1,self.num_particles,self.act_dim)).sum(-1).mean(-1)
                #tmp2 = -2 * K_gamma.view(-1,1) * ((K_grad * K_diff).sum(-1) - self.act_dim * K_XX).mean(-1)
                logp = logp - self.svgd_lr * (term1 + term2) 
            
            return phi, log_prob, score_func 
        
        for t in range(self.num_svgd_steps):
            phi_, q_s_a, dq = phi(a)
            a = self.svgd_optim(a, phi_, dq)
            a = torch.clamp(a, -self.act_limit, self.act_limit).detach()
        
        a = self.act_limit * torch.tanh(a) 

        return a, logp, q_s_a

    def act(self, obs, deterministic=False, with_logprob=True, loss_q_=None):
        logp_a = None
        # logp_normal = None

        if self.actor == "svgd_nonparam":
            a0 = self.a0[torch.randint(len(self.a0), (len(obs),))]
            # a = torch.clamp(a, -self.act_limit, self.act_limit).detach()
        else:
            self.mu, self.sigma = self.p0(obs)
            a0 = Normal(self.mu, self.sigma).rsample()
            a0 = self.act_limit * torch.tanh(a0)

        # run svgd
        a, logp_svgd, q_s_a = self.sampler(obs, a0.detach(), with_logprob) 
        q_s_a = q_s_a.view(-1, self.num_particles)
        # compute the entropy 
        if with_logprob:
            logp_normal = - self.act_dim * 0.5 * np.log(2 * np.pi * self.sigma_p0) - (0.5 / self.sigma_p0) * (a0**2).sum(-1).view(-1,self.num_particles)
            logp_tanh = - ( 2 * (np.log(2) - a - F.softplus(-2 * a))).sum(axis=-1).view(-1,self.num_particles)
            logp_a = (logp_normal + logp_svgd + logp_tanh).mean(-1)
        
        self.a =  a.view(-1, self.num_particles, self.act_dim)

        # at test time
        if (with_logprob == False) and (deterministic == True):
            a = self.a[:,q_s_a.argmax(-1)]

        elif (with_logprob == False) and (deterministic == False):
            # beta = 1
            # soft_max_probs = torch.exp(beta * q_s_a - q_s_a.max(dim=1, keepdim=True)[0])
            # dist = Categorical(soft_max_probs/ torch.sum(soft_max_probs, dim=1, keepdim=True))
            # a = self.a[:,dist.sample()]
            a = self.a.view(-1, self.num_particles, self.act_dim)[:,np.random.randint(self.num_particles),:]
            
        else:
            a = self.a

        ########## Debugging. to be removed
        # a0 = torch.clamp(a0, -self.act_limit, self.act_limit).detach()
        # return a.view(-1, self.act_dim), logp_normal
        return a.view(-1, self.act_dim), logp_a

