import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from networks import MLPSquashedGaussian
from torch.distributions import Normal, Categorical
from actors.kernels import RBF
from utils import GMMDist
import timeit
import math 

class ActorSvgd(torch.nn.Module):
    def __init__(self, actor, obs_dim, act_dim, act_limit, num_svgd_particles, svgd_sigma_p0, num_svgd_steps, svgd_lr, test_action_selection, batch_size, adaptive_sig,
    device, hidden_sizes, q1, q2, activation=torch.nn.ReLU, kernel_sigma=None, adaptive_lr=None):
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.num_particles = num_svgd_particles
        self.num_svgd_steps = num_svgd_steps
        self.svgd_lr_init = svgd_lr
        self.device = device
        self.q1 = q1
        self.q2 = q2
        self.sigma_p0 = svgd_sigma_p0
        self.test_action_selection = test_action_selection
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
            self.Kernel = RBF(True, act_dim, hidden_sizes, sigma=kernel_sigma, device=device)
        else:
            self.Kernel = RBF(num_particles=self.num_particles, sigma=kernel_sigma, adaptive_sig=adaptive_sig, device=device)
        
        # identity
        self.identity = torch.eye(self.num_particles).to(self.device)
        self.identity_mat = torch.eye(self.act_dim).to(self.device)
        self.delta = 1e-4
        self.drv_delta = torch.zeros((self.act_dim, 1, self.act_dim)).to(self.device)
        for i in range(self.act_dim):
            self.drv_delta[i, :, i] = self.delta 
       
        

        self.epsilon_threshold = 0.9
        self.epsilon_decay = (0.9 - 0.0) / 400000
        self.beta = (200100 / (0 + 200))

    def svgd_optim(self, x, dx, dq): 
        dx = dx.view(x.size())
        #print(' ')
        # print()
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
            
                        

            # drv_obs = obs.unsqueeze(0).repeat(self.act_dim, 1, 1)
            # drv_X = X.unsqueeze(0).repeat(self.act_dim, 1, 1)
            # drv_Xp = drv_X + self.drv_delta
            # drv_Xn = drv_X - self.drv_delta
            # term_1 = torch.min(self.q1(drv_obs, drv_Xp), self.q2(drv_obs, drv_Xp))
            # term_2 = torch.min(self.q1(drv_obs, drv_Xn), self.q2(drv_obs, drv_Xn))
            # score_func = ((term_1 - term_2) / (2 * self.delta)).T

            X = X.reshape(-1, self.num_particles, self.act_dim)
            score_func = score_func.reshape(X.size())
            K_XX, K_diff, K_gamma, K_grad = self.Kernel(X, X)
            phi = (K_XX.matmul(score_func) + K_grad.sum(1)) / self.num_particles 

            # Calculation of the adaptive learning rate should be here. to discuss
            self.svgd_lr = self.svgd_lr_init
            
            
            if with_logprob:
                term1 = (K_grad * score_func.unsqueeze(1)).sum(-1).mean(2)
                term2 = -2 * K_gamma.squeeze(-1).squeeze(-1) * ((K_grad.permute(0,2,1,3) * K_diff).sum(-1) - self.act_dim * (K_XX - self.identity)).mean(1)
                self.term1_debug += term1.mean()
                self.term2_debug += term2.mean()
                logp = logp - self.svgd_lr * (term1 + term2) 
            
            return phi, log_prob, score_func 
        
        for t in range(self.num_svgd_steps):
            phi_, q_s_a, dq = phi(a)
            
            a = self.svgd_optim(a, phi_, dq)
        
        a = self.act_limit * torch.tanh(a) 
        
        return a, logp, q_s_a


    def act(self, obs, action_selection=None, with_logprob=True, loss_q_=None, itr=None):
        logp_a = None

        a0 = self.a0[torch.randint(len(self.a0), (len(obs),))]
        
        # run svgd
        a, logp_svgd, q_s_a = self.sampler(obs, a0.detach(), with_logprob) 

        # a, logp_svgd, q_s_a = self.sampler_debug(obs, a0.detach(), with_logprob) 
        q_s_a = q_s_a.view(-1, self.num_particles)

        # compute the entropy 
        if with_logprob:
            logp_normal = - self.act_dim * 0.5 * np.log(2 * np.pi * self.sigma_p0) - (0.5 / self.sigma_p0) * (a0**2).sum(-1).view(-1,self.num_particles)
            
            logp_tanh = - ( 2 * (np.log(2) - a - F.softplus(-2 * a))).sum(axis=-1).view(-1,self.num_particles)
            
            logp_a = (logp_normal + logp_svgd + logp_tanh).mean(-1)
            

        self.a =  a.view(-1, self.num_particles, self.act_dim)

        # at test time
        if action_selection is None:
            a = self.a
            # print('None')
        elif action_selection == 'softmax':
            # print('softmax')
            self.beta = 1
            soft_max_probs = torch.exp((q_s_a - q_s_a.max(dim=1, keepdim=True)[0])/self.beta)
            dist = Categorical(soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
            a = self.a[:,dist.sample()]

        elif action_selection == 'max':
            a = self.a[:,q_s_a.argmax(-1)]
            # print('max')
        
        elif action_selection == 'random':
            a = self.a[:,np.random.randint(self.num_particles),:]
            # print('random')
        
        elif action_selection == 'adaptive_softmax':
            # print('adaptive_softmax')
            if self.beta > 0.5:
                self.beta = (200100 / (itr + 200))
            else:
                self.beta = 0.5
            soft_max_probs = torch.exp((q_s_a - q_s_a.max(dim=1, keepdim=True)[0])/self.beta)
            # print('############# ', soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
            dist = Categorical(soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
            a = self.a[:,dist.sample()]
        elif action_selection == 'softmax_egreedy':
            # print('adaptive_softmax')
            self.epsilon_threshold -= self.epsilon_decay
            eps = np.random.random()
            if eps > self.epsilon_threshold:
                self.beta = 1
                soft_max_probs = torch.exp((q_s_a - q_s_a.max(dim=1, keepdim=True)[0])/self.beta)
                dist = Categorical(soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
                a = self.a[:,dist.sample()]
            else:
                a = self.a[:,np.random.randint(self.num_particles),:]


        return a.view(-1, self.act_dim), logp_a