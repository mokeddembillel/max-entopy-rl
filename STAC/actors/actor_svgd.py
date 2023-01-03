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

class ActorSvgd(torch.nn.Module):
    def __init__(self, actor, obs_dim, act_dim, act_limit, num_svgd_particles, svgd_sigma_p0, num_svgd_steps, svgd_lr, test_deterministic, batch_size, adaptive_sig,
    device, hidden_sizes, q1, q2, activation=torch.nn.ReLU, kernel_sigma=None, adaptive_lr=None, action_selection=None):
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
        self.action_selection = action_selection
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
            self.Kernel = RBF(True, act_dim, hidden_sizes, sigma=kernel_sigma, device=device)
        else:
            self.Kernel = RBF(num_particles=self.num_particles, sigma=kernel_sigma, adaptive_sig=adaptive_sig, device=device)
        
        # identity
        self.identity = torch.eye(self.num_particles).to(self.device)
        self.identity_mat = torch.eye(self.act_dim).to(self.device)
        self.delta=1e-3
        self.drv_delta = torch.zeros((self.act_dim, 1, self.act_dim)).to(self.device)
        for i in range(self.act_dim):
            self.drv_delta[i, :, i] = self.delta 
        # Debugging #########################################
        # gmm = 1
        # if (gmm == 1):
        #     self.init_dist_mu = 4 
        #     self.init_dist_sigma = 0.2 #6
        #     self.target_dist_sigma = 1.0
        #     self.P = torch.distributions.MultivariateNormal(torch.Tensor([0.0,0.0]).to(device),covariance_matrix= self.target_dist_sigma * torch.Tensor([[1.0,0.0],[0.0,1.0]]).to(device))
        # else:
        #     self.init_dist_mu = 0
        #     self.init_dist_sigma = 0.2 #6
        #     self.P = GMMDist(dim=2, n_gmm=gmm, device=self.device)
        




    def svgd_optim(self, x, dx, dq): 
        dx = dx.view(x.size())
        #print(' ')
        # print()
        x = x + self.svgd_lr * dx
        return x

    
    
    def sampler(self, obs, a, with_logprob=True):
        logp = 0
        self.term1_debug = 0
        self.term2_debug = 0
        self.x_t = [a.detach().cpu().numpy().tolist()]
        self.phis = []


        def phi(X):
            nonlocal logp
            X.requires_grad_(True)            
            log_prob1 = self.q1(obs, X)
            log_prob2 = self.q2(obs, X)
            log_prob = torch.min(log_prob1, log_prob2)

            # start = timeit.default_timer()
            score_func = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0]
            # stop = timeit.default_timer()
            # print('Time deriv auto: ', stop - start) 
                        

            # start = timeit.default_timer()
            # drv_obs = obs.unsqueeze(0).repeat(self.act_dim, 1, 1)
            # drv_X = X.unsqueeze(0).repeat(self.act_dim, 1, 1)
            # drv_Xp = drv_X + self.drv_delta
            # drv_Xn = drv_X - self.drv_delta
            # term_1 = torch.min(self.q1(drv_obs, drv_Xp), self.q2(drv_obs, drv_Xp))
            # term_2 = torch.min(self.q1(drv_obs, drv_Xn), self.q2(drv_obs, drv_Xn))
            # score_func = ((term_1 - term_2) / (2 * self.delta)).T
            # stop = timeit.default_timer()
            # print('Time deriv numeric: ', stop - start) 
            # print()

            X = X.reshape(-1, self.num_particles, self.act_dim)
            score_func = score_func.reshape(X.size())
            K_XX, K_diff, K_gamma, K_grad = self.Kernel(X, X)
            phi = (K_XX.matmul(score_func) + K_grad.sum(1)) / self.num_particles 

            # Calculation of the adaptive learning rate should be here. to discuss
            self.svgd_lr = self.svgd_lr_init
            # print('adaptive' , self.adaptive_lr)
            if self.adaptive_lr: 
                # print('Condition: ', (self.svgd_lr * torch.sqrt( (score_func**2).sum(-1)).mean()).detach().item())
                if (self.svgd_lr * torch.sqrt( (score_func**2).sum(-1)).mean() ) > 1.0:
                    self.svgd_lr = 0.1 * (1/torch.sqrt( (score_func**2).sum(-1))).mean().detach().item()
            # print('SVGD_LR', self.svgd_lr)
            
            # compute the entropy
            if with_logprob:
                term1 = (K_grad * score_func.unsqueeze(1)).sum(-1).mean(2)
                term2 = -2 * K_gamma.squeeze(-1).squeeze(-1) * ((K_grad.permute(0,2,1,3) * K_diff).sum(-1) - self.act_dim * (K_XX - self.identity)).mean(1)
                self.term1_debug += term1.mean()
                self.term2_debug += term2.mean()
                logp = logp - self.svgd_lr * (term1 + term2) 
            
            return phi, log_prob, score_func 
        
        for t in range(self.num_svgd_steps):
            phi_, q_s_a, dq = phi(a)
            # print('PHI ################## ', phi_.detach().cpu().numpy())
            # import pdb; pdb.set_trace()
            a = self.svgd_optim(a, phi_, dq)
            # Collect Data for debugging
            self.x_t.append((self.act_limit * torch.tanh(a)).detach().cpu().numpy().tolist())
            self.phis.append((self.svgd_lr * phi_.detach().cpu().numpy()).tolist())

            # if (a > self.act_limit).any():
            #     break
            #a = torch.clamp(a, -self.act_limit, self.act_limit).detach()
            #print("t: ", t, " ", a[0])
        
        a = self.act_limit * torch.tanh(a) 
        #print("___________________")
        # import pdb; pdb.set_trace()
        # if  (a < -1).any() or (a> 1).any():
        #     print('#################### Error ####################')
        return a, logp, q_s_a


    def act(self, obs, deterministic=False, with_logprob=True, loss_q_=None, all_particles=None):
        logp_a = None
        # logp_normal = None

        if self.actor == "svgd_nonparam":
            a0 = self.a0[torch.randint(len(self.a0), (len(obs),))]
            # a0 = torch.clamp(a0, -self.act_limit, self.act_limit).detach()
        else:
            self.mu, self.sigma = self.p0(obs)
            a0 = Normal(self.mu, self.sigma).rsample()
            a0 = self.act_limit * torch.tanh(a0)

        self.a0_debbug = a0.view(-1, self.num_particles, self.act_dim)

        # run svgd
        a, logp_svgd, q_s_a = self.sampler(obs, a0.detach(), with_logprob) 

        # a, logp_svgd, q_s_a = self.sampler_debug(obs, a0.detach(), with_logprob) 
        q_s_a = q_s_a.view(-1, self.num_particles)

        # compute the entropy 
        if with_logprob:
            logp_normal = - self.act_dim * 0.5 * np.log(2 * np.pi * self.sigma_p0) - (0.5 / self.sigma_p0) * (a0**2).sum(-1).view(-1,self.num_particles)
            
            logp_tanh = - ( 2 * (np.log(2) - a - F.softplus(-2 * a))).sum(axis=-1).view(-1,self.num_particles)
            # try:
                # print('######## ', logp_normal.shape)
                # print('######## ', logp_svgd.shape)
                # print('######## ', logp_tanh.shape)
                # print()
                # print()
            logp_a = (logp_normal + logp_svgd + logp_tanh).mean(-1)
            # logp_wrong_a = (logp_normal + logp_wrong + logp_tanh).mean(-1)
            

            self.logp_normal_debug = logp_normal.mean()
            self.logp_svgd_debug = logp_svgd.mean()
            self.logp_tanh_debug = logp_tanh.mean()
            # except:
            #     import pdb; pdb.set_trace()
        self.a =  a.view(-1, self.num_particles, self.act_dim)

        # at test time
        if (not all_particles) and (deterministic == True):
            if self.action_selection == 1:
                a = self.a[:,q_s_a.argmax(-1)]
                # print('maaaax')
            elif self.action_selection == 2:
                # print('softmaax')
                beta = 1
                soft_max_probs = torch.exp(beta * q_s_a - q_s_a.max(dim=1, keepdim=True)[0])
                dist = Categorical(soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
                a = self.a[:,dist.sample()]
            else:
                print('Wroong parameter !!!!!!!!!!!!!!!')
        elif (not all_particles) and (deterministic == False):

            a = self.a.view(-1, self.num_particles, self.act_dim)[:,np.random.randint(self.num_particles),:]
            
        else:
            a = self.a

        ########## Debugging. to be removed
        # a0 = torch.clamp(a0, -self.act_limit, self.act_limit).detach()
        # return a.view(-1, self.act_dim), logp_normal
        return a.view(-1, self.act_dim), logp_a