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
import line_profiler

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

        self.delta = 30
        self.drv_delta = torch.zeros((self.act_dim, 1, self.act_dim)).to(self.device)
        for i in range(self.act_dim):
            self.drv_delta[i, :, i] = self.delta 
        self.drv_delta2 = (torch.eye(self.act_dim).to(self.device) * self.delta).unsqueeze(0)
        self.ones = torch.ones((2, 1, self.act_dim, 1)).to(self.device)

        self.drv_delta3 = (torch.eye(self.act_dim).to(self.device) * self.delta).unsqueeze(0).repeat((self.num_particles, 1, 1))
        self.ones3 = torch.ones((2, self.num_particles, self.act_dim, self.obs_dim)).to(self.device) 
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
        

        self.epsilon_threshold = 0.9
        self.epsilon_decay = (0.9 - 0.0) / 400000
        self.beta = (200100 / (0 + 200))

    def svgd_optim(self, x, dx, dq): 
        dx = dx.view(x.size())
        #print(' ')
        # print()
        x = x + self.svgd_lr * dx
        return x

    
    # @profile
    def sampler(self, obs, a, with_logprob=True):
        logp = 0
        self.term1_debug = 0
        self.term2_debug = 0
        self.x_t = [a.detach().cpu().numpy().tolist()]
        self.phis = []

        # @profile
        def phi(X):
            nonlocal logp
            X.requires_grad_(True)            
            log_prob1 = self.q1(obs, X)
            log_prob2 = self.q2(obs, X)
            log_prob = torch.min(log_prob1, log_prob2)

            # start = timeit.default_timer()
            ###### Method 0
            score_func = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0]
            # stop = timeit.default_timer()
            # print('Time deriv auto: ', stop - start) 
                        
            # # start = timeit.default_timer()
            # ###### Method 1
            # drv_obs = obs.unsqueeze(0).repeat(self.act_dim, 1, 1)
            # drv_X = X.unsqueeze(0).repeat(self.act_dim, 1, 1)
            # drv_Xp = drv_X + self.drv_delta
            # drv_Xn = drv_X - self.drv_delta
            # term_1 = torch.min(self.q1(drv_obs, drv_Xp), self.q2(drv_obs, drv_Xp))
            # term_2 = torch.min(self.q1(drv_obs, drv_Xn), self.q2(drv_obs, drv_Xn))
            # score_func2 = ((term_1 - term_2) / (2 * self.delta)).T 
            
            # # ###### Method 2
            # drv_obs = obs.unsqueeze(1).repeat(1, self.act_dim, 1)
            # drv_X = X.unsqueeze(1)
            # drv_Xp = drv_X + self.drv_delta2
            # drv_Xn = drv_X - self.drv_delta2
            # term_1 = torch.min(self.q1(drv_obs, drv_Xp), self.q2(drv_obs, drv_Xp))
            # term_2 = torch.min(self.q1(drv_obs, drv_Xn), self.q2(drv_obs, drv_Xn))
            # score_func3 = ((term_1 - term_2) / (2 * self.delta))
            


            
            # drv_obs = obs.unsqueeze(1).repeat(1, self.act_dim, 1)
            # drv_X = X.unsqueeze(1)
            
            # drv_obs = drv_obs.detach()
            # drv_X = drv_X.detach()

            # drv_obs = torch.zeros_like(drv_obs)
            # drv_X[0, :, :] = torch.tensor([-0.5, -0.5, -0.5])
            # drv_X[1, :, :] = torch.tensor([0.0, 0.0, 0.0])
            # drv_X[2, :, :] = torch.tensor([0.5, 0.5, 0.5])
            
            # drv_Xp = drv_X + self.drv_delta2
            # drv_Xn = drv_X - self.drv_delta2

            # drv_obs = drv_obs[0:3, :, :]
            # drv_Xp = drv_Xp[0:3, :, :]
            # drv_Xn = drv_Xn[0:3, :, :]



            # term_1 = torch.min(self.q1(drv_obs, drv_Xp), self.q2(drv_obs, drv_Xp))
            # term_2 = torch.min(self.q1(drv_obs, drv_Xn), self.q2(drv_obs, drv_Xn))
            # score_func4 = ((term_1 - term_2) / (2 * self.delta))
             



            # self.ones = torch.ones((2, 1, self.act_dim, 1)).to(self.device)
            # drv_obs2 = obs.unsqueeze(1).unsqueeze(0) * self.ones
            # drv_X2 = X.unsqueeze(1)

            # drv_obs2 = drv_obs2.detach()
            # drv_X2 = drv_X.detach()

            # drv_obs2 = torch.zeros_like(drv_obs2)
            # drv_X2[0, :, :] = torch.tensor([-0.5, -0.5, -0.5])
            # drv_X2[1, :, :] = torch.tensor([0.0, 0.0, 0.0])
            # drv_X2[2, :, :] = torch.tensor([0.5, 0.5, 0.5])
            
            # drv_Xp2 = drv_X2 + self.drv_delta2
            # drv_Xn2 = drv_X2 - self.drv_delta2
            # drv = torch.concat((drv_Xp2.unsqueeze(0), drv_Xn2.unsqueeze(0)), dim=0)

            # drv_obs2 = drv_obs2[:, 0:3, :, :]
            # drv = drv[:, 0:3, :, :]

            # qv_1 = self.q1(drv_obs2, drv)
            # qv_2 = self.q2(drv_obs2, drv)
            # term_1 = torch.min(qv_1[0], qv_2[0])
            # term_2 = torch.min(qv_1[1], qv_2[1])
            # score_func5 = ((term_1 - term_2) / (2 * self.delta))
            
            ###### Method 3
            # self.ones = torch.ones((2, 1, self.act_dim, 1)).to(self.device)
            # drv_obs2 = obs.unsqueeze(1).unsqueeze(0) * self.ones
            # drv_X2 = X.unsqueeze(1)
            # drv_Xp2 = drv_X2 + self.drv_delta2
            # drv_Xn2 = drv_X2 - self.drv_delta2
            # drv = torch.concat((drv_Xp2.unsqueeze(0), drv_Xn2.unsqueeze(0)), dim=0)
            # qv_1 = self.q1(drv_obs2, drv)
            # qv_2 = self.q2(drv_obs2, drv)
            # term_1 = torch.min(qv_1[0], qv_2[0])
            # term_2 = torch.min(qv_1[1], qv_2[1])
            # score_func4 = ((term_1 - term_2) / (2 * self.delta))
            
            # self.drv_delta2 = (torch.eye(self.act_dim).to(self.device) * self.delta).unsqueeze(0)
            # self.ones = torch.ones((2, 1, self.act_dim, 1)).to(self.device)
            # drv_obs2 = obs.unsqueeze(1).unsqueeze(0) * self.ones
            # drv_X2 = X.unsqueeze(1)
            # drv_Xp2 = drv_X2 + self.drv_delta2
            # drv_Xn2 = drv_X2 - self.drv_delta2
            # drv = torch.stack((drv_Xp2, drv_Xn2))
            # drv_tmp2 = drv.reshape((-1, self.act_dim))
            # drv_obs2_tmp2 = drv_obs2.reshape((-1, self.obs_dim))
            # # drv_obs2_tmp2.shape
            # # (drv == drv_tmp2.reshape((2, self.num_particles, self.act_dim, self.act_dim))).all()
            # # (drv_obs2 == drv_obs2_tmp2.reshape((2, self.num_particles, self.act_dim, self.obs_dim))).all()
            # qv_1.reshape((2, self.num_particles, self.act_dim))
            # qv_1 = self.q1(drv_obs2_tmp2, drv_tmp2)
            # qv_2 = self.q2(drv_obs2_tmp2, drv_tmp2)
            # term_1 = torch.min(qv_1[0], qv_2[0])
            # term_2 = torch.min(qv_1[1], qv_2[1])
            # score_func6 = ((term_1 - term_2) / (2 * self.delta))
            

            # ###### Method 4
            # self.ones3 *= obs.unsqueeze(1)
            # drv_X = X.unsqueeze(1)
            # self.drv_delta3 += drv_X
            # term_1 = torch.min(self.q1(self.ones3, self.drv_delta3), self.q2(self.ones3, self.drv_delta3))
            # self.drv_delta3 = 2 * drv_X - self.drv_delta3

            # term_2 = torch.min(self.q1(self.ones3, self.drv_delta3), self.q2(self.ones3, self.drv_delta3))
            # score_func5 = ((term_1 - term_2) / (2 * self.delta))
            # self.drv_delta3 = drv_X - self.drv_delta3
            # self.ones3[:,:,:] = 1

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

    # @profile
    def act(self, obs, action_selection=None, with_logprob=True, loss_q_=None, itr=None):
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
        if action_selection is None:
            a = self.a
            # print('None')
        elif action_selection == 'random':
            a = self.a[:,np.random.randint(self.num_particles),:]
            # print('random')
        elif action_selection == 'max':
            a = self.a[:,q_s_a.argmax(-1)]
            # print('max')
        elif action_selection == 'softmax':
            # print('softmax')

            soft_max_probs = torch.exp((q_s_a - q_s_a.max(dim=1, keepdim=True)[0]))
            dist = Categorical(soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
            a = self.a[:,dist.sample()]
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
        # elif action_selection == 'softmax_egreedy':
        #     # print('adaptive_softmax')
        #     self.epsilon_threshold -= self.epsilon_decay
        #     eps = np.random.random()
        #     if eps > self.epsilon_threshold:
        #         self.beta = 1
        #         soft_max_probs = torch.exp((q_s_a - q_s_a.max(dim=1, keepdim=True)[0])/self.beta)
        #         dist = Categorical(soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
        #         a = self.a[:,dist.sample()]
        #     else:
        #         a = self.a[:,np.random.randint(self.num_particles),:]

        ########## Debugging. to be removed
        # a0 = torch.clamp(a0, -self.act_limit, self.act_limit).detach()
        # return a.view(-1, self.act_dim), logp_normal
        return a.view(-1, self.act_dim), logp_a