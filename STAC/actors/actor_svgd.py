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
    device, hidden_sizes, q1, q2, activation=torch.nn.ReLU, kernel_sigma=None, adaptive_lr=None, alpha=1):
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
        self.alpha = alpha

        #optimizer parameters
        self.adaptive_lr = adaptive_lr
        self.beta = 0.999
        self.v_dx = 0

        if actor == "svgd_nonparam":
            self.a0 = torch.normal(0, self.sigma_p0, size=(5 * batch_size * num_svgd_particles, self.act_dim)).to(self.device)
        elif actor == 'svgd_p0_pram':
            self.p0 = MLPSquashedGaussian(obs_dim, act_dim, hidden_sizes, activation)

        if actor == "svgd_p0_kernel_pram":
            self.Kernel = RBF(True, act_dim, hidden_sizes, sigma=kernel_sigma, device=device)
        else:
            self.Kernel = RBF(num_particles=self.num_particles, sigma=kernel_sigma, adaptive_sig=adaptive_sig, device=device)
        
        # identity
        self.identity = torch.eye(self.num_particles).to(self.device)
        self.identity_mat = torch.eye(self.act_dim).to(self.device)

        self.delta = 1e-3

        self.drv_delta2 = (torch.eye(self.act_dim).to(self.device) * self.delta).unsqueeze(0)
        self.drv_delta2_ = torch.concat((self.drv_delta2.unsqueeze(0), -self.drv_delta2.unsqueeze(0)), dim=0)
        self.ones = torch.ones((2, 1, self.act_dim, 1)).to(self.device)


        self.epsilon_threshold = 0.9
        self.epsilon_decay = (0.9 - 0.0) / 400000
        self.beta = (200100 / (0 + 200))
        self.steps_debug = 0


    def svgd_optim(self, x, dx, dq): 
        dx = dx.view(x.size())
        #print(' ')
        # print()
        x = x + self.svgd_lr * dx
        # x = x + self.svgd_lr / self.alpha * dx
        return x
    
    
    # @profile
    def sampler(self, obs, a, with_logprob=True):
        logp = 0
        # self.term1_debug = 0
        # self.term2_debug = 0
        self.x_t = [a.detach().cpu().numpy().tolist()]
        self.phis = []
        self.score_funcs = []
        self.kernel_sigmas = []
        q_s_a = None

        # @profile
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
                term1 = (K_grad * score_func.unsqueeze(1)).sum(-1).sum(2)/(self.num_particles-1)
                term2 = -2 * K_gamma.squeeze(-1).squeeze(-1) * ((K_grad.permute(0,2,1,3) * K_diff).sum(-1) - self.act_dim * (K_XX - self.identity)).sum(1) / (self.num_particles-1)
                # self.term1_debug += term1.mean()
                # self.term2_debug += term2.mean()
                logp = logp - self.svgd_lr * (term1 + term2) 
                # logp = logp - self.svgd_lr / self.alpha * (term1 + term2) 
            
            return phi, log_prob, score_func 
        

        # if self.num_svgd_steps =='while':
        #     self.steps_debug = 0
        #     prev_grad = 100000
        #     while True:
        #         phi_, q_s_a, dq = phi(a)
        #         # print('PHI ################## ', phi_.detach().cpu().numpy())
        #         # import pdb; pdb.set_trace()
        #         a = self.svgd_optim(a, phi_, dq)
        #         # Collect Data for debugging
        #         self.x_t.append((self.act_limit * torch.tanh(a)).detach().cpu().numpy().tolist())
        #         self.phis.append((self.svgd_lr * phi_.detach().cpu().numpy()).tolist())
        #         self.score_funcs.append(torch.norm(dq.detach(), dim=-1).mean().cpu().numpy().tolist())
        #         dq_norm = torch.norm(dq.detach(), dim=-1).mean().cpu().item()
                
        #         self.steps_debug += 1
        #         # print('Number of steps taken :, ', self.steps_debug, torch.norm(dq.detach(), dim=-1).mean().cpu().item())
        #         if abs(prev_grad - dq_norm) < 0.005 or self.steps_debug == 500:
        #             break
        #         prev_grad = dq_norm


        #     # if (a > self.act_limit).any():
        #     #     break
        #     #a = torch.clamp(a, -self.act_limit, self.act_limit).detach()
        #     #print("t: ", t, " ", a[0])
        # else:
        for t in range(self.num_svgd_steps):
            phi_, q_s_a, dq = phi(a)
            # print('PHI ################## ', phi_.detach().cpu().numpy())
            # import pdb; pdb.set_trace()
            a = self.svgd_optim(a, phi_, dq)

            self.kernel_sigmas.append(self.Kernel.sigma_debug)
            # Collect Data for debugging
            # self.x_t.append((self.act_limit * torch.tanh(a)).detach().cpu().numpy().tolist())
            # self.phis.append((self.svgd_lr * phi_.detach().cpu().numpy()).tolist())
            # self.score_funcs.append(torch.norm(dq.detach(), dim=-1).mean().cpu().numpy().tolist())

            # if (a > self.act_limit).any():
            #     break
            #a = torch.clamp(a, -self.act_limit, self.act_limit).detach()
            #print("t: ", t, " ", a[0])
        
        # a = self.act_limit * torch.tanh(a) 
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
        elif self.actor == 'svgd_p0_pram':
            ############################################################
            ns = int(500/self.num_particles)
            obs_tmp = obs.view(-1, self.num_particles, self.obs_dim).repeat(1, ns, 1)
            self.mu, self.sigma = self.p0(obs_tmp)
            ############################################################
            
            # self.mu, self.sigma = self.p0(obs)
            self.init_dist_normal = Normal(self.mu, self.sigma)
            a0 = self.init_dist_normal.rsample()


            ############################################################
            indicies = torch.logical_and(a0 > -3 * self.sigma, a0 < 3 * self.sigma).all(-1)
            
            if (indicies.type(torch.float32).sum(-1) > self.num_particles).all():
                new_a0 = []
                for i in range(a0.shape[0]):
                    new_a0.append(a0[i][indicies[i]][:self.num_particles])
                new_a0 = torch.stack(new_a0)
                a0 = new_a0.view(-1,self.act_dim)
                self.mu, self.sigma = self.mu[:, :self.num_particles, :].reshape(-1, self.act_dim), self.sigma[:, :self.num_particles, :].reshape(-1, self.act_dim)
                
                self.init_dist_normal = Normal(self.mu, self.sigma)
            else:
                raise Exception("Number of sampled particles not enough")
            ############################################################
            

            # ns = 100
            # c1 = torch.tensor([0.3, 0.5, 0.6]).view(1, 1, -1).repeat(3, ns, 1)

            # t1 = torch.rand((3, ns, 3))
            
            # if ((t1 > c1).all(-1).type(torch.float32).sum(-1) > self.num_particles).all():
            #     (t1 > c1).all(-1)
            #     new_tensor = []
            #     # for i in range(self.batch_size):
            #     for i in range(3):
            #         new_tensor.append(t1[i][(t1 > c1).all(-1)[i]][:self.num_particles])
            #         # print(t1[i][(t1 > c1).all(-1)[i]])
            #     new_a0 = torch.stack(new_tensor)
            #     t1[(t1 > c1).all(-1)].shape


            # print('############################',  self.mu[0], self.sigma[0])
            # a0_tanh = self.act_limit * torch.tanh(a0)

        self.a0_debbug = a0.view(-1, self.num_particles, self.act_dim)

        # run svgd
        # a, logp_svgd, q_s_a = self.sampler(obs, a0.detach(), with_logprob) 
        a, logp_svgd, q_s_a = self.sampler(obs, a0, with_logprob) 

        # a, logp_svgd, q_s_a = self.sampler_debug(obs, a0.detach(), with_logprob) 

        # compute the entropy 
        if with_logprob:
            if self.actor == "svgd_nonparam":
                logp_normal = - self.act_dim * 0.5 * np.log(2 * np.pi * self.sigma_p0) - (0.5 / self.sigma_p0) * (a0**2).sum(-1).view(-1,self.num_particles)
            
            elif self.actor == 'svgd_p0_pram':
                # logp_normal = - (self.act_dim * 0.5 * torch.log(2 * torch.pi * self.sigma.view(-1,self.num_particles, self.act_dim)) - (0.5 / self.sigma.view(-1,self.num_particles, self.act_dim)) * ((a0 - self.mu)**2).view(-1,self.num_particles, self.act_dim)).sum(-1)
                logp_normal = self.init_dist_normal.log_prob(a0).sum(axis=-1).view(-1,self.num_particles)
            # if logp_normal.mean() > 10:
            #     pass
            # else:
            #     pass

            # logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)


            # logp_tanh_1 = - ( 2 * (np.log(2) - a0_tanh - F.softplus(-2 * a0_tanh))).sum(axis=-1).view(-1,self.num_particles)

            logp_tanh_2 = - ( 2 * (np.log(2) - a - F.softplus(-2 * a))).sum(axis=-1).view(-1,self.num_particles)
            # if logp_tanh_2.mean() > 50:
            #     print(self.mu.min().cpu().detach().item(), self.mu.mean().cpu().detach().item(), self.mu.max().cpu().detach().item())
            #     print(self.sigma.min().cpu().detach().item(), self.sigma.mean().cpu().detach().item(), self.sigma.max().cpu().detach().item())
            #     print('#########################')
            #     print('#########################')
            #     import pdb; pdb.set_trace()
            # try:
                # print('######## ', logp_normal.shape)
                # print('######## ', logp_svgd.shape)
                # print('######## ', logp_tanh.shape)
                # print()
                # print()
            # logp_a = (logp_normal + logp_svgd + logp_tanh_1 + logp_tanh_2).mean(-1)
            logp_a = (logp_normal + logp_svgd + logp_tanh_2).mean(-1)
            # logp_wrong_a = (logp_normal + logp_wrong + logp_tanh).mean(-1)
            # if torch.absolute(logp_a).mean().detach().cpu().item() > 500:
            #     import pdb; pdb.set_trace()
                

            self.logp_normal_debug = logp_normal.mean()
            try:
                self.logp_svgd_debug = logp_svgd.mean()
            except:
                self.logp_svgd_debug = torch.tensor(0)
            self.logp_tanh_debug = logp_tanh_2.mean()
            # except:
            #     import pdb; pdb.set_trace()
        
            
        a = self.act_limit * torch.tanh(a) 
        
        self.a =  a.view(-1, self.num_particles, self.act_dim)

        # at test time
        if action_selection is None:
            a = self.a
            # print('None')
            # print('random')
        elif action_selection == 'max':
            # print('max')
            if self.num_svgd_steps == 0:
                q_s_a1 = self.q1(obs, a.view(-1, self.act_dim))
                q_s_a2 = self.q2(obs, a.view(-1, self.act_dim))
                q_s_a = torch.min(q_s_a1, q_s_a2)
            #     self.q_s_a_max = q_s_a.max()
            #     self.q_s_a_all = q_s_a

                a_ = self.mu.view(-1, self.num_particles, self.act_dim)[:, 0, :]
                # q_s_a1_ = self.q1(obs[0].unsqueeze(0), a_.view(-1, self.act_dim))
                # q_s_a2_ = self.q2(obs[0].unsqueeze(0), a_.view(-1, self.act_dim))
                # self.q_s_a_max_orig = torch.min(q_s_a1_, q_s_a2_)
                
            q_s_a = q_s_a.view(-1, self.num_particles)
            # a = a_
            # a = self.mu.view(-1, self.num_particles, self.act_dim)[:, 0, :]
            a = self.a[:,q_s_a.argmax(-1)]


        elif action_selection == 'softmax':
            # print('softmax')
            if self.num_svgd_steps == 0:
                q_s_a1 = self.q1(obs, a.view(-1, self.act_dim))
                q_s_a2 = self.q2(obs, a.view(-1, self.act_dim))
                q_s_a = torch.min(q_s_a1, q_s_a2)

            q_s_a = q_s_a.view(-1, self.num_particles)

            soft_max_probs = torch.exp((q_s_a - q_s_a.max(dim=1, keepdim=True)[0]))
            dist = Categorical(soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
            a = self.a[:,dist.sample()]

        elif action_selection == 'random':
            a = self.a[:,np.random.randint(self.num_particles),:]
        # elif action_selection == 'adaptive_softmax':
        #     # print('adaptive_softmax')
        #     q_s_a = q_s_a.view(-1, self.num_particles)
        #     if self.beta > 0.5:
        #         self.beta = (200100 / (itr + 200))
        #     else:
        #         self.beta = 0.5
        #     soft_max_probs = torch.exp((q_s_a - q_s_a.max(dim=1, keepdim=True)[0])/self.beta)
        #     # print('############# ', soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
        #     dist = Categorical(soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True))
        #     a = self.a[:,dist.sample()]
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