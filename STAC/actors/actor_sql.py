import torch
import torch.nn as nn
from networks import mlp
from torch.distributions import Categorical
from actors.kernels import RBF


class ActorSql(nn.Module):
    def __init__(self, actor, obs_dim, act_dim, act_limit, num_svgd_particles, svgd_lr, test_deterministic, batch_size, device, hidden_sizes, q1, q2, activation=None):
        super(ActorSql, self).__init__()
        self.actor = actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.test_deterministic = test_deterministic
        self.num_particles = num_svgd_particles
        self.batch_size = batch_size
        self.device = device

        self.q1 = q1
        self.q2 = q2

        self.amortized_svgd_net_l1 = mlp([self.obs_dim + self.act_dim] + list(hidden_sizes), activation)
        self.amortized_svgd_net_l2 =  mlp(list(hidden_sizes) + [self.act_dim], nn.Tanh, nn.Tanh)
        
        self.kernel = RBF()

        self.a_0 = torch.rand((5*self.batch_size, self.num_particles, self.act_dim)).view(-1,self.act_dim).to(self.device)
    

    def amortized_svgd_net(self,obs, a):
        out = self.amortized_svgd_net_l1(torch.cat([obs, a],dim=-1))
        out = self.amortized_svgd_net_l2(out)
        return out
    
    
    def act(self, obs, deterministic=None, with_logprob=None, in_q_loss=None):   
        a_0 = self.a_0[torch.randint(len(self.a_0), (len(obs),))]

        if in_q_loss:
            return a_0, None
        
        a = self.amortized_svgd_net(obs, a_0)

        q1_values = self.q1(obs, a)
        q2_values = self.q2(obs, a)
        q_values = torch.min(q1_values, q2_values)

        self.a = a.view(-1, self.num_particles, self.act_dim) # (-1, np, ad)
        q_values = q_values.view(-1, self.num_particles) # (-1, np)

        if (with_logprob == False) and (deterministic == True):
            a = self.a[:,q_values.view(-1, self.num_particles).argmax(-1)]
         
        elif (with_logprob == False) and (deterministic == False):
            beta = 1
            soft_max_porbs = torch.exp(beta * q_values - torch.max(q_values, dim=1, keepdim=True)[0]) # (-1, np)
            dist = Categorical((soft_max_porbs / torch.sum(soft_max_porbs, dim=1, keepdim=True)))
            a = self.a[:,dist.sample()]
        else:
            a = self.a
        return a.view(-1, self.act_dim), None
