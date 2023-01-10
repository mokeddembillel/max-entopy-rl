import torch
import torch.nn as nn
from networks import mlp
from torch.distributions import Categorical
from actors.kernels import RBF
from networks import MLPFunction
import numpy as np


class ActorSql(nn.Module):
    def __init__(self, actor, obs_dim, act_dim, act_limit, num_svgd_particles, svgd_lr, test_action_selection, batch_size, device, hidden_sizes, q1, q2, activation=None):
        super(ActorSql, self).__init__()
        self.actor = actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.test_action_selection = test_action_selection
        self.num_particles = num_svgd_particles
        self.batch_size = batch_size
        self.device = device

        self.q1 = q1
        self.q2 = q2

        self.svgd_net = MLPFunction(self.obs_dim, self.act_dim, self.act_dim, hidden_sizes, activation)
        # self.concat = mlp([self.obs_dim + self.act_dim] + list(hidden_sizes),nn.ReLU, nn.ReLU)
        # self.layer2 =  mlp(list(hidden_sizes) + [self.act_dim], nn.Tanh, nn.Tanh)
        


        self.kernel = RBF()

        # self.a_0 = torch.rand((5*self.batch_size, self.num_particles, self.act_dim)).view(-1,self.act_dim).to(self.device)
    

    def amortized_svgd_net(self,obs, a):
        # samples = self.concat(torch.cat([obs, a],dim=-1))
        # samples = self.layer2(samples)
        # return samples
        out = self.svgd_net(obs, a)
        out = torch.tanh(out) * self.act_limit
        return out
    
    
    def act(self, obs, action_selection=None, with_logprob=None, in_q_loss=None):   
        # a_0 = self.a_0[torch.randint(len(self.a_0), (len(obs),))]
        a_0 = torch.rand((len(obs), self.act_dim)).view(-1,self.act_dim).to(self.device)

        if in_q_loss:
            return a_0, None
        
        a = self.amortized_svgd_net(obs, a_0)
        self.a = a.view(-1, self.num_particles, self.act_dim)

        if action_selection is not None:
            if action_selection == 'random':
                a = self.a.view(-1, self.num_particles, self.act_dim)[:,np.random.randint(self.num_particles),:]
            else:
                q1_values = self.q1(obs, a)
                q2_values = self.q2(obs, a)
                q_values = torch.min(q1_values, q2_values)
                q_values = q_values.view(-1, self.num_particles) # (-1, np)
                if action_selection == 'max':
                    a = self.a[:,q_values.view(-1, self.num_particles).argmax(-1)]
                elif action_selection == 'softmax':
                    beta = 1
                    soft_max_probs = torch.exp(beta * q_values - torch.max(q_values, dim=1, keepdim=True)[0]) # (-1, np)
                    dist = Categorical((soft_max_probs / torch.sum(soft_max_probs, dim=1, keepdim=True)))
                    a = self.a[:,dist.sample()]

        
        return a.view(-1, self.act_dim), None
