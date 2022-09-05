import torch
import torch.nn as nn
from networks import mlp
from torch.distributions import Categorical
from actors.kernels import RBF


class ActorSql(nn.Module):
    def __init__(self, actor, obs_dim, act_dim, act_limit, num_svgd_particles, svgd_lr, test_deterministic, batch_size, device, hidden_sizes, q1, q2, activation=nn.ReLU):
        super(ActorSql, self).__init__()
        self.actor = actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.test_deterministic = test_deterministic
        self.num_particles = num_svgd_particles
        self.batch_size = batch_size
        
        self.q1 = q1
        self.q2 = q2
        self.concat = mlp([self.obs_dim + self.act_dim] + list(hidden_sizes),activation)
        self.layer2 =  mlp(list(hidden_sizes) + [self.act_dim], nn.Tanh, nn.Tanh)
        
        self.kernel = RBF()

        self.device = device
        self.to(self.device)
    
    def forward(self,state, action):
        samples = self.concat(torch.cat([state, action],dim=-1))
        samples = self.layer2(samples)
        return samples
    def act(self, state,  deterministic=None, with_logprob=None):
        # .view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1)
        if with_logprob:
            action_0 = torch.rand((self.batch_size, self.num_particles, self.act_dim)).view(-1,self.act_dim).to(self.device)
        else:
            action_0 = torch.rand((1, self.num_particles, self.act_dim)).view(-1,self.act_dim).to(self.device)

        action = self.forward(state, action_0)

        q1_values = self.q1(state, action)
        q2_values = self.q2(state, action)
        q_values = torch.min(q1_values, q2_values)

        action = action.view(-1, self.num_particles, self.act_dim) # (-1, np, ad)
        q_values = q_values.view(-1, self.num_particles) # (-1, np)

        if (with_logprob == False) and (deterministic == True):
            action = action[:,torch.max(q_values, dim=1)[1]]
         
        elif (with_logprob == False) and (deterministic == False):
            beta = 1
            nominator = torch.exp(beta * q_values - torch.max(q_values, dim=1, keepdim=True)[0]) # (-1, np)
            dist = Categorical((nominator / torch.sum(nominator, dim=1, keepdim=True)))
            action = action[:,dist.sample()].view(-1, self.act_dim)
        else:
            action = action.view(-1, self.act_dim)
        return action, None
