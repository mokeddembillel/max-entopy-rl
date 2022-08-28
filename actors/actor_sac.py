from networks import MLPSquashedGaussian

class SAC_ACTOR():
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, log_std_min, log_std_max):
        self.num_particles = 1
        self.net = MLPSquashedGaussian(self, obs_dim, act_dim, hidden_sizes, activation)
    
    def forward(self):
        return self.net.forward

    def act(self, obs, deterministic=False, with_logprob=True):
        pi_action, logp_pi = self.net.forward(obs, deterministic, with_logprob)
        return pi_action, logp_pi

def act(self, obs, deterministic, with_logprob):#, wandb=None):
        if num_particles is None:
            num_particles = self.num_particles
        
        mu, sigma = self.forward(obs)

        pi_distribution = Normal(self.mu, self.std)
        
        if deterministic:
            pi_action = self.mu
        else:
            pi_action = pi_distribution.rsample()
        
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
            
            if num_particles>0:
                logp_pi = logp_pi.view(-1,num_particles).mean(-1)
        else:
            logp_pi = None
        
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action.reshape(-1,pi_action.size()[-1]), logp_pi


