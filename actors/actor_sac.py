from networks import MLPSquashedGaussian

class SAC_ACTOR():
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, num_particles, log_std_min, log_std_max):
        self.net = MLPSquashedGaussian(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, num_particles, log_std_min, log_std_max)
    
    def forward(self):
        return self.net.forward

    def act(self, obs, deterministic=False, with_logprob=True):
        pi_action, logp_pi = self.net.forward(obs, deterministic, with_logprob)
        return pi_action, logp_pi


