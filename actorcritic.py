import torch.nn as nn


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
	def __init__(self):
		super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, device , hidden_sizes=(256,256), activation=nn.ELU, actor_kwargs=dict()):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.sac_version = sac_version
        self.device = device
        # build policy and value functions
        if (self.sac_version != 'svgd_v1'):
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, num_svgd_particles)#, wandb=wandb)

        self.num_svgd_steps = num_svgd_steps
        self.act_limit = act_limit 
        self.num_svgd_particles = num_svgd_particles
        self.act_dim = act_dim
        self.svgd_lr = svgd_lr

        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        
        if agent == "SAC":
        	self.actor = ACTOR_SAC()
        elif agent == "STAC_SVGD":
        	self.actor = STAC_SVGD()

    def forward():
    	self.actor.act()
    



