import torch.nn as nn
from networks import mlp,MLPQFunction
from actors.actor_sac import ActorSac
from actors.actor_svgd import ActorSvgd
from actors.actor_sql import ActorSql
from actors.actor_diffusion import ActorDiffusion

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, device , hidden_sizes=(256,256), activation=nn.ELU, actor_kwargs=dict()):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        dict_actors = {
            'sac': ActorSac,
            'svgd_nonparam': ActorSvgdNonParam,
            'svgd_p0_pram': ActorSvgdP0Param,
            'svgd_p0_kernel_pram': ActorSvgdP0KernelParam,
            'svgd_sql': ActorSql,
            'diffusion': ActorDiffusion}

        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.pi = dict_actors[actor_kwargs.actor](actor_kwargs)

    def forward(self, obs):
    	self.pi.act(obs)




