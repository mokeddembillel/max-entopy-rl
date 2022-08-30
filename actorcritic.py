import torch.nn as nn
from networks import mlp, MLPQFunction
from actors.actor_sac import ActorSac
from actors.actor_svgd import ActorSvgdNonParam,ActorSvgdP0Param,ActorSvgdP0KernelParam
from actors.actor_sql import ActorSql
from actors.actor_diffusion import ActorDiffusion
from utils import AttrDict

class ActorCritic(nn.Module):
    def __init__(self, actor, observation_space, action_space, critic_kwargs=AttrDict(), actor_kwargs=AttrDict()):
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

        self.q1 = MLPQFunction(obs_dim, act_dim, critic_kwargs.hidden_sizes, critic_kwargs.activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, critic_kwargs.hidden_sizes, critic_kwargs.activation)
        self.pi = dict_actors[actor](obs_dim, act_dim, act_limit, **actor_kwargs)

    def forward(self, obs):
        
    	return self.pi.act(obs)




