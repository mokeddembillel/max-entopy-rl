from copy import deepcopy
import itertools
import numpy as np
import torch 
from torch.optim import Adam
from datetime import datetime
from utils import count_vars
from spinup_utils.logx import EpochLogger
# from torch.utils.tensorboard import SummaryWriter
# from envs.multigoal_env import MultiGoalEnv, QFPolicyPlotter
from actorcritic import ActorCritic
from utils import combined_shape, count_vars, AttrDict
from buffer import ReplayBuffer

class MaxEntrRL():
    def __init__(self, env_fn, tb_logger, env, actor, seed, critic_kwargs=AttrDict(), actor_kwargs=AttrDict(), device="cuda",   
        RL_kwargs=AttrDict(), optim_kwargs=AttrDict(),logger_kwargs=AttrDict()):
        self.env_fn = env_fn
        self.tb_logger = tb_logger
        self.env_name = env
        self.actor = actor 
        self.seed = seed 
        self.critic_kwargs = critic_kwargs
        self.actor_kwargs = actor_kwargs
        self.device = device
        self.RL_kwargs = RL_kwargs
        self.optim_kwargs = optim_kwargs
        self.logger_kwargs = logger_kwargs
        

        # logger 
        self.logger = EpochLogger(**self.logger_kwargs)
        self.logger.save_config(locals())

        # instantiating the environment
        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = ActorCritic(self.actor, self.env.observation_space, self.env.action_space, self.critic_kwargs, self.actor_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # move models to device
        self.ac = self.ac.to(self.device)
        self.ac_targ = self.ac_targ.to(self.device)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.RL_kwargs.replay_size, device=self.device)

        # Set up optimizers for policy and q-function
        if next(self.ac.pi.parameters(), None) is not None:
            self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.optim_kwargs.lr)
        
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=self.optim_kwargs.lr)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)


    def compute_loss_q(self, data, itr):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        # Target actions come from *current* policy
        o2 = o2.view(-1,1,o2.size()[-1]).repeat(1,self.ac.pi.num_particles,1).view(-1,o2.size()[-1])
        
        a2, logp_a2 = self.ac.pi.act(o2, False, True)

        with torch.no_grad(): 
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2).view(-1, self.ac.pi.num_particles).mean(-1)
            q2_pi_targ = self.ac_targ.q2(o2, a2).view(-1, self.ac.pi.num_particles).mean(-1)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            
            backup = r + self.RL_kwargs.gamma * (1 - d) * (q_pi_targ - self.RL_kwargs.alpha * logp_a2)
            #self.tb_logger.add_scalars('loss_q/backup',  {'total ': backup.mean(), 'q_pi_targ': (gamma * (1 - d) * q_pi_targ).mean(),'entr_term': - (gamma * (1 - d)* alpha * logp_a2).mean()  }, itr)
            #self.tb_logger.add_scalar('loss_q/backup/entr', - logp_a2.mean(), itr)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        self.tb_logger.add_scalar('loss_q/loss_q1',loss_q1, itr)
        self.tb_logger.add_scalar('loss_q/loss_q2',loss_q2, itr)
        
        # Useful info for logging
        q_info = AttrDict(Q1Vals=q1.detach().cpu().numpy(),Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info


    def compute_loss_pi(self, data, itr):
        o = data['obs']
        o = o.view(-1,1,o.size()[-1]).repeat(1,self.ac.actor.num_particles,1).view(-1,o.size()[-1])
        
        a, logp_pi = self.ac.pi.act(o, False, True)
        
        # get the final action
        q1_pi = self.ac.q1(o, a).view(-1, self.ac.actor.num_particles).mean(-1)
        q2_pi = self.ac.q2(o, a).view(-1, self.ac.actor.num_particles).mean(-1)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.RL_kwargs.alpha * logp_pi - q_pi).mean()

        self.tb_logger.add_scalar('loss_pi/q_pi',-q_pi.mean(), itr)
        self.tb_logger.add_scalar('loss_pi/logp_pi', logp_pi.mean(), itr)

        # Useful info for logging
        pi_info = AttrDict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info


    def update(self, data, itr):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data, itr)
        self.tb_logger.add_scalar('loss_q/total',loss_q, itr)

        loss_q.backward()
        self.q_optimizer.step()
        
        # self.env.debugging_metrics(itr)

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

        if next(self.ac.pi.parameters(), None) is not None:
            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data, itr)
            self.tb_logger.add_scalar('loss_pi/total',loss_pi, itr)

            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Record things
            self.logger.store(LossPi=loss_pi.item(), **pi_info)

            # log for p_0
            for tag, value in self.ac.named_parameters():
                if value.grad is not None:
                    self.tb_logger.add_histogram(tag + "/grad", value.grad.cpu(), itr)
        

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.optim_kwargs.polyak)
                p_targ.data.add_((1 - self.optim_kwargs.polyak) * p.data)

    def test_agent(self, itr=None):
        for j in range(self.RL_kwargs.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            
            while not(d or (ep_len == self.RL_kwargs.max_ep_len)):
                # Take deterministic actions at test time 
                a = self.ac.act(np.expand_dims(o, axis=0), deterministic=self.ac.pi.test_deterministic, test=True)
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                ep_len += 1
            
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        

    def forward(self):
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters(): 
            p.requires_grad = False
        
        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)
        
        # Prepare for interaction with environment
        o, ep_ret, ep_len = self.env.reset(), 0, 0 

        episode_itr = 0
        step_itr = 0
        
        # Main loop: collect experience in env and update/log each epoch
        while episode_itr < self.RL_kwargs.num_episodes:
            # Until exploration_episodes have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if episode_itr > self.RL_kwargs.exploration_episodes:
                a = self.ac.pi.act(np.expand_dims(o, axis=0), False, False)
            else:
                a = self.env.action_space.sample()  
            
            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.RL_kwargs.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.RL_kwargs.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0
                episode_itr += 1
            
            # Update handling
            if step_itr >= self.RL_kwargs.update_after and step_itr % self.RL_kwargs.update_every == 0:
                for j in range(self.RL_kwargs.update_every):
                    batch = self.replay_buffer.sample_batch(self.optim_kwargs.batch_size)
                    self.update(data=batch, itr=step_itr)

            # if (step_itr+1) % 1000 == 0:
            #     print('Plot ', episode_itr+1)
            #     self.env.plot_paths(episode_itr,1)
            #     self.env.plot_paths(episode_itr,20)


            if (episode_itr+1) % self.RL_kwargs.stats_episode_freq == 0:
                # Save model
                self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent(episode_itr)
                
                self.tb_logger.add_scalar('EpRet',np.mean(self.logger.epoch_dict['EpRet']), episode_itr)
                self.tb_logger.add_scalar('TestEpRet',np.mean(self.logger.epoch_dict['TestEpRet']) , episode_itr)
                
                # Log info about epoch
                self.logger.log_tabular('Episode', episode_itr)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', episode_itr)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('LogPi', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.dump_tabular()
            
            step_itr += 1

