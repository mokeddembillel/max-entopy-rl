from copy import deepcopy
import itertools
import numpy as np
import torch 
from torch.optim import Adam
from datetime import datetime
from actorcritic import ActorCritic
from utils import count_vars, AttrDict
from buffer import ReplayBuffer
from debugging import Debugging

class MaxEntrRL():
    def __init__(self, env_fn, env, actor, critic_kwargs=AttrDict(), actor_kwargs=AttrDict(), device="cuda",   
        RL_kwargs=AttrDict(), optim_kwargs=AttrDict(), tb_logger=None, fig_path=None):
        self.env_fn = env_fn
        self.tb_logger = tb_logger
        self.env_name = env
        self.actor = actor 
        self.device = device
        self.critic_kwargs = critic_kwargs
        self.actor_kwargs = actor_kwargs
        self.RL_kwargs = RL_kwargs
        self.optim_kwargs = optim_kwargs
        
        # instantiating the environment
        self.env, self.test_env = env_fn(self.tb_logger, actor, fig_path), env_fn(self.tb_logger, actor, fig_path)

        self.obs_dim = self.env.observation_space.shape[0]
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
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.RL_kwargs.replay_size, device=self.device, env_name=self.env_name)

        # Set up optimizers for policy and q-function
        if next(self.ac.pi.parameters(), None) is not None:
            self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.optim_kwargs.lr)
        
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=self.optim_kwargs.lr)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.debugger = Debugging(self.fig_path)


    def compute_loss_q(self, data, itr):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        # Target actions come from *current* policy
        # print('########## :', self.obs_dim)
        o2 = o2.view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim)
        a2, logp_a2 = self.ac(o2, deterministic=False, with_logprob=True) 
        # a2 = a2.detach()
        # logp_a2 = logp_a2.detach()

        with torch.no_grad(): 
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2).view(-1, self.ac.pi.num_particles)
            q2_pi_targ = self.ac_targ.q2(o2, a2).view(-1, self.ac.pi.num_particles)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            if self.actor == 'svgd_sql':
                V_soft_ = self.RL_kwargs.alpha * torch.logsumexp(q_pi_targ, dim=-1)
                V_soft_ += self.act_dim * torch.log(torch.tensor([2.]).to(self.device))
                backup = r + self.RL_kwargs.gamma * (1 - d) * V_soft_
            else:
                backup = r + self.RL_kwargs.gamma * (1 - d) * (q_pi_targ.mean(-1) - self.RL_kwargs.alpha * logp_a2)
                #backup = r + self.RL_kwargs.gamma * (1 - d) * (q_pi_targ.mean(-1)) 
        
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        self.tb_logger.add_scalars('Loss_q',  {'loss_q1 ': loss_q1, 'loss_q2': loss_q2, 'total': loss_q  }, itr)
        
        # plz add this to debugger/logger
        #Q1Vals.append(q1.cpu().detach().numpy())
        #Q2Vals.append(q2.cpu().detach().numpy())

        return loss_q


    def compute_loss_pi(self, data, itr):
        o = data['obs'].view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim)
        
        a, logp_pi = self.ac(o, deterministic=False, with_logprob=True)
        
        # get the final action
        q1_pi = self.ac.q1(o, a).view(-1, self.ac.pi.num_particles)
        q2_pi = self.ac.q2(o, a).view(-1, self.ac.pi.num_particles)
        q_pi = torch.min(q1_pi, q2_pi).mean(-1)

        # Entropy-regularized policy loss
        if self.actor == 'svgd_sql':
            loss_pi = (-q_pi).mean()
        else:
            loss_pi = (self.RL_kwargs.alpha * logp_pi - q_pi).mean()
            self.tb_logger.add_scalars('Loss_pi',  {'logp_pi ': (self.RL_kwargs.alpha * logp_pi).mean(), 'q_pi': -q_pi.mean(), 'total': loss_pi  }, itr)
            
        return loss_pi


    def update(self, data, itr):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data, itr)
        loss_q.backward()
        self.q_optimizer.step()
        
        if next(self.ac.pi.parameters(), None) is not None:
            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False
            
            loss_pi = self.compute_loss_pi(data, itr)

            if self.actor == 'svgd_sql':
                o = data['obs'].view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim)
                a, _ = self.ac(o, deterministic=False, with_logprob=True)
                fixed_a, _ = self.ac(o, deterministic=False, with_logprob=True)

                q1_pi = self.ac.q1(o, fixed_a).view(-1, self.ac.pi.num_particles)
                q2_pi = self.ac.q2(o, fixed_a).view(-1, self.ac.pi.num_particles)
                q_pi = torch.min(q1_pi, q2_pi)

                grad_pi = torch.autograd.grad(q_pi.sum(), fixed_a)[0]
                grad_pi = grad_pi.view(self.optim_kwargs.batch_size, self.actor_kwargs.num_svgd_particles, self.act_dim).unsqueeze(2)
                grad_pi = grad_pi.detach()
                assert grad_pi.shape == (self.optim_kwargs.batch_size, self.actor_kwargs.num_svgd_particles, 1, self.act_dim)
                
                a = a.view(self.optim_kwargs.batch_size, self.actor_kwargs.num_svgd_particles, self.act_dim)
                fixed_a = fixed_a.view(self.optim_kwargs.batch_size, self.actor_kwargs.num_svgd_particles, self.act_dim)

                kappa, _, _, grad_kappa = self.ac.pi.kernel(input_1=fixed_a, input_2=a)
                kappa = kappa.unsqueeze(-1)
                assert kappa.shape == (self.optim_kwargs.batch_size, self.actor_kwargs.num_svgd_particles, self.actor_kwargs.num_svgd_particles, 1)
                anneal = 1.
                action_gradients = (1 / self.actor_kwargs.num_svgd_particles) * torch.sum(anneal * kappa * grad_pi + grad_kappa, dim=1)
                assert action_gradients.shape == (self.optim_kwargs.batch_size, self.actor_kwargs.num_svgd_particles, self.act_dim)

                self.pi_optimizer.zero_grad()
                torch.autograd.backward(-a, grad_tensors=action_gradients)
                self.pi_optimizer.step()
            else:
                # Next run one gradient descent step for pi.
                self.pi_optimizer.zero_grad()
                loss_pi.backward()
                self.pi_optimizer.step()
                
            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True
        
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.optim_kwargs.polyak)
                p_targ.data.add_((1 - self.optim_kwargs.polyak) * p.data)

    def test_agent(self, itr=None):
        
        self.test_env.reset_rendering()

        TestEpRet = []
        TestEpLen = []

        for j in range(self.RL_kwargs.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            
            while not(d or (ep_len == self.env.max_steps)):
                o = torch.as_tensor(o, dtype=torch.float32).to(self.device).view(-1,self.obs_dim)
                o_ = o.view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim) # move this inside pi.act
                a, _ = self.ac(o_, deterministic=self.ac.pi.test_deterministic, with_logprob=False)
                o2, r, d, info = self.test_env.step(a.detach().cpu().numpy().squeeze())
                self.debugger.collect_data(self.ac, o, a, r, d, info, phase='test')    
                ep_ret += r
                ep_len += 1
                o = o2
        
            #print('ep_ret ', ep_ret)
            TestEpRet.append(ep_ret)
            TestEpLen.append(ep_len)
            
        
        self.test_env.render(itr=itr)
        self.debugger.plot_path_with_gaussian(itr=itr, phase='test')
        self.debugger.log_data_to_tensorboard(itr=itr, phase='test')
        # tensorboard logging
        self.tb_logger.add_scalars('TestEpRet',  {'Mean ': np.mean(TestEpRet), 'Min': np.min(TestEpRet), 'Max': np.max(TestEpRet)  }, itr)
        self.tb_logger.add_scalar('TestEpLen', np.mean(TestEpLen) , itr)


    def forward(self):
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters(): 
            p.requires_grad = False
                
        # Prepare for interaction with environment
        o, ep_ret, ep_len = self.env.reset(), 0, 0 

        episode_itr = 0
        step_itr = 0
        EpRet = []
        EpLen = []

        # Main loop: collect experience in env and update/log each epoch
        while episode_itr < self.RL_kwargs.num_episodes:
            # Until exploration_episodes have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if episode_itr > self.RL_kwargs.exploration_episodes:
                o_ = torch.as_tensor(o, dtype=torch.float32).to(self.device).view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim)
                a, _ = self.ac(o_, deterministic = False, with_logprob=False)
                a = a.detach().cpu().numpy().squeeze()
            else:
                a = self.env.action_space.sample()  
                
            
            # Step the env
            o2, r, d, info = self.env.step(a)
            ep_ret += r
            ep_len += 1
            #print('episode_itr: ', episode_itr, ' ep_len: ', ep_len)
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.env.max_steps else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d, info)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.env.max_steps):
                EpRet.append(ep_ret)
                EpLen.append(ep_len)

                o, ep_ret, ep_len = self.env.reset(), 0, 0
                episode_itr += 1
                d = True
                
            
            # Update handling
            if step_itr >= self.RL_kwargs.update_after and step_itr % self.RL_kwargs.update_every == 0:
                for j in range(self.RL_kwargs.update_every):
                    batch = self.replay_buffer.sample_batch(self.optim_kwargs.batch_size)
                    print('Update iteration ', episode_itr, j, self.RL_kwargs.update_every)
                    self.update(data=batch, itr=step_itr)

            
            if d and (episode_itr+1) % self.RL_kwargs.stats_episode_freq == 0:
                # Test the performance of the deterministic version of the agent.
                self.test_agent(episode_itr)
                
                for tag, value in self.ac.named_parameters():    ### commented right now ###
                    if value.grad is not None:
                        self.tb_logger.add_histogram(tag + "/grad", value.grad.cpu(), step_itr)
                
                self.tb_logger.add_scalars('EpRet',  {'Mean ': np.mean(EpRet), 'Min': np.min(EpRet), 'Max': np.max(EpRet)  }, episode_itr)
                self.tb_logger.add_scalar('EpLen',  np.mean(EpLen), episode_itr)

                EpRet = []
                EpLen = []
                
            step_itr += 1

