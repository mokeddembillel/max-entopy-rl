from copy import deepcopy
import itertools
import numpy as np
import torch 
from torch.optim import Adam
from datetime import datetime
from actorcritic import ActorCritic
from utils import count_vars, AttrDict, pdf_or_png_to_gif
from buffer import ReplayBuffer
from debugging import Debugger
import pickle
from tqdm import tqdm
from render_browser import render_browser


class MaxEntrRL():
    def __init__(self, train_env, test_env, env, actor, critic_kwargs=AttrDict(), actor_kwargs=AttrDict(), device="cuda",   
        RL_kwargs=AttrDict(), optim_kwargs=AttrDict(), tb_logger=None, fig_path=None):
        self.fig_path = fig_path
        self.env_name = env
        self.actor = actor 
        self.device = device
        self.critic_kwargs = critic_kwargs
        self.actor_kwargs = actor_kwargs
        self.RL_kwargs = RL_kwargs
        self.optim_kwargs = optim_kwargs
        
        # instantiating the environment
        self.env, self.test_env = train_env, test_env

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = ActorCritic(self.actor, self.env.observation_space, self.env.action_space, self.RL_kwargs.evaluation_data_path, 
            self.RL_kwargs.test_time, self.RL_kwargs.model_path, self.critic_kwargs, self.actor_kwargs)
        # self.ac.state_dict
        self.ac_targ = deepcopy(self.ac)

        # move models to device
        self.ac = self.ac.to(self.device)
        self.ac_targ = self.ac_targ.to(self.device)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.RL_kwargs.replay_size, load_replay=self.RL_kwargs.load_replay, replay_path=self.RL_kwargs.replay_path, device=self.device, env_name=self.env_name)

        if next(self.ac.pi.parameters(), None) is not None:
            self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.optim_kwargs.lr_actor)
        
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=self.optim_kwargs.lr_critic)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.debugger = Debugger(tb_logger, self.ac, self.env_name, self.env, self.test_env, self.RL_kwargs.plot_format, self.RL_kwargs.update_after, self.RL_kwargs.num_test_episodes, self.RL_kwargs.alpha, self.RL_kwargs.max_steps, self.RL_kwargs.max_experiment_steps)

        self.evaluation_data = AttrDict()


    def compute_loss_q(self, data, itr):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)


        o2 = o2.view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim)
        a2, logp_a2 = self.ac(o2, action_selection=None, with_logprob=True, in_q_loss=False) 
        
        with torch.no_grad(): 
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2).view(-1, self.ac.pi.num_particles)
            q2_pi_targ = self.ac_targ.q2(o2, a2).view(-1, self.ac.pi.num_particles)

            
            
            ### option 1
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.RL_kwargs.gamma * (1 - d) * (q_pi_targ.mean(-1) - self.RL_kwargs.alpha * logp_a2)  

            self.debugger.add_scalars('Q_target/',  {'r': r.mean(), 'Q': (self.RL_kwargs.gamma * (1 - d) * q_pi_targ.mean(-1)).mean(),\
                'entropy': (self.RL_kwargs.gamma * (1 - d) * self.RL_kwargs.alpha * logp_a2).mean(), 'backup': backup.mean(), 'pure_entropy':logp_a2.mean()}, itr)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        self.debugger.add_scalars('Loss_q',  {'loss_q1 ': loss_q1, 'loss_q2': loss_q2, 'total': loss_q  }, itr)
        
        return loss_q



    def update(self, data, itr):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data, itr)
        loss_q.backward()
        
        self.q_optimizer.step()
        
        
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.optim_kwargs.polyak)
                p_targ.data.add_((1 - self.optim_kwargs.polyak) * p.data)
                
    # @render_browser
    def test_agent(self, itr=None):
        robot_pic_rgb = None

        for j in tqdm(range(self.RL_kwargs.num_test_episodes)):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            
            obs_average = []
            while not(d or (ep_len == self.RL_kwargs.max_steps)):
                o = torch.as_tensor(o, dtype=torch.float32).to(self.device).view(-1,self.obs_dim)
                o_ = o.view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim) # move this inside pi.act
                obs_average.append(o.detach().cpu().numpy())
                a, log_p = self.ac(o_, action_selection=self.ac.pi.test_action_selection, with_logprob=True)
                o2, r, d, _ = self.test_env.step(a.detach().cpu().numpy().squeeze())

                self.debugger.collect_data(o, a.detach(), o2, r, d, log_p, itr, ep_len, robot_pic_rgb=robot_pic_rgb)    
                
                ep_ret += r
                ep_len += 1
                
                o = o2
            print('####### --actor: ', self.actor, ' --alpha: ', str(self.RL_kwargs.alpha) , ' --ep_return: ', ep_ret, ' --ep_length: ', ep_len)
            if not self.RL_kwargs.test_time:
                self.evaluation_data['test_episodes_return'].append(ep_ret)
                self.evaluation_data['test_episodes_length'].append(ep_len)

        self.debugger.log_to_tensorboard(itr=itr)
        self.debugger.reset()
        if not self.RL_kwargs.test_time:
            self.ac.save(itr)
        
    def save_data(self):
        pickle.dump(self.evaluation_data, open(self.RL_kwargs.evaluation_data_path + '/evaluation_data.pickle', "wb"))
        self.ac.save()

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


        self.evaluation_data['train_episodes_return'] = []
        self.evaluation_data['train_episodes_length'] = []
        self.evaluation_data['test_episodes_return'] = []
        self.evaluation_data['test_episodes_length'] = []
        
        # Main loop: collect experience in env and update/log each epoch
        # while step_itr < self.RL_kwargs.max_experiment_steps:
        for step_itr in tqdm(range(self.RL_kwargs.max_experiment_steps)):
            # Until exploration_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if step_itr >= self.RL_kwargs.exploration_steps:
                o_ = torch.as_tensor(o, dtype=torch.float32).to(self.device).view(-1,1,self.obs_dim).repeat(1,self.ac.pi.num_particles,1).view(-1,self.obs_dim)
                a, logp = self.ac(o_, action_selection = self.RL_kwargs.train_action_selection, with_logprob=True, itr=step_itr)
                a = a.detach().cpu().numpy().squeeze()
            else:
                a = self.env.action_space.sample()
               

            # Step the env
            o2, r, d, info = self.env.step(a)
            ep_ret += r
            ep_len += 1
           
            d = False if ep_len == self.RL_kwargs.max_steps else d
            
            self.replay_buffer.store(o, a, r, o2, d, info, step_itr)
                

            o = o2

            # End of trajectory handling
            if d or (ep_len == self.RL_kwargs.max_steps):
                EpRet.append(ep_ret)
                EpLen.append(ep_len)
                self.evaluation_data['train_episodes_return'].append(ep_ret)
                self.evaluation_data['train_episodes_length'].append(ep_len)
                
                o, ep_ret, ep_len = self.env.reset(), 0, 0
                episode_itr += 1
                d = True    

            
            # Update handling
            if step_itr >= self.RL_kwargs.update_after and step_itr % self.RL_kwargs.update_every == 0:
                if step_itr == self.RL_kwargs.update_after:
                    print('######################## Starting models update ########################')
                for j in range(self.RL_kwargs.update_every):
                    batch = self.replay_buffer.sample_batch(self.optim_kwargs.batch_size)
                    self.update(data=batch, itr=step_itr)

            
            if ((step_itr+1)  >= self.RL_kwargs.collect_stats_after and (step_itr+1) % self.RL_kwargs.stats_steps_freq == 0) or step_itr == self.RL_kwargs.max_experiment_steps - 1:
                self.test_agent(step_itr)
                try:
                    self.debugger.add_scalars('EpRet/return_detailed',  {'Mean ': np.mean(EpRet), 'Min': np.min(EpRet), 'Max': np.max(EpRet)  }, step_itr)
                    self.debugger.add_scalars('EpRet/return_mean_only',  {'Mean ': np.mean(EpRet)}, step_itr)
                    self.debugger.add_scalar('EpLen',  np.mean(EpLen), step_itr)
                except:
                    print('Statistics collection frequency should be larger then the length of an episode!')
                EpRet = []
                EpLen = []
            step_itr += 1
        self.save_data()
       
