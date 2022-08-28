from copy import deepcopy
import itertools
import numpy as np
import torch 
from torch.optim import Adam
import gym
import random
import time
import math
#import wandb
from datetime import datetime
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger
from torch.utils.tensorboard import SummaryWriter
from spinup.algos.pytorch.sac.multigoal import MultiGoalEnv
from spinup.algos.pytorch.sac.plotter import QFPolicyPlotter
import torch.nn.functional as F
import gym_max_entropy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#import pdb; pdb.set_trace()
print('device: ', device) 

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size): 
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size) 
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}



# def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,    
#         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,    
#         polyak=0.995, lr_q=1e-3,lr_p=1e-3, alpha_q=5.0, alpha_p=5.0, batch_size=500, start_steps=10000,    
#         update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,   
#         logger_kwargs=dict(), save_freq=1, num_svgd_particles=10, num_svgd_steps=10, svgd_lr=0.01,
#         sac_version=None, svgd_test_deterministic=False, exp_env="multigoal"): 
def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,    
        steps_per_epoch=500, epochs=20, replay_size=int(1e6), gamma=0.99,    
        polyak=0.995, lr_q=1e-3,lr_p=1e-3, alpha_q=5.0, alpha_p=5.0, batch_size=100, start_steps=100000,    
        update_after=0, update_every=1000, num_test_episodes=10, max_ep_len=500,   
        logger_kwargs=dict(), save_freq=1, num_svgd_particles=10, num_svgd_steps=10, svgd_lr=0.01,
        sac_version=None, svgd_test_deterministic=False, exp_env=None): 
# def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,    
#         steps_per_epoch=50, epochs=None, replay_size=int(1e6), gamma=0.99,    
#         polyak=0.995, lr_q=1e-3,lr_p=1e-3, alpha_q=5.0, alpha_p=5.0, batch_size=5, start_steps=0,    
#         update_after=5, update_every=5, num_test_episodes=10, max_ep_len=30,   
#         logger_kwargs=dict(), save_freq=1, num_svgd_particles=10, num_svgd_steps=10, svgd_lr=0.01,
#         sac_version=None, svgd_test_deterministic=False, exp_env="multigoal"): 

    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    assert sac_version in {'orig', 'svgd_v1', 'svgd_v2a'}
    assert exp_env in {'spinningup', 'multigoal'}

    if sac_version == 'orig':
        num_svgd_particles = 0
        num_svgd_steps = 0
        test_deterministic=True
    else:
        test_deterministic = svgd_test_deterministic


    # logger 
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # fix the seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    EPS_START = 0.9
    EPS_END = 0.3
    EPS_DECAY = 200

    if (exp_env == 'spinningup'):
        env, test_env = env_fn(), env_fn()
        env_name = env.unwrapped.spec.id
    elif exp_env == 'multigoal':
        env, test_env = MultiGoalEnv(), MultiGoalEnv()
        env_name = 'multigoal'
        
        


    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # assess state [0,0]
    init_state = torch.tensor([0.0,0.0]).to(device)
    a_up = torch.tensor([0.0,0.7]).to(device)
    a_down = torch.tensor([0.0,-0.7]).to(device)
    a_left = torch.tensor([-0.7,0.0]).to(device)
    a_right = torch.tensor([0.7,0.0]).to(device)


    #tensorboard
    project_name = 'loss_p_200k_elu_'+env_name + '_' + datetime.now().strftime("%b_%d_%Y_%H_%M_%S")+ '_' + sac_version+'_svgd_steps_'+str(num_svgd_steps)+'_svgd_particles_'+str(num_svgd_particles)+'_svgd_lr_'+str(svgd_lr)+'_alphaQ_'+str(alpha_q)+'_alphaP_'+str(alpha_p)+'_batch_size_'+str(batch_size)+'_lr_p_'+str(lr_p)
    writer = SummaryWriter('runs/'+project_name)

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, num_svgd_particles, num_svgd_steps, svgd_lr, sac_version, device, **ac_kwargs)#, writer=None)
    ac_targ = deepcopy(ac)
    
    # move models to GPU
    ac = ac.to(device)
    ac_targ = ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters(): 
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    if (sac_version=="svgd_v1"):
        var_counts = tuple(core.count_vars(module) for module in [ac.q1, ac.q2])
        logger.log('\nNumber of parameters: \t q1: %d, \t q2: %d\n'%var_counts)  
    else:
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data, itr):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        # Target actions come from *current* policy
        if (num_svgd_particles>0):
            o2 = o2.view(-1,1,o2.size()[-1]).repeat(1,num_svgd_particles,1).view(-1,o2.size()[-1])

        if sac_version == 'svgd_v1':
            # sample action from a normal distribution
            a2 = torch.normal(0, 1, size=(len(o)*num_svgd_particles,a.size()[-1])).to(device)
            logp0_a2 = (a2.size()[-1]/2) * np.log(2 * np.pi) + (a2.size()[-1]/2)
            logp0_a2 += (2*(np.log(2) - a2 - F.softplus(-2*a2))).sum(axis=-1)
            a2 = act_limit * torch.tanh(a2) 
            
            #run svgd
            a2, logq_a2, phi_a2 = ac.svgd_sampler(o2, a2.detach()) 
            a2 = a2.detach()
            
            # compute the entropy 
            logp_a2 = (-logp0_a2.view(-1,num_svgd_particles) + logq_a2).mean(-1)
            logp_a2 = logp_a2.detach()

        elif sac_version == 'svgd_v2a':
            #import pdb; pdb.set_trace()
            a2, logp0_a2 = ac.pi(o2)
            a2, logq_a2,_= ac.svgd_sampler(o2, a2.detach())
            a2 = a2.detach()

            logp_a2 = logp0_a2 + logq_a2.mean(-1)
            logp_a2 = logp_a2.detach()

            for k in range(a2.size()[-1]):
                writer.add_scalar('action_std/a['+str(k)+']', ac.pi.std[:,k].mean(0), itr)

        elif sac_version == 'orig':
            a2, logp_a2 = ac.pi(o2)
            logp_a2 = logp_a2.detach()
            a2 = a2.detach()
            
            writer.add_histogram('logp0', logp_a2, itr) 
            
            for k in range(a2.size()[-1]):
                writer.add_scalar('action_std/a['+str(k)+']', ac.pi.std[:,k].mean(0), itr)
        
        with torch.no_grad(): 
            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)

            if (num_svgd_particles > 0):
                q1_pi_targ = q1_pi_targ.view(-1, num_svgd_particles).mean(-1)
                q2_pi_targ = q2_pi_targ.view(-1, num_svgd_particles).mean(-1)

            #import pdb; pdb.set_trace()
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            #import pdb; pdb.set_trace()
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha_q * logp_a2)
            #backup = r + gamma * (1 - d) * q_pi_targ 
            writer.add_scalars('loss_q/backup',  {'total ': backup.mean(), 'q_pi_targ': (gamma * (1 - d) * q_pi_targ).mean(),'entr_term': - (gamma * (1 - d)* alpha_q * logp_a2).mean()  }, itr)
            writer.add_scalar('loss_q/backup/entr', - logp_a2.mean(), itr)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        writer.add_scalar('loss_q/loss_q1',loss_q1, itr)
        writer.add_scalar('loss_q/loss_q2',loss_q2, itr)
        
        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data, itr):

        o = data['obs']
        
        if (num_svgd_particles > 0):
            o = o.view(-1,1,o.size()[-1]).repeat(1,num_svgd_particles,1).view(-1,o.size()[-1])
        
        a_0, logp_pi = ac.pi(o)
        
        if sac_version == 'svgd_v2a':
            
            if (itr > 200000):
                q1_pi_0 = ac.q1(o, a_0).view(-1, num_svgd_particles).mean(-1)
                q2_pi_0 = ac.q2(o, a_0).view(-1, num_svgd_particles).mean(-1)
                q_pi_0 = torch.min(q1_pi_0, q2_pi_0)
            
            a, logq_a,_= ac.svgd_sampler(o,a_0,with_logprob=False)  
        else:
            a = a_0
        
        # get the final action
        q1_pi = ac.q1(o, a)
        q2_pi = ac.q2(o, a)

        if (num_svgd_particles >0):
            q1_pi = q1_pi.view(-1, num_svgd_particles).mean(-1)
            q2_pi = q2_pi.view(-1, num_svgd_particles).mean(-1)

        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        #
        
        if (itr > 200000):
            loss_pi = (alpha_p * logp_pi - 0.5 * q_pi - 0.5 * q_pi_0).mean()
            writer.add_scalar('loss_pi/q_pi',-q_pi_0.mean(), itr)
        else:
            loss_pi = (alpha_p * logp_pi - q_pi).mean()

        writer.add_scalar('loss_pi/q_pi',-q_pi.mean(), itr)
        writer.add_scalar('loss_pi/logp_pi', logp_pi.mean(), itr)

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    if sac_version != 'svgd_v1':
        pi_optimizer = Adam(ac.pi.parameters(), lr=lr_p)
    
    q_optimizer = Adam(q_params, lr=lr_q)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def debugging_metrics(itr):
        q_up = ac.q1(init_state,a_up).detach()
        q_down = ac.q1(init_state,a_down).detach()
        q_left = ac.q1(init_state,a_left).detach()
        q_right = ac.q1(init_state,a_right).detach()
        writer.add_scalars('init_state/q_val',{'q_up': q_up, 'q_down':q_down, 'q_left':q_left, 'q_right':q_right}, itr)

        def compute_q(init_s, action):
            return ac.q1(init_s,action)


        # add curvature and gradient
        a_up_ = a_up.requires_grad_(True)
        init_state_ = init_state.requires_grad_(True)
        grad_up_ = torch.autograd.grad(ac.q1(init_state_,a_up_), a_up_,retain_graph=True, create_graph=True)[0]
        grad_up = torch.abs(grad_up_).mean()
        hess_up = ((torch.abs(torch.autograd.grad(grad_up_[0],a_up_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_up_[1],a_up_,retain_graph=True)[0])).sum()/4).detach()
        #hess_up = torch.autograd.functional.hessian(compute_q, (init_state_,a_up_) )
        #hess_up = torch.abs(hess_up[1][1]).mean().detach()
        ###
        a_down_ = a_down.requires_grad_(True)
        grad_down_ = torch.autograd.grad(ac.q1(init_state_,a_down_), a_down_,retain_graph=True, create_graph=True)[0]
        grad_down = torch.abs(grad_down_).mean()
        hess_down = ((torch.abs(torch.autograd.grad(grad_down_[0],a_down_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_down_[1],a_down_,retain_graph=True)[0])).sum()/4).detach()
        #hess_down = torch.autograd.functional.hessian(compute_q, (init_state_,a_down_) )
        #hess_down = torch.abs(hess_down[1][1]).mean().detach()
        ###
        a_left_ = a_left.requires_grad_(True)
        grad_left_ = torch.autograd.grad(ac.q1(init_state_,a_left_), a_left_,retain_graph=True, create_graph=True)[0]
        grad_left = torch.abs(grad_left_).mean()
        hess_left = ((torch.abs(torch.autograd.grad(grad_left_[0],a_left_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_left_[1],a_left_,retain_graph=True)[0])).sum()/4).detach()
        #hess_left = torch.autograd.functional.hessian(compute_q, (init_state_,a_left_) )
        #hess_left = torch.abs(hess_left[1][1]).mean().detach()
        ###
        a_right_ = a_right.requires_grad_(True)
        grad_right_ = torch.autograd.grad(ac.q1(init_state_,a_right_), a_right_,retain_graph=True, create_graph=True)[0]
        grad_right = torch.abs(grad_right_).mean()
        hess_right = ((torch.abs(torch.autograd.grad(grad_right_[0],a_right_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_right_[1],a_right_,retain_graph=True)[0])).sum()/4).detach()
        #hess_right = torch.autograd.functional.hessian(compute_q, (init_state_,a_right_) )
        #hess_right = torch.abs(hess_right[1][1]).mean().detach()
        
        #import pdb; pdb.set_trace()  
        writer.add_scalars('init_state/hessian',{'hess_up': hess_up, 'hess_down':hess_down, 'hess_left':hess_left, 'hess_right':hess_right}, itr)
        writer.add_scalars('init_state/grad',{'grad_up': grad_up, 'grad_down':grad_down, 'grad_left':grad_left, 'grad_right':grad_right}, itr)
        # compute the variance of running svgd
        num_samples = 100

        s_up = a_up.view(-1,1,a_up.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_up.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_up),2)).to(device)
        a_svgd_up, _, _ = ac.svgd_sampler(s_up, a_rand.detach()) 
        q_svgd_up = ac.q1(s_up,a_svgd_up).detach()
        q_svgd_up_var = torch.var(q_svgd_up)
        
        s_down = a_down.view(-1,1,a_down.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_down.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_down),2)).to(device)
        a_svgd_down, _, _ = ac.svgd_sampler(s_down, a_rand.detach()) 
        q_svgd_down = ac.q1(s_down,a_svgd_down).detach()
        q_svgd_down_var = torch.var(q_svgd_down)
        
        s_left = a_left.view(-1,1,a_left.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_left.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_left),2)).to(device)
        a_svgd_left, _, _ = ac.svgd_sampler(s_left, a_rand.detach()) 
        q_svgd_left = ac.q1(s_left,a_svgd_left).detach()
        q_svgd_left_var = torch.var(q_svgd_left)
        
        s_right = a_left.view(-1,1,a_right.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_right.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_right),2)).to(device)
        a_svgd_right, _, _ = ac.svgd_sampler(s_right, a_rand.detach()) 
        q_svgd_right = ac.q1(s_right,a_svgd_right).detach()
        q_svgd_right_var = torch.var(q_svgd_right)

        #if itr==2000:
        #    import pdb; pdb.set_trace()
        writer.add_scalars('init_state/q_var',{'q_up': q_svgd_up_var, 'q_down':q_svgd_down_var, 'q_left':q_svgd_left_var, 'q_right':q_svgd_right_var}, itr)


    def update(data, itr):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data, itr)
        #wandb_vis.log({'loss_q': loss_q })
        writer.add_scalar('loss_q/total',loss_q, itr)

        loss_q.backward()
        q_optimizer.step()

        #print('loss_q:  ', loss_q)
        # debugging_metrics(itr)

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)


        if sac_version != 'svgd_v1':
            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data, itr)
            #wandb_vis.log({'loss_pi': loss_pi })
            writer.add_scalar('loss_pi/total',loss_pi, itr)

            loss_pi.backward()
            pi_optimizer.step()
            #print('loss_pi:  ', loss_pi)

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item(), **pi_info)

            # log for p_0
            for tag, value in ac.named_parameters():
                if value.grad is not None:
                    writer.add_histogram(tag + "/grad", value.grad.cpu(), itr)
        

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False, num_svgd_particles=None, test=False, plot=False, itr=None, writer=None):
        assert(len(o.shape)==2)
        if sac_version == 'svgd_v1':
            return ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), deterministic, num_svgd_particles, test=test, plot=plot, itr=itr, writer=writer)
        else:
            a, _ = ac.pi(torch.as_tensor(o, dtype=torch.float32).to(device), with_logprob=False)
            a = a.squeeze().detach().cpu().numpy()
            # print('######################## ',a)
            return a

    def test_agent(itr=None):
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                a = get_action(np.expand_dims(o, axis=0), deterministic=test_deterministic, test=True, itr=itr, writer=writer)
                
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        test_env.render()
        test_env.save_fig('/Users/admin/Desktop/Code/GIT repositories/AI/STAC/spinningup/spinup/algos/pytorch/sac/max_entropy_plots_/'+ str(itr))   
        test_env.reset_rendering()
    def plot_paths(epoch, num_episodes, eps):
        paths = []        
        plot_svgd_steps = False

        env = MultiGoalEnv()


        ac_hess_list = []
        ac_score_func_list = []
        ac_hess_eig_max = []
        #
        #for episode in range(50):
        for episode in range(num_episodes): 
            #print("_____________"+str(episode)+"______________")
            observation = env.reset() 
            done = False
            step = 0
            path = {'infos':{'pos':[], 'mu':[], 'std':[], 'svgd_steps':[] }}
            particles = None

            while not done and step < 30 :
                actions = get_action(np.expand_dims(observation, axis=0), test=True, plot=True, writer=writer)
                
                path['infos']['pos'].append(observation)

                #import pdb; pdb.set_trace()
                if (sac_version == "svgd_v1"):
                    path['infos']['mu'].append( np.zeros( observation.shape ) )
                    path['infos']['std'].append( np.ones( observation.shape ) )

                    svgd_actions = torch.stack(ac.svgd_steps)
                    path['infos']['svgd_steps'].append(svgd_actions.cpu().numpy())
                    plot_svgd_steps=True

                    ac_hess_list.append(ac.hess_list)
                    ac_score_func_list.append(ac.score_func_list)
                    ac_hess_eig_max.append(ac.hess_eig_max)
                
                elif (sac_version == "svgd_v2a"):
                    path['infos']['mu'].append(ac.pi.mu.cpu().numpy())
                    path['infos']['std'].append(ac.pi.std.cpu().numpy())

                    svgd_actions = torch.stack(ac.svgd_steps)
                    path['infos']['svgd_steps'].append(svgd_actions.cpu().numpy())
                    plot_svgd_steps=True

                    ac_hess_list.append(ac.hess_list)
                    ac_score_func_list.append(ac.score_func_list)
                    ac_hess_eig_max.append(ac.hess_eig_max)
                else:
                    path['infos']['mu'].append(ac.pi.mu.cpu().numpy())
                    path['infos']['std'].append(ac.pi.std.cpu().numpy())

                #import pdb; pdb.set_trace()
                observation, reward, done, _ = env.step(actions)
                
                #print(observation)
                step +=1
            
            paths.append(path)
        
        print("saving figure..., epoch=",epoch)
        number_of_hits_mode = env.render_rollouts(paths,num_episodes, plot_svgd_steps, epoch=epoch, eps=eps, fout='loss_p_200k_elu_'+sac_version+"_alphaQ_"+str(alpha_q)+"_alphaP_"+str(alpha_p)+'_svgd_steps_'+str(num_svgd_steps)+'_svgd_particles_'+str(num_svgd_particles)+'_svgd_lr_'+str(svgd_lr)+"_epoch_"+str(epoch)+"_batch_size_"+str(batch_size)+'_lr_p_'+str(lr_p)+"_num_episodes_"+str(num_episodes)+".png" )
        total_number_of_hits_mode = number_of_hits_mode.sum()
        if total_number_of_hits_mode > 0.0:
            m0 = number_of_hits_mode[0]/total_number_of_hits_mode
            m1 = number_of_hits_mode[1]/total_number_of_hits_mode
            m2 = number_of_hits_mode[2]/total_number_of_hits_mode
            m3 = number_of_hits_mode[3]/total_number_of_hits_mode
        else:
            m0, m1, m2, m3 = 0, 0, 0, 0
        if sac_version == 'svgd_v1' or sac_version == 'svgd_v2a':
            ac_hess_list = torch.stack(ac_hess_list)
            ac_score_func_list = torch.stack(ac_score_func_list)
            ac_hess_eig_max = torch.stack(ac_hess_eig_max)

            # 
            writer.add_scalar('smoothness/ac_score/mean', torch.abs(ac_score_func_list).mean() , epoch)
            writer.add_scalar('smoothness/ac_score/std', torch.abs(ac_score_func_list).std() , epoch)
            writer.add_scalar('smoothness/hess/mean', torch.abs(ac_hess_list).mean() , epoch)
            writer.add_scalar('smoothness/hess/std', torch.abs(ac_hess_list).std() , epoch)
            writer.add_scalar('smoothness/hess/max_eigen_val/mean', ac_hess_eig_max.mean() , epoch)
            writer.add_scalar('smoothness/hess/max_eigen_val/std', ac_hess_eig_max.std() , epoch)

        # 
        writer.add_scalar('modes/num_modes',(number_of_hits_mode>0).sum(), epoch)
        writer.add_scalar('modes/total_number_of_hits_mode',total_number_of_hits_mode, epoch)
        writer.add_scalar('modes/prob_mod_0',m0, epoch)
        writer.add_scalar('modes/prob_mod_1',m1, epoch)
        writer.add_scalar('modes/prob_mod_2',m2, epoch)
        writer.add_scalar('modes/prob_mod_3',m3, epoch)

        picks = 0

        #import pdb; pdb.set_trace() 
        if (epoch>200000):
            if ((number_of_hits_mode>0).sum()!=4):
                picks += 1
                stability = 1 - (picks/(epoch-200000))
                writer.add_scalar('modes/stability',stability, epoch)
        

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0 

    ep_done = 0
    success_buffer = []
    goals = [0, 0]
    for t in range(start_steps):
        a = env.action_space.sample()  
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        success_buffer.append((o, a, r, o2, d))
        if info['status'] == 'succeeded':
            goals[info['goal'] - 1] +=1
            print('Adding a success traj Iteration number ', t, replay_buffer.size + 1, goals)
            for expr in success_buffer:
                replay_buffer.store(expr[0], expr[1], expr[2], expr[3], expr[4])
        elif info['status'] == 'failed':
            # print('Removing a fail traj')
            success_buffer = []
        
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
            ep_done += 1
    
    
    ep_done = 0
    eps_threshold = 1
    success_buffer = []
    o, ep_ret, ep_len = env.reset(), 0, 0 
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        print('Iteration number ', t)
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        #a = env.action_space.sample()  
        
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * ep_done / EPS_DECAY) 
        writer.add_scalar('eps_threshold', eps_threshold, t)

        eps = random.random()
        #import pdb; pdb.set_trace() 
        ##
        if eps > eps_threshold:
            #print('exploitation ... ')
            a = get_action(np.expand_dims(o, axis=0), writer=writer)# itr=t, writer=writer)
            a = a[np.random.randint(len(a))]
        else:
            if sac_version == 'svgd_v1':
                #print('exploration ... ')
                a = torch.normal(0, 1, size=[act_dim] )
                a = (act_limit * torch.tanh(a)).numpy()
            else:
                a, _ = ac.pi(torch.as_tensor(np.expand_dims(o, axis=0), dtype=torch.float32).to(device), with_logprob=False)
                a = a.squeeze().detach().cpu().numpy()
            
        
        # Step the env
        # print('######################## ',a)
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        # print('############ ', info)
        success_buffer.append((o, a, r, o2, d))
        if info['status'] == 'succeeded':
            goals[info['goal'] - 1] +=1
            print('Adding a success traj Iteration number ', t, replay_buffer.size + 1, goals)
            for expr in success_buffer:
                replay_buffer.store(expr[0], expr[1], expr[2], expr[3], expr[4])
        elif info['status'] == 'failed':
            # print('Removing a fail traj')
            success_buffer = []

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
            ep_done += 1
        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, itr=t)


        # if (t+1) % 1000 == 0:
        #     print('Plot ', t+1)
        #     plot_paths(t,1, eps_threshold)
        #     plot_paths(t,20, eps_threshold)
            
            #plotter = QFPolicyPlotter(qf = ac.q1, policy=ac, obs_lst=[[0,0],[-2.5,-2.5],[2.5,2.5]], eps=eps_threshold, default_action =[np.nan,np.nan], n_samples=1, epoch=t, alpha_q=alpha_q, batch_size=batch_size, sac_version=sac_version, svgd_steps = num_svgd_steps, svgd_particles = num_svgd_particles, svgd_lr = svgd_lr, device=device)
            #plotter.draw()
            
            #plotter._n_samples = 100 
            #plotter.draw()
            
            #plotter._n_samples = 100 
            #plotter.draw(with_entropy=True)
        
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch 

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs): 
                print('save env ...') 
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(t)
            
            writer.add_scalar('EpRet',np.mean(logger.epoch_dict['EpRet']), epoch)
            writer.add_scalar('TestEpRet',np.mean(logger.epoch_dict['TestEpRet']) , epoch)
            
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossQ', average_only=True)

            if (sac_version != 'svgd_v1'):
                logger.log_tabular('LogPi', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)

            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
            
    # save the model in the exchangeable ONNX format
    #torch.onnx.export(ac, ,"model.onnx")
    #wandb_vis.save("model.onnx")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='max-entropy-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--svgd_particles', type=int, default=10)
    parser.add_argument('--svgd_steps', type=int, default=5)
    parser.add_argument('--svgd_lr', type=float, default=0.1)
    parser.add_argument('--sac_version', type=str, default='orig')
    parser.add_argument('--svgd_test_deterministic', type=bool, default=False)
    # parser.add_argument('--exp_env', type=str, default='multigoal')
    parser.add_argument('--exp_env', type=str, default='spinningup')
    args = parser.parse_args()
    
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed,logger_kwargs=logger_kwargs, num_svgd_particles=args.svgd_particles, 
        num_svgd_steps=args.svgd_steps, svgd_lr=args.svgd_lr, sac_version=args.sac_version, svgd_test_deterministic=args.svgd_test_deterministic, exp_env=args.exp_env)
