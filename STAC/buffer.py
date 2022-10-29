import torch 
import numpy as np
from utils import AttrDict

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size, device, env_name, episode_max_steps=500): 
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env_name = env_name
        
        self.obs_buf = np.zeros((size, self.obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device
        self.goals = None
        if self.env_name == 'max-entropy-v0':
            self.episode_max_steps = episode_max_steps
            self.obs_tmp = np.zeros((self.episode_max_steps, self.obs_dim), dtype=np.float32)
            self.obs2_tmp = np.zeros((self.episode_max_steps, self.obs_dim), dtype=np.float32)
            self.act_tmp = np.zeros((self.episode_max_steps, self.act_dim), dtype=np.float32)
            self.rew_tmp = np.zeros((self.episode_max_steps,), dtype=np.float32)
            self.done_tmp = np.zeros((self.episode_max_steps,), dtype=np.float32)
            self.ptr_tmp = 0
            self.goals = [0, 0]

    def store(self, obs, act, rew, next_obs, done, env_info=None):
        
        if self.env_name == 'max-entropy-v0':
        # if False:
            self.obs_tmp[self.ptr_tmp] = obs
            self.obs2_tmp[self.ptr_tmp] = next_obs
            self.act_tmp[self.ptr_tmp] = act
            self.rew_tmp[self.ptr_tmp] = rew
            self.done_tmp[self.ptr_tmp] = done
            self.ptr_tmp = self.ptr_tmp + 1
            
            if env_info['status'] == 'succeeded' or env_info['status'] == 'failed' and np.random.uniform(0,1) > 0.8:
                if env_info['status'] == 'succeeded':
                    self.goals[env_info['goal'] - 1] +=1 # will be removed later
                # print('Adding a trajectory: ', env_info['status'], ' --- goals: ', self.goals, ' --- buffer size: ', self.size + 1)
                if self.max_size - self.ptr < self.ptr_tmp:
                    self.ptr = self.max_size - self.ptr_tmp
                
                self.obs_buf[self.ptr:self.ptr+self.ptr_tmp] = self.obs_tmp[:self.ptr_tmp]
                self.obs2_buf[self.ptr:self.ptr+self.ptr_tmp] = self.obs2_tmp[:self.ptr_tmp]
                self.act_buf[self.ptr:self.ptr+self.ptr_tmp] = self.act_tmp[:self.ptr_tmp]
                self.rew_buf[self.ptr:self.ptr+self.ptr_tmp] = self.rew_tmp[:self.ptr_tmp]
                self.done_buf[self.ptr:self.ptr+self.ptr_tmp] = self.done_tmp[:self.ptr_tmp]
                self.ptr = (self.ptr + self.ptr_tmp) % self.max_size
                self.size = min(self.size + self.ptr_tmp, self.max_size)
                self.ptr_tmp = 0
            elif env_info['status'] == 'failed': 
                self.ptr_tmp = 0
        else:
            self.obs_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.ptr = (self.ptr+1) % self.max_size
            self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size) 
        batch = AttrDict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}

