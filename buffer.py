import torch 
import numpy as np
from utils import combined_shape, AttrDict

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size, device, env_name, episode_max_steps): 
        self.env_name = env_name
        self.episode_max_steps = episode_max_steps
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

        self.obs_tmp = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_tmp = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_tmp = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_tmp = np.zeros(size, dtype=np.float32)
        self.ptr_tmp, self.size_tmp = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        # if self.env_name == 'max-entropy-v0':
        #     success_buffer.append((o, a, r, o2, d))
        #     if info['status'] == 'succeeded':
        #         goals[info['goal'] - 1] +=1
        #         print('Adding a success traj episode number ', episode_itr, replay_buffer.size + 1, goals)
        #         for expr in success_buffer:
        #             replay_buffer.store(expr[0], expr[1], expr[2], expr[3], expr[4])
        #     elif info['status'] == 'failed':
        #         # print('Removing a fail traj')
        #         success_buffer.clear()
        # else:
        #     replay_buffer.store(o, a, r, o2, d)
        self.obs_tmp[self.ptr] = obs
        self.obs2_tmp[self.ptr] = next_obs
        self.act_tmp[self.ptr] = act
        self.rew_tmp[self.ptr] = rew
        self.done_tmp[self.ptr] = done
        self.ptr_tmp = (self.ptr_tmp+1) % self.max_size
        self.size_tmp = min(self.size_tmp+1, self.max_size)


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

