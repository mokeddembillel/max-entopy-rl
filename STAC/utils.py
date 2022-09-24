
import numpy as np
import torch as torch

def count_vars(module):
    num_var = 0
    
    for p in module.parameters():
        num_var += np.prod(p.shape)

    return num_var

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def gaussian(x, mu, sig):
    out = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    out = np.tanh(out)
    return out 

def moving_average(data, window):
    window = 5
    average_y = []
    for ind in range(len(data) - window + 1):
        average_y.append(np.mean(data[ind:ind+window]))
    return np.array(average_y)