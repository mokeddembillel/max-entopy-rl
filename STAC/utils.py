
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