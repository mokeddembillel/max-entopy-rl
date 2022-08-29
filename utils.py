
import numpy as np

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    num_var = 0
    
    for p in module.parameters():
        num_var += np.prod(p.shape)

    return num_var
