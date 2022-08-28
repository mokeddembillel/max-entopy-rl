
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

class AdamOptim():
    #https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.m_dx, self.v_dx = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr
    
    def step(self, t, x, dx, itr=None): 
        
        dx = dx.view(x.size())
        ''' 
        self.m_dx = self.beta1*self.m_dx + (1-self.beta1)*dx
        self.v_dx = self.beta2*self.v_dx + (1-self.beta2)*(dx**2)
        
        ## bias correction
        m_dx_corr = self.m_dx/(1-self.beta1**t)
        v_dx_corr = self.v_dx/(1-self.beta2**t)
        
        x = x + self.lr*(m_dx_corr/(torch.sqrt(v_dx_corr)+self.epsilon)) 
        '''
        x = x + self.lr * dx
        
        return x