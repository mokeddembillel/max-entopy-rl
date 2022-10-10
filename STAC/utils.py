
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

class GMMDist(object):
    def __init__(self, dim, n_gmm, device):
        def _compute_mu(i):
            return 4.0 * torch.Tensor([[torch.tensor(i * math.pi / (n_gmm//2)).sin(),torch.tensor(i * math.pi / (n_gmm//2)).cos()]])
        
        self.mix_probs = 0.25 * torch.ones(n_gmm).to(device)
        # self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        # self.mix_probs = torch.tensor([0.1, 0.1, 0.8])
        # self.means = torch.stack([5 * torch.ones(dim), torch.zeros(dim), -torch.ones(dim) * 5], dim=0)
        self.means = torch.cat([_compute_mu(i) for i in range(n_gmm)], dim=0).to(device)
        #self.means = torch.stack([5 * torch.ones(dim).to(device), -torch.ones(dim).to(device) * 5], dim=0)
        self.sigma = 1.0
        self.std = torch.stack([torch.ones(dim).to(device) * self.sigma for i in range(len(self.mix_probs))], dim=0)

    def sample(self, n):
        n = n[0]
        mix_idx = torch.multinomial(self.mix_probs, n, replacement=True)
        means = self.means[mix_idx]
        stds = self.std[mix_idx]
        return torch.randn_like(means) * stds + means

    def log_prob(self, samples):
        logps = []
        for i in range(len(self.mix_probs)):
            logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * self.sigma ** 2) - 0.5 * np.log(
                2 * np.pi * self.sigma ** 2)) + self.mix_probs[i].log())
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp