import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
import altair as alt
from altair_saver import save 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import glob, os

alt.data_transformers.enable('default', max_rows=None)

device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
identity_mat = torch.eye(2).to(device)

const = 1

save_folder_path = './ToyExperiments/figs/'
########################### Drawing Utilities ###########################

def get_density_chart(P, d=7.0, step=0.1):
    xv, yv = torch.meshgrid([torch.arange(-d, d, step), torch.arange(-d, d, step)])
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
    p_xy = P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu()

    df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
    df = pd.DataFrame({
        'x': df[:, :, 0].ravel(),
        'y': df[:, :, 1].ravel(),
        'p': df[:, :, 2].ravel(),})

    chart = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    color=alt.Color('p:Q', scale=alt.Scale(scheme='viridis')),
    tooltip=['x','y','p'])

    return chart


def get_particles_chart(X, X_svgd=None):
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],})

    chart = alt.Chart(df).mark_circle(color='red').encode(x='x:Q',y='y:Q')

    if X_svgd is not None:
        #import pdb; pdb.set_trace()
        for i in range(np.shape(X_svgd)[1]):
            df_trajectory = pd.DataFrame({'x': X_svgd[:,i,0],'y': X_svgd[:,i,1],})
            chart += alt.Chart(df_trajectory).mark_line().mark_circle(color='green').encode(x='x:Q',y='y:Q')

    return chart




########################### RBF Kernel version 2 ###########################

class RBF:
    def __init__(self, sigma=None):
        self.sigma = sigma

    def forward(self, input_1, input_2):
        _, out_dim1 = input_1.size()[-2:]
        _, out_dim2 = input_2.size()[-2:]
        num_particles = input_2.size()[-2]
        assert out_dim1 == out_dim2
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)
        
        # Get median.
        if self.sigma is None:
            median_sq = torch.median(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1)[0]
            median_sq = median_sq.unsqueeze(1).unsqueeze(1)
            h = median_sq / (2 * np.log(num_particles + 1.))
            sigma = const * torch.sqrt(h)
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma**2) 
        kappa = (-gamma * dist_sq).exp() 
        
        kappa_grad = -2. * (diff * gamma) * kappa
        return kappa.squeeze(), diff, h, kappa_grad, gamma

################################# SVGD v2 ###############################
class AdamOptim():
    def __init__(self, lr=None):
        self.m_dx, self.v_dx = 0, 0
        self.lr = lr
    
    def step(self,x, dx): 
        dx = dx.view(x.size())
        phi_x = self.lr * dx
        x = x +  phi_x
        return x, phi_x

class SVGD_v2:
    def __init__(self, P, K, optimizer, with_logprob=False):
        self.P = P
        self.K = K
        self.optim = optimizer

        
        self.logp = 0
        self.logp_line1 = 0
        self.logp_line2 = 0
        self.entropy = 0
        self.entropy_line1_ld = 0
        self.entropy_line2_ld = 0
        self.with_logprob = with_logprob
        self.act_limit = 1

        #########
        mu_ld_noise = torch.zeros((dim,)) 
        sigma_ld_noise = torch.eye(dim) * 0.1
        self.init_dist_ld = torch.distributions.MultivariateNormal(mu_ld_noise,covariance_matrix=sigma_ld_noise)

    def phi(self, X):
        X = X.requires_grad_(True)
        log_prob = self.P.log_prob(X)
        # score_func = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0]
        score_func = autograd.grad(log_prob.sum(), X)[0].reshape(X.size())

        self.score_func = score_func.reshape(X.size())
        # self.K_XX, self.K_diff, self.gamma, self.K_grad_v1 = self.K(X, X.detach())
        self.K_XX, self.K_diff, self.K_h, self.K_grad, self.K_gamma = self.K.forward(X, X)        
        # K_grad = -self.K_grad_v1.sum(1)
        self.num_particles=  X.size(0)
        
        # self.phi_term1 = self.K_XX.matmul(score_func.detach())/self.num_particles
        # self.phi_term2 = self.K_grad/self.num_particles
        # phi = self.phi_term1 + self.phi_term2

        self.phi_term1 = self.K_XX.matmul(score_func) / X.size(0)
        self.phi_term2 = self.K_grad.sum(0) / X.size(0)
        phi = self.phi_term1 + self.phi_term2

        # print(self.phi_term1)
        # print(self.phi_term2)
        return phi
    
    def LD(self, X):
        X = X.requires_grad_(True)
        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0].reshape(X.size())
        
        ld = self.optim.lr * score_func 
        ld += (np.sqrt(self.optim.lr) * self.init_dist_ld.sample((len(X),)) ).to(device)
        
        print('tmp1: ', (score_func).mean() )
        #print('tmp2: ', (torch.sqrt(torch.tensor(self.optim.lr)) * torch.normal(torch.tensor(0.), torch.tensor(0.5))).mean() )
        return ld

    def compute_entropy_LD(self, phi, X):
        grad_phi =[]
        for i in range(len(X)):
            grad_phi_tmp = []
            for j in range(X.size(1)):
                grad_ = autograd.grad(phi[i][j], X, retain_graph=True)[0][i].detach()
                grad_phi_tmp.append(grad_)
            grad_phi_tmp = torch.stack(grad_phi_tmp)
            grad_phi.append(grad_phi_tmp)

        ###############################################debugging entropy###################################################
        self.grad_phi = torch.stack(grad_phi) 
        #self.logp_line1 -= torch.log(torch.abs(torch.det(identity_mat.reshape(-1,2,2) + self.optim.lr * self.grad_phi)))
        self.logp_line1 -= torch.log(torch.abs(torch.det(identity_mat.reshape(-1,2,2) + self.grad_phi)))
        self.logp_line2 -= torch.stack([torch.trace(self.grad_phi[i]) for i in range(len(X))])
        
        self.entropy_line1_ld -= self.optim.lr * self.logp_line1.squeeze()
        self.entropy_line2_ld -= self.optim.lr * self.logp_line2.squeeze()


    def compute_entropy(self, phi, X):
        grad_phi =[]
        grad_phi_term1 =[]
        grad_phi_term2 =[]
        
        for i in range(len(X)):
            grad_phi_tmp = []
            grad_phi_term1_tmp = []
            grad_phi_term2_tmp = []
            
            for j in range(X.size(1)):
                grad_ = autograd.grad(phi[i][j], X, retain_graph=True)[0][i].detach()
                grad_phi_tmp.append(grad_)
                grad_phi_term1_tmp.append(autograd.grad(self.phi_term1[i][j], X, retain_graph=True)[0][i])
                grad_phi_term2_tmp.append(autograd.grad(self.phi_term2[i][j], X, retain_graph=True)[0][i] )

            grad_phi_tmp = torch.stack(grad_phi_tmp)
            grad_phi_term1_tmp = torch.stack(grad_phi_term1_tmp)
            grad_phi_term2_tmp = torch.stack(grad_phi_term2_tmp)

            grad_phi.append(grad_phi_tmp)
            grad_phi_term1.append(grad_phi_term1_tmp)
            grad_phi_term2.append(grad_phi_term2_tmp)

        ###############################################debugging entropy####################################################
        grad_phi_term1 = torch.stack(grad_phi_term1)
        grad_phi_term2 = torch.stack(grad_phi_term2)
        self.grad_phi = torch.stack(grad_phi) 
        

        #import pdb; pdb.set_trace()
        #self.logp_line1 -= torch.log(torch.abs(torch.det(identity_mat.reshape(-1,2,2) + self.optim.lr * self.grad_phi)))
        self.logp_line1 -= torch.log(torch.abs(torch.det(identity_mat.reshape(-1,2,2) + self.grad_phi)))
        trace_grad_phi = torch.stack([torch.trace(self.grad_phi[i]) for i in range(len(X))])
        #self.logp_line2 -= self.optim.lr*trace_grad_phi
        self.logp_line2 -= trace_grad_phi

        tr_phi_term1 = torch.stack([torch.trace(grad_phi_term1[i]) for i in range(len(X))]).squeeze()
        tr_phi_term2 = torch.stack([torch.trace(grad_phi_term2[i]) for i in range(len(X))]).squeeze()
        

        # line_4 = (self.K_grad * self.score_func.reshape(1,-1,2)).sum(-1).mean(-1).squeeze()
        line_4 = (self.K_grad * self.score_func.unsqueeze(1)).sum(-1).mean(0)
        # line_5 = -2 * self.K_gamma * ((-self.K_grad * self.K_diff).sum(-1) - 2 * self.K_XX).mean(-1).squeeze()
        line_5 = -2 * self.K_gamma.view(-1,1) * ((self.K_grad * self.K_diff).sum(-1) - X.size(1) * self.K_XX).mean(0)

        
        self.entropy -= self.optim.lr * (line_4+line_5).squeeze()



    def step(self, X, itr):
        # phi_X = self.phi(X) 
        phi_X = self.LD(X) 
        X_new = X + phi_X
        
        #X_new, phi_X_new = self.optim.step(X, phi_X) 

        if self.with_logprob: 
            # self.compute_entropy(phi_X_new, X)
            self.compute_entropy_LD(phi_X, X)
        
        X = X_new.detach()
        return X, phi_X 


################################# Experiment: Unimodal Gaussian ###############################
files = glob.glob(save_folder_path + '*')
[os.remove(file) for file in files]

lr = 0.5
dim = 2
n = 10 

# Initial distribution of SVGD
mu = torch.zeros((dim,)) + 4
sigma_ = 1.
sigma = torch.eye(dim) * sigma_

init_dist = torch.distributions.MultivariateNormal(mu,covariance_matrix=sigma)
X_init = init_dist.sample((n,))


# writer = SummaryWriter('./runs/new/g/'+alg+'/svgd_lr_'+str(svgd_lr)+'/const_'+str(const)+'/n_'+str(n)+'/'+datetime.now().strftime("%b_%d_%Y_%H_%M_%S"))
gauss = torch.distributions.MultivariateNormal(torch.Tensor([-0.6871,0.8010]).to(device),
    covariance_matrix=5 * torch.Tensor([[0.2260,0.1652],[0.1652,0.6779]]).to(device))





##################################

init_dist_entr_gt = init_dist.entropy()
init_dist_entr_by_hand = init_dist.log_prob(X_init)
#init_dist_entr_by_hand = - dim * 0.5 * np.log(2 * np.pi * sigma_) - (0.5 / sigma_) * (X_init ** 2 ).sum(-1)

print('__________________________')
print('init_dist_entr_GT ', init_dist_entr_gt) 
#print('init_dist_entr_mean ', init_dist_entr_mean)
print('init_dist_entr_by_hand ', -init_dist_entr_by_hand.mean()) 
print('__________________________')

################################## 

gauss_chart = get_density_chart(gauss, d=7.0, step=0.1) 
chart = gauss_chart + get_particles_chart(X_init.cpu().numpy())
save(chart,save_folder_path + 'unimodal_gaussian.pdf') 

X = X_init.clone()
K = RBF()
svgd = SVGD_v2(gauss, K, AdamOptim(lr), with_logprob=True) 

X_svgd_=[]
for t in tqdm(range(300)): 
    X, phi_X = svgd.step(X, t)
    X_svgd_.append(X.clone())
    
    #print(X_svgd_)

    if (t%40)==0: 
        chart = gauss_chart + get_particles_chart(X.detach().cpu().numpy())
        chart_ = gauss_chart + get_particles_chart(X.detach().cpu().numpy(), torch.stack(X_svgd_).detach().cpu().numpy())
        save(alt.hconcat(chart, chart_),save_folder_path + 'unimodal_gaussian_svgd_sig_'+str(const)+ '_iteration_' + str(t) +'.pdf') 
    # writer.add_scalar('stein_identity', phi_X.mean(), t) 
    # writer.add_scalar('entropy/total_line1',  -(init_dist.log_prob(X_init) + svgd.logp_line1).mean(), t) 
    # writer.add_scalar('entropy/total_line2',  -(init_dist.log_prob(X_init) + svgd.logp_line2).mean(), t) 
    # writer.add_scalar('entropy/total_final',  -svgd.entropy.mean(), t) 

    # writer.add_scalar('entropy/svgd_line1',  -(svgd.logp_line1).mean(), t) 
    # writer.add_scalar('entropy/svgd_line2',  -(svgd.logp_line2).mean(), t)

    # #import pdb; pdb.set_trace()
    # writer.add_scalar('phi/phi_term1',  (svgd.phi_term1).mean(), t)  
    # writer.add_scalar('phi/phi_term2',  (svgd.phi_term2).mean(), t)  
    # writer.add_scalar('phi/phi_total',  (phi_X).mean(), t)

    # #import pdb; pdb.set_trace()
    # writer.add_scalar('entropy_intermediate/grad_phi' , svgd.grad_phi.mean(), t) 
    # writer.add_scalar('entropy_intermediate/det(I+grad_phi)' ,(torch.det(identity_mat.reshape(-1,2,2) + svgd.grad_phi)).mean(), t)
    # writer.add_scalar('entropy_intermediate/abs(det(I+grad_phi))' ,torch.abs(torch.det(identity_mat.reshape(-1,2,2) + svgd.grad_phi)).mean(), t)
    # writer.add_scalar('entropy_intermediate/log(abs(det(I+grad_phi)))' ,torch.log(torch.abs(torch.det(identity_mat.reshape(-1,2,2) + svgd.grad_phi))).mean(), t)

'''
X_svgd_ = torch.stack(X_svgd_)

chart = gauss_chart + get_particles_chart(X.detach().cpu().numpy())
chart_ = gauss_chart + get_particles_chart(X.detach().cpu().numpy(), X_svgd_.detach().cpu().numpy())
save(alt.hconcat(chart, chart_),save_folder_path + 'unimodal_gaussian_svgd_sig_'+str(const)+'.pdf') 
'''

print('___________Gaussian____________')
print('stein at convergence: ', phi_X.mean() )
print(' ')

# print('logp_gt: ', gauss.entropy())
#print('logp_svgd: ',  init_dist_entr_by_hand + svgd.logp)

print(' ')
print('logp_gt: ', -gauss.log_prob(X).mean() )
print('entropy gt: ', gauss.entropy() )  
print('entropy svgd (line 1): ',  -(init_dist.log_prob(X_init) + svgd.logp_line1).mean())
print(' ')
print('Initial distribution log_prob :', -init_dist.log_prob(X_init).mean().item())
print(' ')
#print('entropy svgd (line 2): ',  -(init_dist.log_prob(X_init) + svgd.logp_line2).mean())

# writer.add_scalar('entropy/gt',  gauss.entropy() , t) 
# writer.add_scalar('entropy/diff',  ((gauss.log_prob(X)-(init_dist.log_prob(X_init) + svgd.logp_line1))**2).mean() , t) 


#print('__entropy svgd q0: ',  init_dist.log_prob(X_init).mean())
#print('__entropy svgd part 2 (line 1): ',  (svgd.logp_line1).mean())
#print('__entropy svgd part 2 (line 2): ',  (svgd.logp_line2).mean())


################################# Experiment: Mixture of 2 Gaussians ###############################

# class MoG(torch.distributions.Distribution):
#     def __init__(self, loc, covariance_matrix):
#         self.num_components = loc.size(0)
#         self.loc = loc
#         self.covariance_matrix = covariance_matrix

#         self.dists = [
#           torch.distributions.MultivariateNormal(mu, covariance_matrix=sigma)
#           for mu, sigma in zip(loc, covariance_matrix)
#         ]
#         super(MoG, self).__init__(torch.Size([]), torch.Size([loc.size(-1)]))

#     @property
#     def arg_constraints(self):
#         return self.dists[0].arg_constraints

#     @property
#     def support(self):
#         return self.dists[0].support

#     @property
#     def has_rsample(self):
#         return False

#     def log_prob(self, value):
#         return torch.cat([p.log_prob(value).unsqueeze(-1) for p in self.dists], dim=-1).logsumexp(dim=-1)

#     def enumerate_support(self):
#         return self.dists[0].enumerate_support()


# class MoG2(MoG):
#     def __init__(self, device=None):
#         loc = torch.Tensor([[-5.0, 0.0], [5.0, 0.0]]).to(device)
#         cov = torch.Tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(2, 1, 1).to(device)

#         super(MoG2, self).__init__(loc, cov)


# mog2 = MoG2(device=device)

# alg ="gd" #"adam"
# svgd_lr = 1#e-1
# n = 100

# print('n: ',n)
# writer = SummaryWriter('./runs/new/gmm2/'+alg+'/n_'+str(n)+'/svgd_lr_'+str(svgd_lr)+'/const_'+str(const)+'/'+datetime.now().strftime("%b_%d_%Y_%H_%M_%S"))

# X_init = (5 * torch.randn(n, *mog2.event_shape)).to(device)

# mog2_chart = get_density_chart(mog2, d=7.0, step=0.1)


# # try K_v2 and svgd_v2
# X = X_init.clone()
# K_v2 = RBF_v2()
# svgd = SVGD_v2(mog2, K_v2, AdamOptim(alg=alg,lr=1e-1), with_logprob=True, kernel_type=2)

# X_svgd_=[]
# for t in tqdm(range(500)): 
#     X, phi_X = svgd.step(X, t)
#     X_svgd_.append(X.clone())
#     writer.add_scalar('stein_identity', phi_X.mean(), t) 
#     writer.add_scalar('entropy/total_line1',  -(init_dist.log_prob(X_init) + svgd.logp_line1).mean(), t) 
#     writer.add_scalar('entropy/total_line2',  -(init_dist.log_prob(X_init) + svgd.logp_line2).mean(), t) 
#     writer.add_scalar('entropy/total_final',  -svgd.entropy.mean(), t) 

#     writer.add_scalar('entropy/svgd_line1',  -(svgd.logp_line1).mean(), t) 
#     writer.add_scalar('entropy/svgd_line2',  -(svgd.logp_line2).mean(), t) 
#     #import pdb; pdb.set_trace()

#     writer.add_scalar('phi/phi_term1',  (svgd.phi_term1).mean(), t)  
#     writer.add_scalar('phi/phi_term2',  (svgd.phi_term2).mean(), t)  
#     writer.add_scalar('phi/phi_total',  (phi_X).mean(), t)

#     writer.add_scalar('entropy_intermediate/grad_phi' , svgd.grad_phi.mean(), t) 
#     writer.add_scalar('entropy_intermediate/det(I+grad_phi)' ,(torch.det(identity_mat.reshape(-1,2,2) + svgd.grad_phi)).mean(), t)
#     writer.add_scalar('entropy_intermediate/abs(det(I+grad_phi))' ,torch.abs(torch.det(identity_mat.reshape(-1,2,2) + svgd.grad_phi)).mean(), t)
#     writer.add_scalar('entropy_intermediate/log(abs(det(I+grad_phi)))' ,torch.log(torch.abs(torch.det(identity_mat.reshape(-1,2,2) + svgd.grad_phi))).mean(), t)



# X_svgd_ = torch.stack(X_svgd_)


# chart = mog2_chart + get_particles_chart(X.detach().cpu().numpy())
# chart_ = mog2_chart + get_particles_chart(X.detach().cpu().numpy(), X_svgd_.detach().cpu().numpy())
# save(alt.hconcat(chart, chart_),'gmm2_svgd_sig_'+str(const)+'.pdf') 


# print('___________GMM 2____________')
# print('stein at convergence: ', phi_X.mean() )
# print(' ')

# writer.add_scalar('entropy/gt',  -mog2.log_prob(X).mean(), t) 
# writer.add_scalar('entropy/diff',  ((mog2.log_prob(X)-(init_dist.log_prob(X_init) + svgd.logp_line1))**2).mean() , t) 

# print('entropy gt: ', -mog2.log_prob(X).mean() )  
# print('entropy svgd (line 1): ',  -(init_dist.log_prob(X_init) + svgd.logp_line1).mean())
# print('entropy svgd (line 2): ',  -(init_dist.log_prob(X_init) + svgd.logp_line2).mean())
# print(' ')
# print('__entropy svgd q0: ',  init_dist.log_prob(X_init).mean())
# print('__entropy svgd part 2 (line 1): ',  (svgd.logp_line1).mean())
# print('__entropy svgd part 2 (line 2): ',  (svgd.logp_line2).mean())


# ################################# Experiment: Mixture of 6 Gaussians ###############################


# class MoG6(MoG):
#     def __init__(self, device=None):
#         def _compute_mu(i):
#             return 5.0 * torch.Tensor([[torch.tensor(i * math.pi / 3.0).sin(),torch.tensor(i * math.pi / 3.0).cos()]])

#         loc = torch.cat([_compute_mu(i) for i in range(1, 7)], dim=0).to(device)
#         cov = torch.Tensor([0.5, 0.5]).diag().unsqueeze(0).to(device).repeat(6, 1, 1) 

#         super(MoG6, self).__init__(loc, cov)

# mog6 = MoG6(device=device)

# n = 100
# alg ="gd" #"adam"
# svgd_lr = 1#e-1
# n = 100 



# writer = SummaryWriter('./runs/new/gmm6/'+alg+'/n_'+str(n)+'/svgd_lr_'+str(svgd_lr)+'/const_'+str(const)+'/'+datetime.now().strftime("%b_%d_%Y_%H_%M_%S"))


# X_init = (5 * torch.randn(n, *mog6.event_shape)).to(device)


# mog6_chart = get_density_chart(mog6, d=7.0, step=0.1)


# # try K_v2 and svgd_v2
# X = X_init.clone()
# K_v2 = RBF_v2()
# svgd = SVGD_v2(mog6, K_v2, AdamOptim(alg=alg,lr=1e-1), with_logprob=True, kernel_type=2)

# for t in tqdm(range(500)): 
#     X, phi_X = svgd.step(X, t)
#     writer.add_scalar('stein_identity', phi_X.mean(), t) 
#     writer.add_scalar('entropy/total_line1',  -(init_dist.log_prob(X_init) + svgd.logp_line1).mean(), t) 
#     writer.add_scalar('entropy/total_line2',  -(init_dist.log_prob(X_init) + svgd.logp_line2).mean(), t) 
#     writer.add_scalar('entropy/total_final',  -svgd.entropy.mean(), t) 

#     writer.add_scalar('entropy/svgd_line1',  -(svgd.logp_line1).mean(), t) 
#     writer.add_scalar('entropy/svgd_line2',  -(svgd.logp_line2).mean(), t) 
#     #import pdb; pdb.set_trace()
#     writer.add_scalar('entropy_intermediate/grad_phi' , svgd.grad_phi.mean(), t) 
#     writer.add_scalar('entropy_intermediate/det(I+grad_phi)' ,(torch.det(identity_mat.reshape(-1,2,2) + svgd.grad_phi)).mean(), t)
#     writer.add_scalar('entropy_intermediate/abs(det(I+grad_phi))' ,torch.abs(torch.det(identity_mat.reshape(-1,2,2) + svgd.grad_phi)).mean(), t)
#     writer.add_scalar('entropy_intermediate/log(abs(det(I+grad_phi)))' ,torch.log(torch.abs(torch.det(identity_mat.reshape(-1,2,2) + svgd.grad_phi))).mean(), t)




# chart = (mog6_chart + get_particles_chart(X_init.cpu().numpy())) | (mog6_chart + get_particles_chart(X.detach().cpu().numpy()))
# save(chart,'MoG6_sig_'+str(const)+'.pdf')  


# print('___________GMM 6____________')
# print('stein at convergence: ', phi_X.mean() )
# #print(' ')
# #print('logp_gt: ', mog6.log_prob(X) )
# #print('logp_svgd: ', init_dist_entr_by_hand + svgd.logp)
# writer.add_scalar('entropy/gt',  -mog6.log_prob(X).mean(), t) 
# writer.add_scalar('entropy/diff',  ((mog6.log_prob(X)-(init_dist.log_prob(X_init) + svgd.logp_line1))**2).mean() , t) 

# print('entropy gt: ', -mog6.log_prob(X).mean() )  
# print('entropy svgd (line 1): ',  -(init_dist.log_prob(X_init) + svgd.logp_line1).mean())
# print('entropy svgd (line 2): ',  -(init_dist.log_prob(X_init) + svgd.logp_line2).mean())
# print(' ')
# print('__entropy svgd q0: ',  init_dist.log_prob(X_init).mean())
# print('__entropy svgd part 2 (line 1): ',  (svgd.logp_line1).mean())
# print('__entropy svgd part 2 (line 2): ',  (svgd.logp_line2).mean())

