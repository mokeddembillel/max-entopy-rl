

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, input_1, input_2,  h_min=1e-3):
        k_fix, out_dim1 = input_1.size()[-2:]
        k_upd, out_dim2 = input_2.size()[-2:]
        num_particles = input_2.size()[-2]
        assert out_dim1 == out_dim2
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)
        
        # Get median.
        median_sq = torch.median(dist_sq.detach().reshape(-1, num_particles*num_particles), dim=1)[0]
        median_sq = median_sq.reshape(-1,1,1,1)
        
        h = median_sq / (2 * np.log(num_particles + 1.))
        sigma = torch.sqrt(h)
        gamma = 1.0 / (1e-8 + 2 * sigma**2) 
        #

        kappa = (-gamma * dist_sq).exp()
        kappa_grad = -2. * (diff * gamma) * kappa
        return kappa.squeeze(-1), diff, gamma, kappa_grad

class SVGD_ACTOR():
    def __init__(self):
        self.K = RBF()

    def sampler(self, obs, a, with_logprob=True, num_svgd_particles=None, plot=False, itr=None, writer=None):
        svgd_optim = AdamOptim(lr=self.svgd_lr)
        logp = 0
        #print('_____________svgd_sampler___________')

        if plot:
            self.svgd_steps = []
            self.score_func_list =[]
            self.hess_list =[]
            self.hess_eig_max =[]

        if num_svgd_particles is None:
            num_svgd_particles = self.num_svgd_particles


        def phi(X):
            nonlocal logp
            X = X.requires_grad_(True)
            log_prob = self.q1(obs, X)
            score_func = autograd.grad(log_prob.sum(), X, retain_graph=True, create_graph=True)[0]
            
            if plot:
                hess_0 = torch.autograd.grad(score_func[:,0].sum(),X,retain_graph=True)[0]
                hess_1 = torch.autograd.grad(score_func[:,1].sum(),X,retain_graph=True)[0]
                hess = torch.stack([hess_0,hess_1]).permute(1,0,2)

                self.hess_list.append(hess_0.detach())
                self.hess_list.append(hess_1.detach())
                self.score_func_list.append(score_func.detach())
                
                #
                for z in range(len(hess)):
                    self.hess_eig_max.append( torch.max(torch.abs(torch.eig(hess[z])[0][:,0])) ) 

            X = X.reshape(-1, num_svgd_particles, self.act_dim)
            score_func = score_func.reshape(X.size())
            K_XX, K_diff, K_gamma, K_grad = self.K(X, X.detach())
            phi = (K_XX.detach().matmul(score_func) - K_grad.sum(2)) / num_svgd_particles 
            
            # compute the entropy
            if with_logprob:
                #import pdb; pdb.set_trace()
                line_4 = (K_grad * score_func.reshape(-1,1,num_svgd_particles,self.act_dim)).sum(-1).mean(-1)
                line_5 = -2 * K_gamma.view(-1,1) * ((-K_grad * K_diff).sum(-1) - self.act_dim * K_XX).mean(-1)
                logp -= self.svgd_lr*(line_4+line_5)
            
            return phi 

        
        for t in range(self.num_svgd_steps):
            #print('____t: ', t)
            if plot:
                self.svgd_steps.append(a.detach())
            
            a = svgd_optim.step(t+1, a, phi(a), itr)
            a = torch.clamp(a, -self.act_limit, self.act_limit).detach()
        
        if plot:
            #print('stacking')
            self.svgd_steps.append(a.detach())  
            self.score_func_list = torch.stack(self.score_func_list)
            self.hess_list = torch.stack(self.hess_list)
            self.hess_eig_max = torch.stack(self.hess_eig_max)
        
        #if itr is not None:   
        #    plot = False
        #    writer.add_histogram('stein_identity', phi(a).mean(), itr)
        plot = False 
        return a, logp, phi(a)  

    def act():
        obs = obs.view(-1,1,obs.size()[-1]).repeat(1,self.num_svgd_particles,1).view(-1,obs.size()[-1])

        with torch.no_grad():
            a = torch.normal(0, 1, size=(b_size * self.num_svgd_particles, self.act_dim)).to(self.device)
            a = self.act_limit * torch.tanh(a)


        #if Version_1 
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
        
        #if Version_2
        a2, logp0_a2 = ac.pi(o2)
        a2, logq_a2,_= ac.svgd_sampler(o2, a2.detach())
        a2 = a2.detach()

        logp_a2 = logp0_a2 + logq_a2.mean(-1)
        logp_a2 = logp_a2.detach()







