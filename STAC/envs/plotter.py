import numpy as np
import matplotlib.pyplot as plt
import torch as T


def svgd_plotting(ax, path_svgd_steps, xx, yy, add_obs=False):
    
    svgd_chain = T.stack(path_svgd_steps).cpu().numpy()

    if len(svgd_chain.shape)==3:
        num_svgd_steps, num_svgd_particles, _ = svgd_chain.shape
        num_epised_steps = 1
        xx = [xx]
        yy = [yy]
        svgd_chain = np.expand_dims(svgd_chain,0)
    else:
        num_epised_steps, num_svgd_steps, num_svgd_particles, _ = svgd_chain.shape

    

    for k in range(num_epised_steps):
        #print(k)
        for i in range(num_svgd_particles):
            for h in range(num_svgd_steps):
                svgd_chain_xx = svgd_chain[k,h,i,0]#+xx[k] 
                svgd_chain_yy = svgd_chain[k,h,i,1]#+yy[k]

                if add_obs:
                   svgd_chain_xx += xx[k] 
                   svgd_chain_yy += yy[k]

                #print('step: ', h,' particle: ',i ,'x: ',svgd_chain_xx, ' y: ', svgd_chain_yy)
                #import pdb; pdb.set_trace()
                #ax.annotate(str(h+1), (svgd_chain_xx, svgd_chain_yy), xytext=(10,10), textcoords='offset points')
                ax.plot(svgd_chain_xx, svgd_chain_yy, marker="o", markersize=2, color='green') 

    #import pdb; pdb.set_trace()


class QFPolicyPlotter:
    def __init__(self, qf, policy, obs_lst, eps, default_action, n_samples, epoch, alpha, batch_size, sac_version, svgd_steps, svgd_particles, svgd_lr, device):

        self._svgd_steps = svgd_steps
        self._svgd_particles = svgd_particles
        self._svgd_lr = svgd_lr
        self._qf = qf
        self._policy = policy
        self._obs_lst = obs_lst
        self._default_action = default_action
        self._n_samples = n_samples
        self._epoch = epoch
        self._alpha = alpha
        self._batch_size = batch_size
        self._sac_version = sac_version
        self._device = device
        self._eps = eps

        self._var_inds = np.where(np.isnan(default_action))[0]
        assert len(self._var_inds) == 2


    def draw(self):
        n_plots = len(self._obs_lst)

        x_size = 5 * n_plots
        y_size = 5

        self.fig = plt.figure(figsize=(x_size, y_size))
        self._ax_lst = []
        for i in range(n_plots):
            ax = self.fig.add_subplot(100 + n_plots * 10 + i + 1)
            ax.set_xlim((-2, 2))
            ax.set_ylim((-2, 2))
            ax.grid(True)
            self._ax_lst.append(ax)

        self._line_objects = list()
        # noinspection PyArgumentList
        [h.remove() for h in self._line_objects]
        self._line_objects = list()

        self._plot_level_curves()

        self._plot_action_samples() 

        #plt.draw()
        #plt.pause(0.001)
        plt.title("Epoch " + str(self._epoch) + " " + str(self._eps))
        plt.savefig("./multigoal_plots_/state"+self._sac_version+"_svgd_steps_" + str(self._svgd_steps) +"_svgd_particles_" + str(self._svgd_particles) +"_svgd_lr_" + str(self._svgd_lr)+"_state_plots_n_samples_"+ str(self._n_samples)+"_"+ str(self._epoch)+ "_alpha_"+str(self._alpha)+"_batch_size_"+str(self._batch_size)+".png" )
        plt.close() 

    def _plot_level_curves(self):
        # Create mesh grid.
        xs = np.linspace(-2, 2, 200)
        ys = np.linspace(-2, 2, 200)
        xgrid, ygrid = np.meshgrid(xs, ys)
        N = len(xs)*len(ys)

        # Copy default values along the first axis and replace nans with
        # the mesh grid points.
        actions = np.tile(self._default_action, (N, 1))
        actions[:, self._var_inds[0]] = xgrid.ravel()
        actions[:, self._var_inds[1]] = ygrid.ravel()
        actions = T.from_numpy(actions.astype(np.float32)).to(self._device)

        for ax, obs in zip(self._ax_lst, self._obs_lst):
            obs = T.FloatTensor(obs).repeat([actions.shape[0],1]).to(self._device)
            with T.no_grad():
                #import pdb; pdb.set_trace()
                qs = self._qf(obs, actions).cpu().detach().numpy()

            qs = qs.reshape(xgrid.shape)

            cs = ax.contour(xgrid, ygrid, qs, 40)
            self._line_objects += cs.collections
            self._line_objects += ax.clabel(cs, inline=1, fontsize=10, fmt='%.2f')


    

    def _plot_action_samples(self):
        for ax, obs in zip(self._ax_lst, self._obs_lst):
            #print('obs: ', obs)
            #with T.no_grad():
            if (self._n_samples == 1): 
                #
                #print('________________________obs ', obs)
                actions = self._policy.act(T.FloatTensor(obs).to(self._device).repeat([self._n_samples,1]), test=True ,plot=True)
                path = self._policy.svgd_steps
                svgd_plotting(ax, path, obs[0], obs[1])
                x, y = actions[0], actions[1]
                
                #print('best x , y: ', x, ' ', y)
                #print('best: ', x+obs[0], ' ', y+obs[1] ) 
                #import pdb; pdb.set_trace()
            else:
                actions = self._policy.act(T.FloatTensor(obs).to(self._device).repeat([self._n_samples,1]), plot=True)
                x, y = actions[:, 0], actions[:, 1]

            ax.title.set_text(str(obs))
            #print('orig: ', x, ' ', y ) 
            self._line_objects += ax.plot(x, y, 'r*')
            #import pdb; pdb.set_trace() 


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

