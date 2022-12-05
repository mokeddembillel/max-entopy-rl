import numpy as np 
import torch as torch
import matplotlib.pyplot as plt
from utils import gaussian
from tqdm import tqdm
import matplotlib.colors as mpl_colors

class Debugger():
    def __init__(self, tb_logger, ac, env_name, train_env, test_env, plot_format):
        # Still need some improvements that i will do tomorrow
        self.ac = ac
        self.tb_logger = tb_logger
        self.env_name = env_name
        self.train_env = train_env
        self.test_env = test_env
        self.episodes_information = []
        self.episode_counter = 0
        self.colors = ['red', 'orange', 'purple']
        self.episodes_information_svgd = []
        self.plot_format = plot_format

    def collect_data(self, o, a, o2, r, d, log_p):
        
        if self.test_env.ep_len == 1:
            self.episodes_information.append({
                'observations':[],
                'action': [],
                'actions': [],
                # Debugging ##########
                'log_p': [],
                'term1': [],
                'term2': [],
                'logp_normal': [],
                'logp_svgd': [],
                'logp_tanh': [],
                # 'logp_toy_line1': [],
                # 'logp_toy_line2': [],
                # 'logp_toy_line4': [],
                # 'logp_wrong': [],
                ######################
                'rewards': [],
                'expected_reward': None, 
                'episode_length': None,
                # p_0
                'mu': [],
                'sigma': [],
                # scores
                'q_score': [],
                'q_score_start': None, 
                'q_score_mid': None, 
                'q_score_end': None, 
                # hessian
                'q_hess' : [],
                'q_hess_start': None, 
                'q_hess_mid': None, 
                'q_hess_end': None, 
                })

        self.episodes_information[-1]['observations'].append(o.detach().cpu().numpy().squeeze())

        if self.ac.pi.actor != 'sac':
            self.episodes_information[-1]['actions'].append(self.ac.pi.a.detach().cpu().numpy().squeeze())
        self.episodes_information[-1]['action'].append(a.detach().cpu().numpy().squeeze())
        self.episodes_information[-1]['rewards'].append(r)

        a.requires_grad = True
        q1_value = self.ac.q1(o,a)
        q2_value = self.ac.q2(o,a)

        # if self.ac.pi.actor in ['svgd_nonparam']:
        #     self.episodes_information[-1]['log_p'].append(-log_p.detach().item())
        #     self.episodes_information[-1]['term1'].append(self.ac.pi.term1_debug.detach().cpu())
        #     self.episodes_information[-1]['term2'].append(self.ac.pi.term2_debug.detach().cpu())
        #     self.episodes_information[-1]['logp_normal'].append(self.ac.pi.logp_normal_debug.detach().item())
        #     self.episodes_information[-1]['logp_svgd'].append(self.ac.pi.logp_svgd_debug.detach().item())
        #     self.episodes_information[-1]['logp_tanh'].append(self.ac.pi.logp_tanh_debug.detach().item())
            # self.episodes_information[-1]['logp_toy_line1'].append(self.ac.pi.logp_line1.mean().cpu().detach().item())
            # self.episodes_information[-1]['logp_toy_line2'].append(self.ac.pi.logp_line2.mean().cpu().detach().item())
            # self.episodes_information[-1]['logp_toy_line4'].append(self.ac.pi.logp_line4.mean().cpu().detach().item())
            # self.episodes_information[-1]['logp_wrong'].append(self.ac.pi.logp_wrong.mean().cpu().detach().item())


        if self.ac.pi.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            self.episodes_information[-1]['mu'].append(self.ac.pi.mu.detach().cpu().numpy())
            self.episodes_information[-1]['sigma'].append(self.ac.pi.sigma.detach().cpu().numpy())
        
        # print('***********************',len( self.episodes_information[-1]['observations'] ),'********', len(self.episodes_information[-1]['mu']))
        
        # grad_q_ = torch.autograd.grad(torch.min(q1_value, q2_value), a, retain_graph=True, create_graph=True)[0].squeeze()
        # hess_q = ((torch.abs(torch.autograd.grad(grad_q_[0], a, retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_q_[1], a, retain_graph=True)[0])).sum()/4)
        # self.episodes_information[-1]['q_score'].append(torch.abs(grad_q_).mean().detach().cpu().item())
        # self.episodes_information[-1]['q_hess'].append(hess_q.detach().cpu().item())

        # self.episodes_information[-1]['q1_values'] = q1_value.detach().cpu().numpy()
        # self.episodes_information[-1]['q2_values'] = q2_value.detach().cpu().numpy()
        
        if (self.test_env.ep_len >= self.test_env.max_steps) or d: 
            self.episodes_information[-1]['observations'].append(o2.squeeze())
            self.episodes_information[-1]['expected_reward'] = np.sum(self.episodes_information[-1]['rewards'])
            self.episodes_information[-1]['episode_length'] = self.test_env.ep_len
            
            # if self.test_env.ep_len >= 5:
            #     self.episodes_information[-1]['q_score_start'] = np.mean(self.episodes_information[-1]['q_score'][:5])
            #     self.episodes_information[-1]['q_hess_start'] = np.mean(self.episodes_information[-1]['q_hess'][:5])
            # if self.test_env.ep_len >= 17:
            #     self.episodes_information[-1]['q_score_mid'] = np.mean(self.episodes_information[-1]['q_score'][12:17])
            #     self.episodes_information[-1]['q_hess_mid'] = np.mean(self.episodes_information[-1]['q_hess'][12:17])
            # if self.test_env.ep_len >= 30:
            #     self.episodes_information[-1]['q_score_end'] = np.mean(self.episodes_information[-1]['q_score'][25:self.test_env.ep_len])
            #     self.episodes_information[-1]['q_hess_end'] = np.mean(self.episodes_information[-1]['q_hess'][25:self.test_env.ep_len])
    
    def create_entropy_plots(self, itr):
        if self.env_name in ['Multigoal', 'multigoal-max-entropy'] and (itr + 1)%6800 == 0:
        # if 1:
            for s in range(self.test_env.max_steps):
                feed_dict = {}
                for i in range(len(self.episodes_information)):
                    if len(self.episodes_information[i]['log_p']) > s:
                        feed_dict['episode_' + str(i)] = self.episodes_information[i]['log_p'][s]
                self.add_scalars('Entropy/episode_entropy_' + str(itr),feed_dict, s)
    
    def collect_svgd_data(self, exploration, observation, particles=None, logp=None):
        if self.train_env.ep_len == 0:
            self.episodes_information_svgd.append({
                'step': [], 
                'exploration': [],
                'entropy': [], 
                'observations': [], 
                'particles': [],
                'gradients': [],
                'svgd_lr': [],
            })
        

        self.episodes_information_svgd[-1]['step'].append(self.train_env.ep_len)
        self.episodes_information_svgd[-1]['observations'].append(list(observation))
        self.episodes_information_svgd[-1]['exploration'].append(exploration)
        if not exploration:
            self.episodes_information_svgd[-1]['entropy'].append(-logp.detach().cpu().item())
            self.episodes_information_svgd[-1]['particles'].append(self.ac.pi.x_t)
            self.episodes_information_svgd[-1]['gradients'].append(self.ac.pi.phis)
            self.episodes_information_svgd[-1]['svgd_lr'].append(self.ac.pi.svgd_lr)
        else:
            self.episodes_information_svgd[-1]['particles'].append(list(particles))

        # print(self.episodes_information_svgd[-1]['exploration'])

    def plot_svgd_particles_q_contours(self, fig_path):
        self._ax_lst = []
        _n_samples = 100
        _obs_lst = self.episodes_information_svgd[-1]['observations']
        # for i in range(len(_obs_lst)):
        for episode_step in range(1):

            self.fig_env = plt.figure(figsize=(4, 4), constrained_layout=True) 
            self._ax_lst.append(plt.subplot2grid((1,1), (0,0), colspan=3, rowspan=3))
            self._ax_lst[0].set_xlim((-1, 1))
            self._ax_lst[0].set_ylim((-1, 1))
            self._ax_lst[0].set_title('SVGD Particles Plot')
            self._ax_lst[0].set_xlabel('x')
            self._ax_lst[0].set_ylabel('y')
            self._ax_lst[0].grid(True)
            self._line_objects = []


            xs = np.linspace(-1, 1, 50)
            ys = np.linspace(-1, 1, 50)
            xgrid, ygrid = np.meshgrid(xs, ys)
            a = np.concatenate((np.expand_dims(xgrid.ravel(), -1), np.expand_dims(ygrid.ravel(), -1)), -1)
            a = torch.from_numpy(a.astype(np.float32)).to(self.ac.pi.device)
            o = torch.Tensor(_obs_lst[episode_step]).repeat([a.shape[0],1]).to(self.ac.pi.device)
            with torch.no_grad():
                qs = self.ac.q1(o.to(self.ac.pi.device), a).cpu().detach().numpy()
            qs = qs.reshape(xgrid.shape)
            cs = self._ax_lst[0].contour(xgrid, ygrid, qs, 20)
            self._line_objects += cs.collections
            self._line_objects += self._ax_lst[0].clabel(
                cs, inline=1, fontsize=10, fmt='%.2f')

            o = _obs_lst[episode_step]
            actions = np.array(self.episodes_information_svgd[-1]['particles'][episode_step])
            entropy = self.episodes_information_svgd[-1]['entropy'][episode_step]
            # actions = actions.cpu().detach().numpy().squeeze()
            # if self.episodes_information_svgd[-1]['exploration'][episode_step]:
            # else:
            no_of_colors=10
            # colors = ["#"+''.join([np.random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(no_of_colors)]
            for particle_idx in range(actions.shape[1]):
                x, y = actions[:, particle_idx, 0], actions[:, particle_idx, 1]
                color = (0.99, 0.5, np.random.random())
                self._ax_lst[0].title.set_text(str([round(_obs_lst[episode_step][0], 2), round(_obs_lst[episode_step][1], 2)]) + ' -- Entropy: ' + str(round(entropy, 2)))
                if self.episodes_information_svgd[-1]['exploration'][episode_step]:
                    self._line_objects += self._ax_lst[0].plot(x, y, c=color + '*')
                else:
                    self._line_objects += self._ax_lst[0].plot(x, y, c=color)
            plt.savefig(fig_path+ '/svgd_episode_' + str(episode_step) + '_step_' + str(self.episodes_information_svgd[-1]['step'][episode_step]) + '.' + self.plot_format)

    def entropy_plot(self):
        if self.env_name in ['Multigoal', 'multigoal-max-entropy']:
            log_p = []
            term1 = []
            term2 = []
            logp_normal = []
            logp_svgd = []
            logp_tanh = []
            # logp_toy_line1 = []
            # logp_toy_line2 = []
            # logp_toy_line4 = []
            # logp_wrong = []
            for indx, i in enumerate([0, 10, 25]):
                if len(self.episodes_information[-1]['log_p']) > i+1:
                    log_p.append(self.episodes_information[-1]['log_p'][i])
                    term1.append(self.episodes_information[-1]['term1'][i])
                    term2.append(self.episodes_information[-1]['term2'][i])
                    logp_normal.append(self.episodes_information[-1]['logp_normal'][i])
                    logp_svgd.append(self.episodes_information[-1]['logp_svgd'][i])
                    logp_tanh.append(self.episodes_information[-1]['logp_tanh'][i])
                    # logp_toy_line1.append(self.episodes_information[-1]['logp_toy_line1'][i])
                    # logp_toy_line2.append(self.episodes_information[-1]['logp_toy_line2'][i])
                    # logp_toy_line4.append(self.episodes_information[-1]['logp_toy_line4'][i])
                    # logp_wrong.append(self.episodes_information[-1]['logp_wrong'][i])
            if len(log_p) == 3:
                # print('############################# ', (len(self.episodes_information)- 1))
                self.add_scalars(tb_path='Entropy/entropy', value={'step_0': log_p[0], 'step_10': log_p[1], 'step_25': log_p[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/term1', value={'step_0': term1[0], 'step_10': term1[1], 'step_25': term1[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/term2', value={'step_0': term2[0], 'step_10': term2[1], 'step_25': term2[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_0': logp_normal[0], 'step_10': logp_normal[1], 'step_25': logp_normal[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_0': logp_svgd[0], 'step_10': logp_svgd[1], 'step_25': logp_svgd[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_0': logp_tanh[0], 'step_10': logp_tanh[1], 'step_25': logp_tanh[2]}, itr=self.episode_counter)

                # self.add_scalars(tb_path='Entropy/logp_toy_line1', value={'step_0': logp_toy_line1[0], 'step_10': logp_toy_line1[1], 'step_25': logp_toy_line1[2]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line2', value={'step_0': logp_toy_line2[0], 'step_10': logp_toy_line2[1], 'step_25': logp_toy_line2[2]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line4', value={'step_0': logp_toy_line4[0], 'step_10': logp_toy_line4[1], 'step_25': logp_toy_line4[2]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_wrong', value={'step_0': logp_wrong[0], 'step_10': logp_wrong[1], 'step_25': logp_wrong[2]}, itr=self.episode_counter)


            elif len(log_p) == 2:
                self.add_scalars(tb_path='Entropy/entropy', value={'step_0': log_p[0], 'step_10': log_p[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/term1', value={'step_0': term1[0], 'step_10': term1[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/term2', value={'step_0': term2[0], 'step_10': term2[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_0': logp_normal[0], 'step_10': logp_normal[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_0': logp_svgd[0], 'step_10': logp_svgd[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_0': logp_tanh[0], 'step_10': logp_tanh[1]}, itr=self.episode_counter)

                # self.add_scalars(tb_path='Entropy/logp_toy_line1', value={'step_0': logp_toy_line1[0], 'step_10': logp_toy_line1[1]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line2', value={'step_0': logp_toy_line2[0], 'step_10': logp_toy_line2[1]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line4', value={'step_0': logp_toy_line4[0], 'step_10': logp_toy_line4[1]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_wrong', value={'step_0': logp_wrong[0], 'step_10': logp_wrong[1]}, itr=self.episode_counter)

            elif len(log_p) == 1:
                self.add_scalars(tb_path='Entropy/entropy', value={'step_0': log_p[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/term1', value={'step_0': term1[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/term2', value={'step_0': term2[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_0': logp_normal[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_0': logp_svgd[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_0': logp_tanh[0]}, itr=self.episode_counter)

                # self.add_scalars(tb_path='Entropy/logp_toy_line1', value={'step_0': logp_toy_line1[0]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line2', value={'step_0': logp_toy_line2[0]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line4', value={'step_0': logp_toy_line4[0]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_wrong', value={'step_0': logp_wrong[0]}, itr=self.episode_counter)

            self.episode_counter += 1

         
    def plot_policy(self, itr, fig_path, plot):
        
        if plot and self.env_name in ['Multigoal', 'multigoal-max-entropy']:
            ax = self.test_env._init_plot(x_size=7, y_size=7, grid_size=(1,1), debugging=True)
            path = self.episodes_information[0]
            positions = np.stack(path['observations'])

            if self.ac.pi.actor != 'sac':
                for indx, i in enumerate([0, 10, 25]):
                    if len(positions) > i+1:
                        new_positions = np.clip(np.expand_dims(positions[i], 0) + path['actions'][i], self.test_env.observation_space.low, self.test_env.observation_space.high)
                        ax.plot(new_positions[:, 0], new_positions[:, 1], '+', color=self.colors[indx])
                
            ax.plot(positions[:, 0], positions[:, 1], '+b')

            for i in range(len(positions)):
                ax.annotate(str(i), (positions[i,0], positions[i,1]), fontsize=6)

            for i in range(len(positions)-1):
                if self.ac.pi.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
                    mu = path['mu'][i][0]
                    std = path['sigma'][i][0]
                    # print('########################## ', mu, std)
                else:
                    mu = 0
                    std = 1

                x_values = np.linspace(positions[i] + mu + self.test_env.action_space.low, positions[i] + mu + self.test_env.action_space.high , 30)
                plt.plot(x_values[:,0] , gaussian(x_values, positions[i]+mu, std)[:,0])
            
            plt.savefig(fig_path + '/path_vis_'+ str(itr) + '.' + self.plot_format)   
            plt.close()


    def add_scalar(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_scalar(tb_path, value, itr)
    
    def add_scalars(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_scalars(tb_path, value, itr)

    def add_histogram(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_histogram(tb_path, value, itr)

    def entorpy_landscape(self, fig_path):
        self.ac.pi.num_particles = 100
        self.ac.pi.batch_size = 100
        if not self.ac.pi.actor=='sac':
            self.ac.pi.Kernel.num_particles = 100
            self.ac.pi.identity = torch.eye(self.ac.pi.num_particles).to(self.ac.pi.device)
        # resolution = 0.5
        resolution = 0.05
        self._ax = []
        x_lim = (-7, 7)
        y_lim = (-7, 7)
        self.entropy_fig = plt.figure(figsize=(7.5, 7), constrained_layout=True) 
        self._ax.append(plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1))
        self._ax[0].axis('equal')
        self._ax[0].set_xlim(x_lim)
        self._ax[0].set_ylim(y_lim)
        self._ax[0].autoscale_view('tight')
        self._ax[0].set_title('Entropy Heatmap', fontsize=18)
        # self._ax[0].set_xlabel('x', fontsize=13)
        # self._ax[0].set_ylabel('y', fontsize=13)
        self._ax[0].xaxis.set_tick_params(labelsize=13)
        self._ax[0].yaxis.set_tick_params(labelsize=13)
        x_min, x_max = tuple(np.round(1.1 * np.array(x_lim), 3))
        y_min, y_max = tuple(np.round(1.1 * np.array(x_lim), 3))
        x_range = np.arange(x_min, x_max, resolution)
        y_range = np.arange(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_range, y_range)
        X_shape = X.shape
        X__ = X.reshape(-1, 1)
        Y__ = Y.reshape(-1, 1)
        input_ = np.concatenate((X__, Y__), axis=1)
        
        entropy = []
        idx_start = 0
        idx_end = self.ac.pi.batch_size
        pbar = tqdm(total=len(input_) + 1)
        while idx_start < len(input_):
            o = input_[idx_start:idx_end, :]
            o = torch.tensor(o, dtype=torch.float32).view(-1,1,self.train_env.observation_space.shape[0])\
                .repeat(1,self.ac.pi.num_particles,1).view(-1,self.train_env.observation_space.shape[0]).to(self.ac.pi.device)
            log_p = []
            for _ in range(10):
                a, logp_pi = self.ac(o, deterministic=False, with_logprob=True, all_particles=False)
                if self.ac.pi.actor=='sac':
                    logp_pi = logp_pi.view(-1,self.ac.pi.num_particles).mean(dim=1)
                log_p.append(list(-logp_pi.detach().cpu().numpy()))
            entropy += np.array(log_p).mean(0).tolist()
            idx_start = idx_end
            idx_end = min(idx_end + self.ac.pi.batch_size, len(input_))
            pbar.update(idx_end - idx_start)
        pbar.close()

        entropy = np.array(entropy).reshape(X_shape)

        min_entropy = entropy.min()
        max_entropy = entropy.max()
        print('##################### ', min_entropy, max_entropy)
        extent = (x_min, x_max, y_min, y_max)
        # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, norm=mpl_colors.PowerNorm(gamma = 0.7), alpha=.9, interpolation='bilinear', extent=extent)
        # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent)
        cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent)
        # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent, vmin=-0.7, vmax=0.7)
        # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent, vmin=-0.3, vmax=0.7)
        # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent, vmin=-0.5, vmax=0.5)
        # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent, vmin=-0.3, vmax=0.5)
        # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent, vmin=-0.3, vmax=0.5)
        # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.gist_ncar, extent=extent, vmin=-0.4, vmax=0.4)
        # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent, vmin=-1.8)
        self._ax[0].plot(self.test_env.goal_positions[:, 0], self.test_env.goal_positions[:, 1], 'ro')
        self._ax[0].plot(np.array([0]), np.array([0]), 'bo')
        


        

        range_ = np.arange(min_entropy, max_entropy, 0.1)[1::2]
        # range_ = np.arange(-1.5, 1.5, 0.1)[1::2]
        # plt.colorbar(mappable=cmap)
        self.entropy_fig.colorbar(mappable=cmap, ticks=range_)


        plt.savefig(fig_path + '/entropy_landscape.' + self.plot_format)   
        plt.close()

    def log_to_tensorboard(self, itr):
        # related to the modes
        if self.env_name in ['Multigoal', 'multigoal-max-entropy']:
            self.tb_logger.add_scalar('modes/num_modes',(self.test_env.number_of_hits_mode>0).sum(), itr)
            self.tb_logger.add_scalar('modes/total_number_of_hits_mode',self.test_env.number_of_hits_mode.sum(), itr)
            if self.env_name == 'multigoal-max-entropy':
                self.tb_logger.add_scalars('modes/hits_mode_accurate',{'mode_0': self.test_env.number_of_hits_mode_acc[0], 'mode_1': self.test_env.number_of_hits_mode_acc[1], 'mode_2': self.test_env.number_of_hits_mode_acc[2]}, itr)
            elif self.env_name == 'Multigoal':
                self.tb_logger.add_scalars('modes/hits_mode_accurate',{'mode_0': self.test_env.number_of_hits_mode_acc[0], 'mode_1': self.test_env.number_of_hits_mode_acc[1], 'mode_2': self.test_env.number_of_hits_mode_acc[2], 'mode_3': self.test_env.number_of_hits_mode_acc[3]}, itr)
            for ind in range(self.test_env.num_goals):
                self.tb_logger.add_scalar('modes/prob_mod_'+str(ind),self.test_env.number_of_hits_mode[ind]/self.test_env.number_of_hits_mode.sum() if self.test_env.number_of_hits_mode.sum() != 0 else 0.0, itr)
        if self.env_name in ['max-entropy-v0','multigoal-max-entropy']:
            feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.entropy_list[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
            self.tb_logger.add_scalars('Entropy/max_entropy_env_Entropies',  feed_dict, itr)

            # feed_dict = {str(self.test_env.entropy_obs_names[i]): self.ac.q1(torch.tensor(self.entropy_obs_names[i]), ) for i in range(self.test_env.entropy_obs_names.shape[0])}
            # self.tb_logger.add_scalars('Entropy/max_entropy_env_Q_values',  feed_dict, itr)
            # if self.env_name in ['max-entropy-v0']:
            #     self.tb_logger.add_scalars('Entropy/max_entropy_env_Paths', {f'path_{i + 1}': self.test_env.paths[i] for i in range(len(self.test_env.paths))}, itr)
            #     self.tb_logger.add_scalars('Entropy/max_entropy_env_Failures', {'goal_1': self.train_env.failures[0], 'goal_2': self.train_env.failures[1]}, itr)

            #     if self.ac.pi.actor == 'sac':
            #         feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.mean_list_x[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
            #         self.tb_logger.add_scalars('Entropy/max_entropy_env_Means_x',  feed_dict, itr)
            #         feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.mean_list_y[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
            #         self.tb_logger.add_scalars('Entropy/max_entropy_env_Means_y',  feed_dict, itr)
            #         feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.sigma_list_x[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
            #         self.tb_logger.add_scalars('Entropy/max_entropy_env_Sigmas_x',  feed_dict, itr)
            #         feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.sigma_list_y[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
            #         self.tb_logger.add_scalars('Entropy/max_entropy_env_Sigmas_y',  feed_dict, itr)
            

            # if self.env_name in ['max-entropy-v0']:
            #     feed_dict = {}
            #     o = torch.tensor(self.test_env.entropy_obs_list[0].astype(np.float32)).to(self.ac.pi.device)
            #     left = torch.from_numpy(np.array([-1, 1]).astype(np.float32)).to(self.ac.pi.device)
            #     right = torch.from_numpy(np.array([1, 1]).astype(np.float32)).to(self.ac.pi.device)
            #     feed_dict[self.test_env.entropy_obs_names[0] + '_left'] = torch.min(self.ac.q1(o, left), self.ac.q2(o, left)).detach().cpu().item()
            #     feed_dict[self.test_env.entropy_obs_names[0] + '_right'] = torch.min(self.ac.q1(o, right), self.ac.q2(o, right)).detach().cpu().item()

            #     o = torch.tensor(self.test_env.entropy_obs_list[1].astype(np.float32)).to(self.ac.pi.device)
            #     left = torch.from_numpy(np.array([-1, 0]).astype(np.float32)).to(self.ac.pi.device)
            #     right = torch.from_numpy(np.array([1, 0]).astype(np.float32)).to(self.ac.pi.device)
            #     feed_dict[self.test_env.entropy_obs_names[1] + '_left'] = torch.min(self.ac.q1(o, left), self.ac.q2(o, left)).detach().cpu().item()
            #     feed_dict[self.test_env.entropy_obs_names[1] + '_right'] = torch.min(self.ac.q1(o, left), self.ac.q2(o, left)).detach().cpu().item()
            
            #     o = torch.tensor(self.test_env.entropy_obs_list[2].astype(np.float32)).to(self.ac.pi.device)
            #     left = torch.from_numpy(np.array([-1, 0]).astype(np.float32)).to(self.ac.pi.device)
            #     feed_dict[self.test_env.entropy_obs_names[2] + '_left'] = torch.min(self.ac.q1(o, left), self.ac.q2(o, left)).detach().cpu().item()
            
            #     o = torch.tensor(self.test_env.entropy_obs_list[3].astype(np.float32)).to(self.ac.pi.device)
            #     top = torch.from_numpy(np.array([0, 1]).astype(np.float32)).to(self.ac.pi.device)
            #     feed_dict[self.test_env.entropy_obs_names[3] + '_top'] = torch.min(self.ac.q1(o, top), self.ac.q2(o, top)).detach().cpu().item()
            
            #     o = torch.tensor(self.test_env.entropy_obs_list[4].astype(np.float32)).to(self.ac.pi.device)
            #     top = torch.from_numpy(np.array([0, 1]).astype(np.float32)).to(self.ac.pi.device)
            #     feed_dict[self.test_env.entropy_obs_names[4] + '_top'] = torch.min(self.ac.q1(o, top), self.ac.q2(o, top)).detach().cpu().item()
            
            #     o = torch.tensor(self.test_env.entropy_obs_list[5].astype(np.float32)).to(self.ac.pi.device)
            #     top = torch.from_numpy(np.array([0, 1]).astype(np.float32)).to(self.ac.pi.device)
            #     feed_dict[self.test_env.entropy_obs_names[5] + '_top'] = torch.min(self.ac.q1(o, top), self.ac.q2(o, top)).detach().cpu().item()

            #     self.tb_logger.add_scalars('Entropy/max_entropy_env_q_values',  feed_dict, itr)

            # self._obs_lst = np.array([
        #     [-3.5, -2.5],
        #     [-3.5, 0],
        #     [-3.5, 2.5],
        #     [-2, -1.5],
        #     [-2, 1.5],
        #     [-1, 0],
        #     [0, 0],
        #     [1, 0],
        #     [2, 0],
        #     [4, 0],
        # ])
        # self._obs_lst = np.array([
        #     [-3.5, -2.5],
        #     [-3.5, 0],
        #     [-3.5, 2.5],
        #     [-1, 0],
        #     [0, 0],
        #     [1, 0],
        #     [4, 0],
        # ])

            if self.env_name in ['multigoal-max-entropy']:
                feed_dict = {}
                o = torch.tensor(self.test_env._obs_lst[4].astype(np.float32)).to(self.ac.pi.device)
                left = torch.from_numpy(np.array([-1, 0]).astype(np.float32)).to(self.ac.pi.device)
                right = torch.from_numpy(np.array([1, 0]).astype(np.float32)).to(self.ac.pi.device)
                feed_dict[self.test_env.entropy_obs_names[4] + '_left'] = torch.min(self.ac.q1(o, left), self.ac.q2(o, left)).detach().cpu().item()
                feed_dict[self.test_env.entropy_obs_names[4] + '_right'] = torch.min(self.ac.q1(o, right), self.ac.q2(o, right)).detach().cpu().item()
                
                o = torch.tensor(self.test_env._obs_lst[3].astype(np.float32)).to(self.ac.pi.device)
                topleft = torch.from_numpy(np.array([-1, 1]).astype(np.float32)).to(self.ac.pi.device)
                left = torch.from_numpy(np.array([-1, 0]).astype(np.float32)).to(self.ac.pi.device)
                bottomleft = torch.from_numpy(np.array([-1, -1]).astype(np.float32)).to(self.ac.pi.device)
                feed_dict[self.test_env.entropy_obs_names[3] + '_topleft'] = torch.min(self.ac.q1(o, topleft), self.ac.q2(o, topleft)).detach().cpu().item()
                feed_dict[self.test_env.entropy_obs_names[3] + '_left'] = torch.min(self.ac.q1(o, left), self.ac.q2(o, left)).detach().cpu().item()
                feed_dict[self.test_env.entropy_obs_names[3] + '_bottomleft'] = torch.min(self.ac.q1(o, bottomleft), self.ac.q2(o, bottomleft)).detach().cpu().item()

                o = torch.tensor(self.test_env._obs_lst[5].astype(np.float32)).to(self.ac.pi.device)
                right = torch.from_numpy(np.array([1, 0]).astype(np.float32)).to(self.ac.pi.device)
                feed_dict[self.test_env.entropy_obs_names[5] + '_right'] = torch.min(self.ac.q1(o, right), self.ac.q2(o, right)).detach().cpu().item()

                self.tb_logger.add_scalars('Entropy/max_entropy_env_q_values',  feed_dict, itr)
                
            # if self.env_name == 'max-entropy-v0':
            #     for s in range(self.test_env.max_steps - 1):
            #         feed_dict_x = {}
            #         feed_dict_x['test_itr' + str(itr)] = 0
            #         feed_dict_y = {}
            #         feed_dict_y['test_itr' + str(itr)] = 0
            #         for i in range(len(self.episodes_information)):
            #             if len(self.episodes_information[i]['actions']) > s:
            #                 feed_dict_x['test_itr' + str(itr)] += self.episodes_information[i]['actions'][s][0]
            #                 feed_dict_y['test_itr' + str(itr)] += self.episodes_information[i]['actions'][s][1]
            #         feed_dict_x['test_itr' + str(itr)] /= 20
            #         feed_dict_y['test_itr' + str(itr)] /= 20
            #         self.add_scalars('Entropy/episode_action_X' ,feed_dict_x, s)
            #         self.add_scalars('Entropy/episode_action_Y' ,feed_dict_y, s)
    

            # for i in range()

        #############################################################################################################################################
        # # investigating smoothness of the q-landscape by computing the 1st and 2nd order derivatives
        # q_score_ = list(map(lambda x: np.stack(x['q_score']), self.episodes_information))
        # q_score_mean = list(map(lambda x: x.mean(), q_score_))
        # q_score_min = list(map(lambda x: x.min(), q_score_))
        # q_score_max = list(map(lambda x: x.max(), q_score_))

        # q_hess_ = list(map(lambda x: np.stack(x['q_hess']), self.episodes_information))
        # q_hess_mean = list(map(lambda x: x.mean(), q_hess_))
        # q_hess_min = list(map(lambda x: x.min(), q_hess_))
        # q_hess_max = list(map(lambda x: x.max(), q_hess_))
        
        # self.tb_logger.add_scalars('smoothness/q_score',  {'Mean ': np.mean(q_score_mean), 'Min': np.mean(q_score_min), 'Max': np.mean(q_score_max)  }, itr)
        # self.tb_logger.add_scalars('smoothness/q_hess', {'Mean ': np.mean(q_hess_mean), 'Min': np.mean(q_hess_min), 'Max': np.mean(q_hess_max)  }, itr)
        
        # q_score_averaged = []
        # q_hess_averaged = []

        # for i in ['_start', '_mid', '_end']:
        #     q_score_i = np.array(list(map(lambda x: x['q_score' + i], self.episodes_information)))
        #     q_score_averaged.append(np.mean(q_score_i[q_score_i != np.array(None)]))

        #     q_hess_i = np.array(list(map(lambda x: x['q_hess' + i], self.episodes_information)))
        #     q_hess_averaged.append(np.mean(q_hess_i[q_hess_i != np.array(None)]))

        # self.tb_logger.add_scalars('smoothness/q_score_averaged',  {'Start ': q_score_averaged[0], 'Mid': q_score_averaged[1], 'End': q_score_averaged[2] }, itr)
        # self.tb_logger.add_scalars('smoothness/q_hess_averaged', {'Start ': q_hess_averaged[0], 'Mid': q_hess_averaged[1], 'End': q_hess_averaged[2] }, itr)

        # # 
        # expected_rewards = list(map(lambda x: x['expected_reward'], self.episodes_information))
        # episode_length = list(map(lambda x: x['episode_length'], self.episodes_information))

        # self.tb_logger.add_scalars('Test_EpRet',  {'Mean ': np.mean(expected_rewards), 'Min': np.min(expected_rewards), 'Max': np.max(expected_rewards) }, itr)
        # self.tb_logger.add_scalar('Test_EpLen', np.mean(episode_length) , itr)
        
    def reset(self,):
        self.episodes_information = []



    


    






