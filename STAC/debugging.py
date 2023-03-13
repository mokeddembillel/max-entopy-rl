import numpy as np 
import torch as torch
import matplotlib.pyplot as plt
from utils import gaussian
from tqdm import tqdm
import matplotlib.colors as mpl_colors
from mpl_toolkits.mplot3d import Axes3D


class Debugger():
    def __init__(self, tb_logger, ac, env_name, train_env, test_env, plot_format, update_after, num_test_episodes, alpha, env_max_steps, max_experiment_steps):
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
        self.plot_cumulative_entropy = update_after + 5000
        self.num_test_episodes = num_test_episodes
        self.alpha = alpha
        self.boundary_action_counter = 0 
        self.boundary_all_actions_counter = 0 
        self.action_counter = 0 
        self.env_max_steps = env_max_steps
        self.max_experiment_steps = max_experiment_steps - 1
        
        if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            self.average_cumulative_entropy = np.zeros((self.test_env.num_goals))
            self.cumulative_entropy_coutner = np.zeros((self.test_env.num_goals))

    def collect_data(self, o, a, o2, r, d, log_p, itr, ep_len, robot_pic_rgb=None):
        
        if ep_len == 0:
            self.episodes_information.append({
                'observations':[],
                'action': [],
                'actions': [],
                'x_t': [],
                # Debugging ##########
                'log_p': [],
                'term1': [],
                'term2': [],
                'logp_normal': [],
                'logp_svgd': [],
                'logp_tanh': [],
                'goal': None,
                'intersection': False,
                'diff_p0_p': [],
                'diff_st_st1': [],
                'st_components': [],
                'next_state_rgb': [],

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
                'q_hess_mat' : [],
                'max_eigenval' : [],
                'q_hess_start': None, 
                'q_hess_mid': None, 
                'q_hess_end': None, 
                })

        self.episodes_information[-1]['observations'].append(o.detach().cpu().numpy().squeeze())

        if self.ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
            self.episodes_information[-1]['actions'].append(self.ac.pi.a.detach().cpu().numpy().squeeze())
            self.episodes_information[-1]['x_t'].append(np.array(self.ac.pi.x_t))
        self.episodes_information[-1]['action'].append(a.detach().cpu().numpy().squeeze())
        self.episodes_information[-1]['rewards'].append(r)

        a.requires_grad = True
        q1_value = self.ac.q1(o,a)
        q2_value = self.ac.q2(o,a)

        if log_p is not None:
            self.episodes_information[-1]['log_p'].append(-log_p.detach().item())
            # self.episodes_information[-1]['term1'].append(self.ac.pi.term1_debug.detach().cpu())
            # self.episodes_information[-1]['term2'].append(self.ac.pi.term2_debug.detach().cpu())
            self.episodes_information[-1]['logp_normal'].append(self.ac.pi.logp_normal_debug.detach().item())
            self.episodes_information[-1]['logp_svgd'].append(self.ac.pi.logp_svgd_debug.detach().item())
            self.episodes_information[-1]['logp_tanh'].append(self.ac.pi.logp_tanh_debug.detach().item())
            # self.episodes_information[-1]['logp_toy_line1'].append(self.ac.pi.logp_line1.mean().cpu().detach().item())
            # self.episodes_information[-1]['logp_toy_line2'].append(self.ac.pi.logp_line2.mean().cpu().detach().item())
            # self.episodes_information[-1]['logp_toy_line4'].append(self.ac.pi.logp_line4.mean().cpu().detach().item())
            # self.episodes_information[-1]['logp_wrong'].append(self.ac.pi.logp_wrong.mean().cpu().detach().item())

        if self.env_name in ['Hopper-v2'] and robot_pic_rgb is not None:
            self.episodes_information[-1]['next_state_rgb'].append(robot_pic_rgb.tolist())

        if self.ac.pi.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
            self.episodes_information[-1]['mu'].append(self.ac.pi.mu.detach().cpu().numpy())
            self.episodes_information[-1]['sigma'].append(self.ac.pi.sigma.detach().cpu().numpy())
        
        if self.ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
            self.episodes_information[-1]['diff_p0_p'].append(torch.linalg.norm(self.ac.pi.a - self.ac.pi.a0_debbug, dim=2).mean().cpu().detach().item())
        self.episodes_information[-1]['diff_st_st1'].append(np.linalg.norm(o2 - o.squeeze().cpu().detach().numpy()))
        self.episodes_information[-1]['st_components'].append(o2)

        grad_q_ = torch.autograd.grad(torch.min(q1_value, q2_value), a, retain_graph=True, create_graph=True)[0].squeeze()
        self.episodes_information[-1]['q_score'].append(torch.abs(grad_q_).mean().detach().cpu().item())
        if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            hess_q_mat = torch.stack([torch.autograd.grad(grad_q_[0], a, retain_graph=True)[0], torch.autograd.grad(grad_q_[1], a, retain_graph=True)[0]]).squeeze()
            hess_q = ((torch.abs(torch.autograd.grad(grad_q_[0], a, retain_graph=True)[0]) + torch.abs(torch.autograd.grad(grad_q_[1], a, retain_graph=True)[0])).sum()/4)
            self.episodes_information[-1]['q_hess'].append(hess_q.detach().cpu().item())
            # import pdb; pdb.set_trace()
            self.episodes_information[-1]['q_hess_mat'].append(hess_q_mat.detach().cpu().numpy())
            self.episodes_information[-1]['max_eigenval'].append(torch.max(torch.linalg.eigvals(hess_q_mat).real).detach().cpu().item())
        self.episodes_information[-1]['q1_values'] = q1_value.detach().cpu().numpy()
        self.episodes_information[-1]['q2_values'] = q2_value.detach().cpu().numpy()



        if self.ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
            # print(self.ac.pi.a)
            if (a < -0.95).any() or (a > 0.95).any():
                self.boundary_action_counter += 1
            if (self.ac.pi.a < -0.95).any() or (self.ac.pi.a > 0.95).any():
                self.boundary_all_actions_counter += 1
        self.action_counter +=  1




        if ((ep_len + 1) >= self.env_max_steps) or d: 
            self.episodes_information[-1]['observations'].append(o2.squeeze())
            self.episodes_information[-1]['expected_reward'] = np.sum(self.episodes_information[-1]['rewards'])
            self.episodes_information[-1]['episode_length'] = ep_len
            if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
                self.episodes_information[-1]['intersection'] = self.test_env.intersection
                # if d == False:
                #     import pdb; pdb.set_trace()
                #     print('############# ', d)
                if d:
                    self.episodes_information[-1]['goal'] = self.test_env.min_dist_index
                    # print('#########$$$$$$$$$$$################# ', self.episodes_information[-1]['goal'])
                if itr > self.plot_cumulative_entropy and d:
                    # print('####################### ', self.average_cumulative_entropy)
                    self.cumulative_entropy_coutner[self.test_env.min_dist_index] += 1
                    self.average_cumulative_entropy[self.test_env.min_dist_index] = ((self.cumulative_entropy_coutner[self.test_env.min_dist_index] - 1) * self.average_cumulative_entropy[self.test_env.min_dist_index] + np.array(self.episodes_information[-1]['log_p']).mean()) / (self.cumulative_entropy_coutner[self.test_env.min_dist_index])
                    # print('####################### ', self.average_cumulative_entropy)

            if ep_len >= 5:
                self.episodes_information[-1]['q_score_start'] = np.mean(self.episodes_information[-1]['q_score'][:5])
                self.episodes_information[-1]['q_hess_start'] = np.mean(self.episodes_information[-1]['q_hess'][:5])
            if ep_len >= 17:
                self.episodes_information[-1]['q_score_mid'] = np.mean(self.episodes_information[-1]['q_score'][12:17])
                self.episodes_information[-1]['q_hess_mid'] = np.mean(self.episodes_information[-1]['q_hess'][12:17])
            if ep_len >= 30:
                self.episodes_information[-1]['q_score_end'] = np.mean(self.episodes_information[-1]['q_score'][25:ep_len])
                self.episodes_information[-1]['q_hess_end'] = np.mean(self.episodes_information[-1]['q_hess'][25:ep_len])
    
    def create_entropy_plots(self, itr):
        if self.env_name in ['Hopper-v2', 'Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles'] and self.ac.pi.actor in ['svgd_nonparam'] and ((itr + 1)%64000 == 0 or itr == self.max_experiment_steps):
        # if 1:
            for s in range(len(self.episodes_information[0]['log_p'])):
                feed_dict = {}
                feed_dict['episode_' + str(0)] = self.episodes_information[0]['log_p'][s]
                self.add_scalars('Entropy/episode_entropy_' + str(itr),feed_dict, s)
    
    # def create_entropy_plots(self, itr):
    #     if self.env_name in ['Hopper-v2', 'Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles'] and self.ac.pi.actor in ['svgd_nonparam'] and ((itr + 1)%64000 == 0 or itr == self.max_experiment_steps):
    #     # if 1:
    #         for s in range(self.env_max_steps):
    #             feed_dict = {}
    #             for i in range(len(self.episodes_information)):
    #                 if len(self.episodes_information[i]['log_p']) > s:
    #                     feed_dict['episode_' + str(i)] = self.episodes_information[i]['log_p'][s]
    #             self.add_scalars('Entropy/episode_entropy_' + str(itr),feed_dict, s)
    
    def create_states_distances_plots(self, itr):
        if self.env_name in ['Hopper-v2'] and ((itr + 1)%64000 == 0 or itr == self.max_experiment_steps):
        # if 1:
            for s in range(len(self.episodes_information[0]['diff_st_st1'])):
                feed_dict = {}
                feed_dict['episode_' + str(0)] = self.episodes_information[0]['diff_st_st1'][s]
                self.add_scalars('actions/episode_s_norm_' + str(itr),feed_dict, s)
    # def create_states_distances_plots(self, itr):
    #     if self.env_name in ['Hopper-v2'] and ((itr + 1)%64000 == 0 or itr == self.max_experiment_steps):
    #     # if 1:
    #         for s in range(self.env_max_steps):
    #             feed_dict = {}
    #             for i in range(len(self.episodes_information)):
    #                 still_plotting = False
    #                 if len(self.episodes_information[i]['diff_st_st1']) > s:
    #                     still_plotting = True
    #                     feed_dict['episode_' + str(i)] = self.episodes_information[i]['diff_st_st1'][s]
    #                 if not still_plotting:
    #                     break
    #             self.add_scalars('actions/episode_s_norm_' + str(itr),feed_dict, s)

    def create_states_components_plots(self, itr):
        if self.env_name in ['Hopper-v2'] and ((itr + 1)%64000 == 0 or itr == self.max_experiment_steps):
        # if 1:
            paths = self.episodes_information[0]['st_components']
            for s in range(len(paths)):
                feed_dict = {}
                feed_dict = {'component_' + str(i): paths[s][i] for i in range(len(paths[0]))}
                self.add_scalars('actions/s_components_step_' + str(itr),feed_dict, s)
                

    def create_actions_components_plots(self, itr):
        if self.env_name in ['Hopper-v2'] and ((itr + 1)%64000 == 0 or itr == self.max_experiment_steps):
        # if 1:
            paths = self.episodes_information[0]['action']
            for s in range(len(paths)):
                feed_dict = {}
                feed_dict = {'component_' + str(i): paths[s][i] for i in range(len(paths[0]))}
                self.add_scalars('actions/act_components_step_' + str(itr),feed_dict, s)
                

         
    def plot_policy(self, itr, fig_path, plot, num_agents=1):
        num_agents = self.num_test_episodes
        if plot and self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            ax = self.test_env._init_plot(x_size=7, y_size=7, grid_size=(1,1), debugging=True)

            cats_success = np.zeros((3,))
            cats_steps = np.zeros((3,))
            # for a in range(num_agents):
            path = self.episodes_information[0]
            positions = np.stack(path['observations'])
            
            # if self.ac.pi.actor != 'sac':
            #     for indx, i in enumerate([0, 10, 25]):
            #         if len(positions) > i+1:
            #             new_positions = np.clip(np.expand_dims(positions[i], 0) + path['actions'][i], self.test_env.observation_space.low, self.test_env.observation_space.high)
            #             ax.plot(positions[i, 0], positions[i, 1], marker='+', color='black', zorder=4, markersize=10, mew=2)
            #             ax.plot(new_positions[:, 0], new_positions[:, 1], '+', color=self.colors[indx], zorder=3)

            if not path['intersection']:
                cats_success[0]+= 1
                cats_steps[0]+= len(positions)
                ax.plot(positions[:, 0], positions[:, 1], color='blue')
            else:
                # print('########################## ', path['goal'])
                if path['goal'] is not None:
                    cats_success[1]+= 1
                    cats_steps[1]+= len(positions)
                    ax.plot(positions[:, 0], positions[:, 1], color='lime')
                else:
                    cats_success[2]+= 1
                    cats_steps[2]+= len(positions)
                    ax.plot(positions[:, 0], positions[:, 1], color='red')

                # for i in range(len(positions)):
                #     ax.annotate(str(i), (positions[i,0], positions[i,1]), fontsize=6)
                
                # ax.annotate(str(len(positions)), (positions[-1,0], positions[-1,1]), fontsize=6)
            # path = self.episodes_information[0]
            # positions = np.stack(path['observations'])
            if self.ac.pi.actor in ['sac', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
                for i in range(len(positions)-1):
                    mu = path['mu'][i][0]
                    std = path['sigma'][i][0]
                    # print('################################ ', mu, std)
                    x_values = np.linspace(positions[i] + mu + self.test_env.action_space.low, positions[i] + mu + self.test_env.action_space.high , 30)
                    plt.plot(x_values[:,0] , gaussian(x_values, positions[i]+mu, std)[:,0])
            # print('######################## ', cats_success)
            # print('######################## ', cats_steps/cats_success)
            # plt.savefig(fig_path + '/path_vis_'+ str(itr) + '.' + self.plot_format)   
            plt.savefig(fig_path + '/path_vis_'+ str(itr) + '.' + 'png')   
            plt.close()   


    def add_scalar(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_scalar(tb_path, value, itr)
    
    def add_scalars(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_scalars(tb_path, value, itr)

    def add_histogram(self, tb_path=None, value=None, itr=None):
            self.tb_logger.add_histogram(tb_path, value, itr)







    def mujoco_debugging_plots(self, itr, fig_path, fig_size=(10, 18)):
        grid_size=(4,7)
        labels = [['$a_2$', '$a_3$'], ['$a_2$', '$a_3$'], ['$a_2$', '$a_3$'], ['$a_1$', '$a_2$'], ['$a_2$', '$a_3$'], ['$a_1$', '$a_3$']]
        
        _n_samples = 10


        obs = self.episodes_information[0]['observations']
        log_p = self.episodes_information[0]['log_p']
        actions = self.episodes_information[0]['action']
        obs_rgb = self.episodes_information[0]['next_state_rgb']
        obs_rgb.insert(0, np.zeros_like(obs_rgb[0]).tolist())
        if ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
            particles = np.array(self.episodes_information[0]['x_t']) #### (n_steps, np, ad)
        else:
            particles = self.episodes_information[0]['action']

        if ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
            num_particles_tmp = self.ac.pi.num_particles
            self.ac.pi.num_particles = _n_samples
            self.ac.pi.Kernel.num_particles = self.ac.pi.num_particles
            self.ac.pi.identity = torch.eye(self.ac.pi.num_particles).to(self.ac.pi.device)


        for i in range(min(500, len(obs))-1):
        
            ################################################
            ################################################
            ### SETING UP THE GRID
        
            _ax_lst = []
            fig_env = plt.figure(figsize=fig_size, layout="constrained") 
            # fig_env.set_constrained_layout_pads(w_pad=2./12., h_pad=4./12.,
            # hspace=0., wspace=0.)
            ax = plt.subplot2grid(grid_size, (0,0),  projection='3d', colspan=3, rowspan=3)
            ax.set_title(r'$H(s_t)=$' + str(round(log_p[i], 2)))
            ax.set_xlabel('$a_1$')
            ax.set_ylabel('$a_2$')
            ax.set_zlabel('$a_3$')
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            ax.set_zlim((-1, 1))
            ax.set_box_aspect([2, 2, 2])  # equal aspect ratio
            _ax_lst.append(ax)

            ax = plt.subplot2grid(grid_size, (0,3), colspan=3, rowspan=3)
            _ax_lst.append(ax)

            ax_cl = plt.subplot2grid(grid_size, (0,6), colspan=1, rowspan=4)

            for j in range(6):
                ax = plt.subplot2grid(grid_size, (3+j//(grid_size[1]-1),j%(grid_size[1]-1)))
                ax.set_xlabel(labels[j][0])
                ax.set_ylabel(labels[j][1])
                ax.set_xlim((-1, 1))
                ax.set_ylim((-1, 1))
                ax.grid(True)
                _ax_lst.append(ax)

            ################################################
            ################################################
            ### FILLING UP THE 3D PLOT AND PROJECTIONS PLOTS
            if self.ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
                x, y, z = particles[i][-1][:, 0], particles[i][-1][:, 1], particles[i][-1][:, 2]  # for show
                o = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).repeat(particles[i][-1].shape[0], 1).to(self.ac.pi.device).detach()
                a = torch.tensor(particles[i][-1], dtype=torch.float32).to(self.ac.pi.device).detach()
            else:
                # import pdb;pdb.set_trace()
                x, y, z = np.array([particles[i][0]]), np.array([particles[i][1]]), np.array([particles[i][2]])
                o = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(self.ac.pi.device).detach()
                a = torch.tensor(particles[i], dtype=torch.float32).unsqueeze(0).to(self.ac.pi.device).detach()

            q1_value = self.ac.q1(o,a)
            q2_value = self.ac.q2(o,a)
            q_values = np.round(torch.min(q1_value, q2_value).detach().cpu().numpy(), 3)
            p = _ax_lst[0].scatter(x, y, z, c=q_values, cmap=plt.cm.viridis, alpha=1)
            _ax_lst[5].scatter(x, y, c=q_values, cmap=plt.cm.viridis, alpha=1)
            _ax_lst[6].scatter(y, z, c=q_values, cmap=plt.cm.viridis, alpha=1)
            _ax_lst[7].scatter(x, z, c=q_values, cmap=plt.cm.viridis, alpha=1)
            fig_env.colorbar(p, cax=ax_cl)

            if self.ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
                for j in range(particles.shape[2]):
                    x_p, y_p, z_p = particles[i, :, j, 0], particles[i, :, j, 1], particles[i, :, j, 2]
                    _ax_lst[0].plot(x_p, y_p, z_p, c='red')
                    _ax_lst[5].plot(x_p, y_p, c='red')
                    _ax_lst[6].plot(y_p, z_p, c='red')
                    _ax_lst[7].plot(x_p, z_p, c='red')
                
            ################################################
            ################################################
            ### FILLING UP THE STATE RGB

            _ax_lst[1].imshow(np.array(obs_rgb[i]))            
            
            ################################################
            ################################################
            ### FILLING UP THE CONTOURS PLOTS 
            act_dim_0 = np.array(actions)[:,0]
            _act_lst = np.array([np.quantile(act_dim_0, 0.25), np.quantile(act_dim_0, 0.5), np.quantile(act_dim_0, 0.75)])
            # _act_lst = np.array([-0.2, 0, 0.2])
            _line_objects = list()
            xs = np.linspace(-1, 1, 50)
            ys = np.linspace(-1, 1, 50)
            xgrid, ygrid = np.meshgrid(xs, ys)
            a_ = np.concatenate((np.expand_dims(xgrid.ravel(), -1), np.expand_dims(ygrid.ravel(), -1)), -1)
            
            for j in range(len(_act_lst)):
                a = torch.from_numpy(np.concatenate((np.full((a_.shape[0], 1), _act_lst[j]), a_), axis=1).astype(np.float32)).to(self.ac.pi.device)
                o = torch.Tensor(obs[i]).repeat([a_.shape[0],1]).to(self.ac.pi.device)
                with torch.no_grad():
                    qs = self.ac.q1(o.to(self.ac.pi.device), a).cpu().detach().numpy()
                qs = qs.reshape(xgrid.shape)
                cs = _ax_lst[j+2].contour(xgrid, ygrid, qs, 20)
                _line_objects += cs.collections
                _line_objects += _ax_lst[j+2].clabel(
                    cs, inline=1, fontsize=10, fmt='%.2f')

            
            
            for j in range(len(_act_lst)):
                if self.ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
                    o = torch.as_tensor(obs[i], dtype=torch.float32).view(-1,1,self.test_env.observation_space.shape[0]).repeat(1,self.ac.pi.num_particles,1).view(-1,self.test_env.observation_space.shape[0]).to(self.ac.pi.device)
                else:
                    o = torch.as_tensor(obs[i], dtype=torch.float32).repeat([_n_samples,1]).to(self.ac.pi.device)
                actions, _ = self.ac(o, action_selection=None, with_logprob=False)
                actions = actions.cpu().detach().numpy().squeeze()

                x, y = actions[:, 1], actions[:, 2]

                _ax_lst[j+2].set_title('action_dim_' + str(round(_act_lst[j], 3)), fontsize=12)
                _line_objects += _ax_lst[j+2].plot(x, y, 'b*')

            
            plt.savefig(fig_path + '/mujoco_debugging_plot_step_'+ str(i) + '.' + 'png')   
            plt.close()

        if self.ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
            self.ac.pi.num_particles = num_particles_tmp
            self.ac.pi.Kernel.num_particles = self.ac.pi.num_particles
            self.ac.pi.identity = torch.eye(self.ac.pi.num_particles).to(self.ac.pi.device)
        



    def log_to_tensorboard(self, itr):
        # related to the modes
        if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            self.tb_logger.add_scalar('modes/num_modes',(self.test_env.number_of_hits_mode>0).sum(), itr)
            self.tb_logger.add_scalar('modes/total_number_of_hits_mode',self.test_env.number_of_hits_mode.sum(), itr)
            self.tb_logger.add_scalars('modes/hits_mode_accurate',{'mode_' + str(i): self.test_env.number_of_hits_mode_acc[i] for i in range(self.test_env.num_goals)}, itr)
            # elif self.env_name == 'Multigoal':
            #     self.tb_logger.add_scalars('modes/hits_mode_accurate',{'mode_0': self.test_env.number_of_hits_mode_acc[0], 'mode_1': self.test_env.number_of_hits_mode_acc[1], 'mode_2': self.test_env.number_of_hits_mode_acc[2], 'mode_3': self.test_env.number_of_hits_mode_acc[3]}, itr)
            for ind in range(self.test_env.num_goals):
                self.tb_logger.add_scalar('modes/prob_mod_'+str(ind),self.test_env.number_of_hits_mode[ind]/self.test_env.number_of_hits_mode.sum() if self.test_env.number_of_hits_mode.sum() != 0 else 0.0, itr)

            
        if self.env_name in ['max-entropy-v0','multigoal-max-entropy', 'multigoal-max-entropy-obstacles'] and self.ac.pi.actor != 'svgd_sql':
            feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.entropy_list[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
            self.tb_logger.add_scalars('Entropy/max_entropy_env_Entropies',  feed_dict, itr)

            feed_dict = {'goal_' + str(i): self.average_cumulative_entropy[i] for i in range(self.test_env.num_goals)}
            self.tb_logger.add_scalars('Entropy/max_entropy_env_CumulEntropies',  feed_dict, itr)
            ########################################################################################################################################################
            # if self.env_name in ['max-entropy-v0']:
            #     self.tb_logger.add_scalars('Entropy/max_entropy_env_Paths', {f'path_{i + 1}': self.test_env.paths[i] for i in range(len(self.test_env.paths))}, itr)
            #     self.tb_logger.add_scalars('Entropy/max_entropy_env_Failures', {'goal_1': self.train_env.failures[0], 'goal_2': self.train_env.failures[1]}, itr)

                # if self.ac.pi.actor == 'sac':
                #     feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.mean_list_x[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
                #     self.tb_logger.add_scalars('Entropy/max_entropy_env_Means_x',  feed_dict, itr)
                #     feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.mean_list_y[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
                #     self.tb_logger.add_scalars('Entropy/max_entropy_env_Means_y',  feed_dict, itr)
                #     feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.sigma_list_x[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
                #     self.tb_logger.add_scalars('Entropy/max_entropy_env_Sigmas_x',  feed_dict, itr)
                #     feed_dict = {str(self.test_env.entropy_obs_names[i]): self.test_env.sigma_list_y[i] for i in range(self.test_env.entropy_obs_names.shape[0])}
                #     self.tb_logger.add_scalars('Entropy/max_entropy_env_Sigmas_y',  feed_dict, itr)
            

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
            ########################################################################################################################################################

        
            if self.env_name in ['multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
                feed_dict = {}
                o = torch.tensor(self.test_env._obs_lst[3].astype(np.float32)).to(self.ac.pi.device)
                left = torch.from_numpy(np.array([-1, 0]).astype(np.float32)).to(self.ac.pi.device)
                topleft = torch.from_numpy(np.array([-1, 0.75]).astype(np.float32)).to(self.ac.pi.device)
                bottomleft = torch.from_numpy(np.array([-1, -0.75]).astype(np.float32)).to(self.ac.pi.device)
                # topleft = torch.from_numpy(np.array([-1, 0.4364186]).astype(np.float32)).to(self.ac.pi.device)
                # bottomleft = torch.from_numpy(np.array([-1, -0.4364186]).astype(np.float32)).to(self.ac.pi.device)
                # topleft = torch.from_numpy(np.array([-0.4364186, 1]).astype(np.float32)).to(self.ac.pi.device)
                # bottomleft = torch.from_numpy(np.array([-0.4364186, -1]).astype(np.float32)).to(self.ac.pi.device)
                right = torch.from_numpy(np.array([1, 0]).astype(np.float32)).to(self.ac.pi.device)
                feed_dict[self.test_env.entropy_obs_names[3] + '_left'] = torch.min(self.ac.q1(o, left), self.ac.q2(o, left)).detach().cpu().item()
                feed_dict[self.test_env.entropy_obs_names[3] + '_topleft'] = torch.min(self.ac.q1(o, topleft), self.ac.q2(o, topleft)).detach().cpu().item()
                feed_dict[self.test_env.entropy_obs_names[3] + '_bottomleft'] = torch.min(self.ac.q1(o, bottomleft), self.ac.q2(o, bottomleft)).detach().cpu().item()
                feed_dict[self.test_env.entropy_obs_names[3] + '_right'] = torch.min(self.ac.q1(o, right), self.ac.q2(o, right)).detach().cpu().item()
                
                o = torch.tensor(self.test_env._obs_lst[2].astype(np.float32)).to(self.ac.pi.device)
                left = torch.from_numpy(np.array([-1, 0]).astype(np.float32)).to(self.ac.pi.device)
                topleft = torch.from_numpy(np.array([-1, 1]).astype(np.float32)).to(self.ac.pi.device)
                bottomleft = torch.from_numpy(np.array([-1, -1]).astype(np.float32)).to(self.ac.pi.device)
                # topleft = torch.from_numpy(np.array([-1, 0.5582575]).astype(np.float32)).to(self.ac.pi.device)
                # bottomleft = torch.from_numpy(np.array([-1, -0.5582575]).astype(np.float32)).to(self.ac.pi.device)
                # topleft = torch.from_numpy(np.array([-0.3273268, 1]).astype(np.float32)).to(self.ac.pi.device)
                # bottomleft = torch.from_numpy(np.array([-0.3273268, -1]).astype(np.float32)).to(self.ac.pi.device)
                feed_dict[self.test_env.entropy_obs_names[2] + '_topleft'] = torch.min(self.ac.q1(o, topleft), self.ac.q2(o, topleft)).detach().cpu().item()
                feed_dict[self.test_env.entropy_obs_names[2] + '_left'] = torch.min(self.ac.q1(o, left), self.ac.q2(o, left)).detach().cpu().item()
                feed_dict[self.test_env.entropy_obs_names[2] + '_bottomleft'] = torch.min(self.ac.q1(o, bottomleft), self.ac.q2(o, bottomleft)).detach().cpu().item()

                o = torch.tensor(self.test_env._obs_lst[4].astype(np.float32)).to(self.ac.pi.device)
                right = torch.from_numpy(np.array([1, 0]).astype(np.float32)).to(self.ac.pi.device)
                feed_dict[self.test_env.entropy_obs_names[4] + '_right'] = torch.min(self.ac.q1(o, right), self.ac.q2(o, right)).detach().cpu().item()

                self.tb_logger.add_scalars('Entropy/max_entropy_env_q_values',  feed_dict, itr)
                
            # if self.env_name == 'max-entropy-v0':
            #     for s in range(self.env_max_steps - 1):
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
        # investigating smoothness of the q-landscape by computing the 1st and 2nd order derivatives

        if self.ac.pi.actor in ['svgd_p0_pram'] and self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:
            feed_dict_mu_x = {}
            feed_dict_mu_y = {}
            feed_dict_sigma_x = {}
            feed_dict_sigma_y = {}

            for i in range(len(self.test_env._obs_lst)):
                
                o = torch.as_tensor(self.test_env._obs_lst[i], dtype=torch.float32).view(-1,1,self.test_env.observation_space.shape[0]).repeat(1,self.ac.pi.num_particles,1).view(-1,self.test_env.observation_space.shape[0]).to(self.ac.pi.device)
                actions, _ = self.ac(o, action_selection=None, with_logprob=False)
                mu, sigma = self.ac.pi.mu, self.ac.pi.sigma
                feed_dict_mu_x[self.test_env.entropy_obs_names[i]] = mu[0][0]
                feed_dict_mu_y[self.test_env.entropy_obs_names[i]] = mu[0][1]
                feed_dict_sigma_x[self.test_env.entropy_obs_names[i]] = sigma[0][0]
                feed_dict_sigma_y[self.test_env.entropy_obs_names[i]] = sigma[0][1]
            self.tb_logger.add_scalars('Entropy/mean_x',  feed_dict_mu_x, itr)
            self.tb_logger.add_scalars('Entropy/mean_y',  feed_dict_mu_y, itr)
            self.tb_logger.add_scalars('Entropy/std_x',  feed_dict_sigma_x, itr)
            self.tb_logger.add_scalars('Entropy/std_y',  feed_dict_sigma_y, itr)




        if self.ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
            diff_p0_p = list(map(lambda x: np.stack(x['diff_p0_p']).mean(), self.episodes_information))
            self.tb_logger.add_scalar('actions/particles_l2', np.array(diff_p0_p).mean(), itr)            
        actions_norm = list(map(lambda x: np.linalg.norm(np.stack(x['action']), axis=1), self.episodes_information))

        actions_norm_min = list(map(lambda x: x.min(), actions_norm))
        actions_norm_mean = list(map(lambda x: x.mean(), actions_norm))
        actions_norm_max = list(map(lambda x: x.max(), actions_norm))
        self.tb_logger.add_scalars('actions/actions_norm_detailed',  {'Mean ': np.mean(actions_norm_mean), 'Min': np.mean(actions_norm_min), 'Max': np.mean(actions_norm_max)  }, itr)
        self.tb_logger.add_scalars('actions/actions_norm_mean_only',  {'Mean ': np.mean(actions_norm_mean)}, itr)



        if self.ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
            self.tb_logger.add_scalars('actions/action_boundry', {'all_particles': self.boundary_action_counter/self.action_counter, 'actions': self.boundary_all_actions_counter/self.action_counter}, itr)
            
        q_score_ = list(map(lambda x: np.stack(x['q_score']), self.episodes_information))
        q_score_mean = list(map(lambda x: x.mean(), q_score_))
        q_score_min = list(map(lambda x: x.min(), q_score_))
        q_score_max = list(map(lambda x: x.max(), q_score_))
        self.tb_logger.add_scalars('smoothness/q_score_detailed',  {'Mean ': np.mean(q_score_mean), 'Min': np.mean(q_score_min), 'Max': np.mean(q_score_max)  }, itr)
        self.tb_logger.add_scalars('smoothness/q_score_mean_only',  {'Mean ': np.mean(q_score_mean)}, itr)
        q_score_averaged = []

        for i in ['_start', '_mid', '_end']:
            q_score_i = np.array(list(map(lambda x: x['q_score' + i], self.episodes_information)))
            q_score_averaged.append(np.mean(q_score_i[q_score_i != np.array(None)]))
        self.tb_logger.add_scalars('smoothness/q_score_averaged',  {'Start ': q_score_averaged[0], 'Mid': q_score_averaged[1], 'End': q_score_averaged[2] }, itr)

        if self.env_name in ['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles']:

            q_hess_ = list(map(lambda x: np.stack(x['q_hess']), self.episodes_information))
            q_hess_mean = list(map(lambda x: x.mean(), q_hess_))
            q_hess_min = list(map(lambda x: x.min(), q_hess_))
            q_hess_max = list(map(lambda x: x.max(), q_hess_))

            q_eigenvals = list(map(lambda x: np.stack(x['max_eigenval']), self.episodes_information))
            q_eigenvals_min = list(map(lambda x: x.min(), q_eigenvals))
            q_eigenvals_mean = list(map(lambda x: x.mean(), q_eigenvals))
            q_eigenvals_max = list(map(lambda x: x.max(), q_eigenvals))
            q_eigenvals_abs_min = list(map(lambda x: np.absolute(x).min(), q_eigenvals))
            q_eigenvals_abs_mean = list(map(lambda x: np.absolute(x).mean(), q_eigenvals))
            q_eigenvals_abs_max = list(map(lambda x: np.absolute(x).max(), q_eigenvals))

        

            self.tb_logger.add_scalars('smoothness/q_hess', {'Mean ': np.mean(q_hess_mean), 'Min': np.mean(q_hess_min), 'Max': np.mean(q_hess_max)  }, itr)
            self.tb_logger.add_scalars('smoothness/q_eigenvals', {'Mean ': np.mean(q_eigenvals_mean), 'Min': np.mean(q_eigenvals_min), 'Max': np.mean(q_eigenvals_max)  }, itr)
            self.tb_logger.add_scalars('smoothness/q_eigenvals_abs', {'Mean ': np.mean(q_eigenvals_abs_mean), 'Min': np.mean(q_eigenvals_abs_min), 'Max': np.mean(q_eigenvals_abs_max)  }, itr)
            
            q_hess_averaged = []


            for i in ['_start', '_mid', '_end']:
                q_hess_i = np.array(list(map(lambda x: x['q_hess' + i], self.episodes_information)))
                q_hess_averaged.append(np.mean(q_hess_i[q_hess_i != np.array(None)]))
            self.tb_logger.add_scalars('smoothness/q_hess_averaged', {'Start ': q_hess_averaged[0], 'Mid': q_hess_averaged[1], 'End': q_hess_averaged[2] }, itr)



        



        # 
        expected_rewards = list(map(lambda x: x['expected_reward'], self.episodes_information))
        episode_length = list(map(lambda x: x['episode_length'], self.episodes_information))

        # import pdb; pdb.set_trace()
        self.tb_logger.add_scalars('Test_EpRet/return_detailed',  {'Mean ': np.mean(expected_rewards), 'Min': np.min(expected_rewards), 'Max': np.max(expected_rewards) }, itr)
        self.tb_logger.add_scalars('Test_EpRet/return_mean_only',  {'Mean ': np.mean(expected_rewards)}, itr)
        self.tb_logger.add_scalar('Test_EpLen', np.mean(episode_length) , itr)
        
    def reset(self,):
        self.episodes_information = []
        self.boundary_action_counter = 0 
        self.boundary_all_actions_counter = 0 
        self.action_counter = 0 



    


    


#############################################################################################################################################################################################################################
#############################################################################################################################################################################################################################
#############################################################################################################################################################################################################################
#############################################################################################################################################################################################################################
#############################################################################################################################################################################################################################
#############################################################################################################################################################################################################################
#############################################################################################################################################################################################################################





    # def entorpy_landscape(self, fig_path):
    #     self.ac.pi.num_particles = 10 ####### for sac
    #     self.ac.pi.batch_size = 50
    #     # if not self.ac.pi.actor=='sac':
    #         # self.ac.pi.Kernel.num_particles = 100
    #         # self.ac.pi.identity = torch.eye(self.ac.pi.num_particles).to(self.ac.pi.device)
    #     # resolution = 0.5
    #     resolution = 0.05
    #     self._ax = []
    #     x_lim = (-7, 7)
    #     y_lim = (-7, 7)
    #     self.entropy_fig = plt.figure(figsize=(10.5, 10), constrained_layout=True) 
    #     self._ax.append(plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1))
    #     self._ax[0].axis('equal')
    #     self._ax[0].set_xlim(x_lim)
    #     self._ax[0].set_ylim(y_lim)
    #     self._ax[0].autoscale_view('tight')
    #     self._ax[0].set_title('Entropy Heatmap', fontsize=18)
    #     # self._ax[0].set_xlabel('x', fontsize=13)
    #     # self._ax[0].set_ylabel('y', fontsize=13)
    #     self._ax[0].xaxis.set_tick_params(labelsize=13)
    #     self._ax[0].yaxis.set_tick_params(labelsize=13)
    #     x_min, x_max = tuple(np.round(1.1 * np.array(x_lim), 3))
    #     y_min, y_max = tuple(np.round(1.1 * np.array(x_lim), 3))



    #     # self._obs_lst = np.array([
    #     #     [0,0],
    #     #     [1, 0],
    #     #     [2, 0],
    #     #     [2.5,2.5],
    #     #     [4, 0],
    #     #     [2.5,-2.5],
    #     #     [1.5,-0.5],
    #     #     [1.5,0.5],
    #     #     [3,1.5],
    #     #     [3,-1.5],
    #     #     [3, 0],
    #     #     [3.5, 1],
    #     #     [3.5, -1],
    #     # ])
    #     # self.entropy_obs_names_plotting = np.array(['$s_a$', '$s_b$', '$s_c$', '$s_d$', '$s_e$', '$s_f$', '$s_g$', '$s_h$', '$s_i$', '$s_j$', '$s_k$', '$s_l$', '$s_l$'])


    #     # o = torch.as_tensor(self._obs_lst, dtype=torch.float32).to(self.ac.pi.device).view(-1,1,self.test_env.observation_space.shape[0]).repeat(1,self.ac.pi.num_particles,1).view(-1,self.test_env.observation_space.shape[0])

    #     # self.entropy_list = []
    #     # for i in range(100):
    #     #     a, log_p2 = self.ac(o, action_selection=self.ac.pi.test_action_selection, with_logprob=True)
    #     # self.entropy_list.append(list(np.round(-log_p2.cpu().detach().numpy(), 2)))
    #     # self.entropy_list = np.array(self.entropy_list).mean(axis=0)


    #     # for i in range(len(self._obs_lst)):
    #     # for i in range(len(self.test_env._obs_lst)):
    #     #     self._ax[0].scatter(self.test_env._obs_lst[i, 0], self.test_env._obs_lst[i, 1], c='white', marker=self.test_env.entropy_obs_names_plotting[i], s=100, zorder=3)
    #     #     self._ax[0].annotate(str(self.test_env.entropy_list[i]), (self.test_env._obs_lst[i,0] - 0.4, self.test_env._obs_lst[i,1] + 0.2), fontsize=10, color='white', zorder=3)
    #         # self._ax[0].annotate(str(self.entropy_list[i]), (self._obs_lst[i,0] - 0.4, self._obs_lst[i,1] + 0.2), fontsize=10, color='white', zorder=3)
    #         # self._ax_lst[0].annotate(self.test_env.entropy_obs_names_plotting[i], (self.test_env._obs_lst[i,0] - 0.25, self.test_env._obs_lst[i,1] + 0.2), fontsize=18, color='#003f40', zorder=2)
        



    #     x_range = np.arange(x_min, x_max, resolution)
    #     y_range = np.arange(y_min, y_max, resolution)
    #     X, Y = np.meshgrid(x_range, y_range)
    #     X_shape = X.shape
    #     X__ = X.reshape(-1, 1)
    #     Y__ = Y.reshape(-1, 1)
    #     input_ = np.concatenate((X__, Y__), axis=1)
        
    #     entropy = []
    #     idx_start = 0
    #     idx_end = self.ac.pi.batch_size
    #     pbar = tqdm(total=len(input_) + 1)
    #     while idx_start < len(input_):
    #         o = input_[idx_start:idx_end, :]
    #         o = torch.tensor(o, dtype=torch.float32).view(-1,1,self.train_env.observation_space.shape[0]).repeat(1,self.ac.pi.num_particles,1).view(-1,self.train_env.observation_space.shape[0]).to(self.ac.pi.device)
    #         log_p = []
    #         for _ in range(10):
    #             a, logp_pi = self.ac(o, action_selection=self.ac.pi.test_action_selection, with_logprob=True)
    #             if self.ac.pi.actor=='sac':
    #                 logp_pi = logp_pi.view(-1,self.ac.pi.num_particles).mean(dim=1)
    #             log_p.append(list(-logp_pi.detach().cpu().numpy()))
    #         entropy += np.array(log_p).mean(0).tolist()
    #         idx_start = idx_end
    #         idx_end = min(idx_end + self.ac.pi.batch_size, len(input_))
    #         pbar.update(idx_end - idx_start)
    #     pbar.close()

    #     entropy = np.array(entropy).reshape(X_shape)
    #     np.save(fig_path + '/entropy_matrix', entropy)
        
    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_02.npy')
    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_1.npy')
    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_5.npy') ################
    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_10.npy')
    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_15.npy')
    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_20.npy')

    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_sac_02.npy')
    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_sac_1.npy')
    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_sac_10.npy')
    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_sac_15.npy')
    #     # entropy = np.load('./STAC/multi_goal_plots_/entropy_matrix_sac_20.npy')

    #     min_entropy = entropy.min()
    #     max_entropy = entropy.max()
    #     print('##################### ', min_entropy, max_entropy)
    #     extent = (x_min, x_max, y_min, y_max)
    #     # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, norm=mpl_colors.PowerNorm(gamma = 0.7), alpha=.9, interpolation='bilinear', extent=extent)
    #     # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent)
    #     cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent)
    #     # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent, vmin=-0.12, vmax=0.1)
    #     # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent, vmin=-0.8, vmax=1)
    #     # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent, vmin=-2.7, vmax=-0.5)
    #     # cmap = self._ax[0].imshow(entropy, cmap=plt.cm.viridis, extent=extent, vmin=0.6)
    #     self._ax[0].plot(self.test_env.goal_positions[:, 0], self.test_env.goal_positions[:, 1], 'ro')
    #     self._ax[0].plot(np.array([0]), np.array([0]), 'bo')
    #     range_ = np.arange(min_entropy, max_entropy, 0.1)[1::2]
    #     # range_ = np.arange(-1.5, 1.5, 0.1)[1::2]
    #     # plt.colorbar(mappable=cmap)
    #     self.entropy_fig.colorbar(mappable=cmap, ticks=range_)


    #     plt.savefig(fig_path + '/entropy_landscape.' + self.plot_format)   
    #     plt.savefig(fig_path + '/entropy_landscape.' + 'png')   
    #     plt.close()




    # def actions_3d_scatter_plot(self, itr, fig_path):
    #     grid_size=(2,2)
    #     fig_size=(10, 10)
    #     labels = [['$a_1$', '$a_2$'], ['$a_2$', '$a_3$'], ['$a_1$', '$a_3$']]
        
    #     obs = self.episodes_information[0]['observations']
    #     particles = np.array(self.episodes_information[0]['x_t']) #### (n_steps, np, ad)
        
        

    #     for i in range(min(500, len(obs))-1):
    #     # for i in range(len(particles)):
    #         _ax_lst = []
    #         fig_env = plt.figure(figsize=fig_size) 
    #         ax = plt.subplot2grid(grid_size, (0,0),  projection='3d')
    #         ax.set_xlabel('$a_1$')
    #         ax.set_ylabel('$a_2$')
    #         ax.set_zlabel('$a_3$')
    #         ax.set_xlim((-1, 1))
    #         ax.set_ylim((-1, 1))
    #         ax.set_zlim((-1, 1))
    #         ax.set_box_aspect([2, 2, 2])  # equal aspect ratio
    #         _ax_lst.append(ax)
    #         # _ax_lst[0].view_init(0, 90, 0)
    #         # plt.margins(2)
    #         plt.subplots_adjust(right=0.9)
    #         fig_env.tight_layout(pad=10.0)
    #         for j in range(1, 4):
    #             ax = plt.subplot2grid(grid_size, (j//2,j%2))
    #             ax.set_xlabel(labels[j-1][0])
    #             ax.set_ylabel(labels[j-1][1])
    #             ax.set_xlim((-1, 1))
    #             ax.set_ylim((-1, 1))
    #             ax.grid(True)
    #             _ax_lst.append(ax)



    #         # data = np.random.rand(3, 100)
    #         x, y, z = particles[i][-1][:, 0], particles[i][-1][:, 1], particles[i][-1][:, 2]  # for show
    #         # if  (x < -1).any() or (x > 1).any() or (y < -1).any() or (y > 1).any() or (z < -1).any() or (z > 1).any():
    #         #     print('#################### Error ####################')
    #         o = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).repeat(particles[i][-1].shape[0], 1).to(self.ac.pi.device).detach()
    #         a = torch.tensor(particles[i][-1], dtype=torch.float32).to(self.ac.pi.device).detach()
    #         q1_value = self.ac.q1(o,a)
    #         q2_value = self.ac.q2(o,a)
    #         q_values = np.round(torch.min(q1_value, q2_value).detach().cpu().numpy(), 3)
    #         # c = np.arange(len(x)) / len(x)  # create some colours
    #         p = _ax_lst[0].scatter(x, y, z, c=q_values, cmap=plt.cm.viridis, alpha=1)
    #         _ax_lst[1].scatter(x, y, c=q_values, cmap=plt.cm.viridis, alpha=1)
    #         _ax_lst[2].scatter(y, z, c=q_values, cmap=plt.cm.viridis, alpha=1)
    #         _ax_lst[3].scatter(x, z, c=q_values, cmap=plt.cm.viridis, alpha=1)
    #         cbar_ax = fig_env.add_axes([0.88, 0.15, 0.05, 0.7])
    #         fig_env.colorbar(p, cax=cbar_ax)
    #         # fig_env.colorbar(p, ax=_ax_lst[0])

    #         # import pdb; pdb.set_trace()
    #         for k in range(particles.shape[2]):
    #             # print('########## ', str(i), particles[i, :, k, :])
    #             x_p, y_p, z_p = particles[i, :, k, 0], particles[i, :, k, 1], particles[i, :, k, 2]
    #             _ax_lst[0].plot(x_p, y_p, z_p, c='red')
    #             _ax_lst[1].plot(x_p, y_p, c='red')
    #             _ax_lst[2].plot(y_p, z_p, c='red')
    #             _ax_lst[3].plot(x_p, z_p, c='red')
                
            
    #         # np.ptp(particles[0][0])
            
    #         # ax.set_box_aspect([np.ptp(particles[i][:, j]) for j in range(len(particles[i][0]))])  # equal aspect ratio

    #         # fig.colorbar(p, ax=ax)

    #         # ax.view_init(0, 0, 0)
    #         # ax.view_init(-90, 0, 0)

    #         # plt.savefig(fig_path + '/actions_plot_step_'+ str(i) + '.' + self.plot_format)   
    #         plt.savefig(fig_path + '/actions_plot_step_'+ str(i) + '.' + 'png')   
    #         plt.close()
    #         # import pdb; pdb.set_trace()


    # def create_slices_plot(self, itr, fig_path, fig_size=(14, 7)):
    #     n_plots = 3
    #     _n_samples = 10
    #     # grid_size=grid_size
    #     grid_size=(1,3)
    #     # _obs_lst = np.array([1.1373217, 0.01862574, -0.1822641, -0.27725494, 0.4657382, 0.04975163, -0.01738176, 0.00199971, -0.0298201, -0.04012394, 0.07060421])
    #     obs = self.episodes_information[0]['observations']
    #     _act_lst = np.array([-0.2, 0, 0.2])
    #     ###### Setup the environment plot ######
    #     for k in range(min(500, len(obs))-1):
    #         _ax_lst = []
    #         fig_env = plt.figure(figsize=fig_size, constrained_layout=True) 
    #         for i in range(n_plots):
    #             ax = plt.subplot2grid(grid_size, (i//3,i%3))
    #             ax.xaxis.set_tick_params(labelsize=15)
    #             ax.yaxis.set_tick_params(labelsize=15)
    #             ax.set_xlim((-1, 1))
    #             ax.set_ylim((-1, 1))
    #             ax.grid(True)
    #             _ax_lst.append(ax)
    #         _line_objects = list()
    #         xs = np.linspace(-1, 1, 50)
    #         ys = np.linspace(-1, 1, 50)
    #         xgrid, ygrid = np.meshgrid(xs, ys)
    #         a_ = np.concatenate((np.expand_dims(xgrid.ravel(), -1), np.expand_dims(ygrid.ravel(), -1)), -1)
            
    #         for i in range(len(_act_lst)):
    #             a = torch.from_numpy(np.concatenate((np.full((a_.shape[0], 1), _act_lst[i]), a_), axis=1).astype(np.float32)).to(self.ac.pi.device)
    #             o = torch.Tensor(obs[k]).repeat([a_.shape[0],1]).to(self.ac.pi.device)
    #             with torch.no_grad():
    #                 qs = self.ac.q1(o.to(self.ac.pi.device), a).cpu().detach().numpy()
    #             qs = qs.reshape(xgrid.shape)
    #             cs = _ax_lst[i].contour(xgrid, ygrid, qs, 20)
    #             _line_objects += cs.collections
    #             _line_objects += _ax_lst[i].clabel(
    #                 cs, inline=1, fontsize=10, fmt='%.2f')

    #         if ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
    #             num_particles_tmp = self.ac.pi.num_particles
    #             self.ac.pi.num_particles = _n_samples
    #             self.ac.pi.Kernel.num_particles = self.ac.pi.num_particles
    #             self.ac.pi.identity = torch.eye(self.ac.pi.num_particles).to(self.ac.pi.device)
            
    #         for i in range(len(_act_lst)):
    #             if ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
    #                 # self.ac.pi.num_particles = _n_samples
    #                 # self.ac.pi.Kernel.num_particles = self.ac.pi.num_particles
    #                 # self.ac.pi.identity = torch.eye(self.ac.pi.num_particles).to(self.ac.pi.device)
    #                 o = torch.as_tensor(obs[k], dtype=torch.float32).view(-1,1,self.test_env.observation_space.shape[0]).repeat(1,self.ac.pi.num_particles,1).view(-1,self.test_env.observation_space.shape[0]).to(self.ac.pi.device)
    #             else:
    #                 o = torch.as_tensor(obs[k], dtype=torch.float32).repeat([_n_samples,1]).to(self.ac.pi.device)
    #             actions, _ = self.ac(o, action_selection=None, with_logprob=False)
    #             actions = actions.cpu().detach().numpy().squeeze()
    #             # if ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
    #             # print(actions)
    #             # else:
    #             x, y = actions[:, 1], actions[:, 2]

    #             _ax_lst[i].set_title('action_dim_' + str(_act_lst[i]), fontsize=15)
    #             _line_objects += _ax_lst[i].plot(x, y, 'b*')
    #         # self.ac.pi.num_particles = num_particles_tmp
    #         # self.ac.pi.Kernel.num_particles = self.ac.pi.num_particles
    #         # self.ac.pi.identity = torch.eye(self.ac.pi.num_particles).to(self.ac.pi.device)
    #         if ac.pi.actor in ['svgd_nonparam', 'svgd_p0_pram']:
    #             self.ac.pi.num_particles = num_particles_tmp
    #             self.ac.pi.Kernel.num_particles = self.ac.pi.num_particles
    #             self.ac.pi.identity = torch.eye(self.ac.pi.num_particles).to(self.ac.pi.device)

    #         plt.savefig(fig_path+ '/slices_fig_step_' + str(k) + '.png')   
    #         plt.close()
    #         # print()
    #         # plt.savefig(fig_path+ '/slices_fig_' + str(itr) + '.' + self.plot_format)   


    # def collect_svgd_data(self, exploration, observation, particles=None, logp=None):
    #     if self.train_env.ep_len == 0:
    #         self.episodes_information_svgd.append({
    #             'step': [], 
    #             'exploration': [],
    #             'entropy': [], 
    #             'observations': [], 
    #             'particles': [],
    #             'gradients': [],
    #             'svgd_lr': [],
    #         })
        

    #     self.episodes_information_svgd[-1]['step'].append(self.train_env.ep_len)
    #     self.episodes_information_svgd[-1]['observations'].append(list(observation))
    #     self.episodes_information_svgd[-1]['exploration'].append(exploration)
    #     if not exploration:
    #         self.episodes_information_svgd[-1]['entropy'].append(-logp.detach().cpu().item())
    #         self.episodes_information_svgd[-1]['particles'].append(self.ac.pi.x_t)
    #         self.episodes_information_svgd[-1]['gradients'].append(self.ac.pi.phis)
    #         self.episodes_information_svgd[-1]['svgd_lr'].append(self.ac.pi.svgd_lr)
    #     else:
    #         self.episodes_information_svgd[-1]['particles'].append(list(particles))

        # print(self.episodes_information_svgd[-1]['exploration'])

    # def plot_svgd_particles_q_contours(self, fig_path):
    #     self._ax_lst = []
    #     _n_samples = 100
    #     _obs_lst = self.episodes_information_svgd[-1]['observations']
    #     # for i in range(len(_obs_lst)):
    #     for episode_step in range(1):

    #         self.fig_env = plt.figure(figsize=(4, 4), constrained_layout=True) 
    #         self._ax_lst.append(plt.subplot2grid((1,1), (0,0), colspan=3, rowspan=3))
    #         self._ax_lst[0].set_xlim((-1, 1))
    #         self._ax_lst[0].set_ylim((-1, 1))
    #         self._ax_lst[0].set_title('SVGD Particles Plot')
    #         self._ax_lst[0].set_xlabel('x')
    #         self._ax_lst[0].set_ylabel('y')
    #         self._ax_lst[0].grid(True)
    #         self._line_objects = []


    #         xs = np.linspace(-1, 1, 50)
    #         ys = np.linspace(-1, 1, 50)
    #         xgrid, ygrid = np.meshgrid(xs, ys)
    #         a = np.concatenate((np.expand_dims(xgrid.ravel(), -1), np.expand_dims(ygrid.ravel(), -1)), -1)
    #         a = torch.from_numpy(a.astype(np.float32)).to(self.ac.pi.device)
    #         o = torch.Tensor(_obs_lst[episode_step]).repeat([a.shape[0],1]).to(self.ac.pi.device)
    #         with torch.no_grad():
    #             qs = self.ac.q1(o.to(self.ac.pi.device), a).cpu().detach().numpy()
    #         qs = qs.reshape(xgrid.shape)
    #         cs = self._ax_lst[0].contour(xgrid, ygrid, qs, 20)
    #         self._line_objects += cs.collections
    #         self._line_objects += self._ax_lst[0].clabel(
    #             cs, inline=1, fontsize=10, fmt='%.2f')

    #         o = _obs_lst[episode_step]
    #         actions = np.array(self.episodes_information_svgd[-1]['particles'][episode_step])
    #         entropy = self.episodes_information_svgd[-1]['entropy'][episode_step]
    #         # actions = actions.cpu().detach().numpy().squeeze()
    #         # if self.episodes_information_svgd[-1]['exploration'][episode_step]:
    #         # else:
    #         no_of_colors=10
    #         # colors = ["#"+''.join([np.random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(no_of_colors)]
    #         for particle_idx in range(actions.shape[1]):
    #             x, y = actions[:, particle_idx, 0], actions[:, particle_idx, 1]
    #             color = (0.99, 0.5, np.random.random())
    #             self._ax_lst[0].title.set_text(str([round(_obs_lst[episode_step][0], 2), round(_obs_lst[episode_step][1], 2)]) + ' -- Entropy: ' + str(round(entropy, 2)))
    #             if self.episodes_information_svgd[-1]['exploration'][episode_step]:
    #                 self._line_objects += self._ax_lst[0].plot(x, y, c=color + '*')
    #             else:
    #                 self._line_objects += self._ax_lst[0].plot(x, y, c=color)
    #         plt.savefig(fig_path+ '/svgd_episode_' + str(episode_step) + '_step_' + str(self.episodes_information_svgd[-1]['step'][episode_step]) + '.' + self.plot_format)

    def entropy_plot(self):
        if self.ac.pi.actor in ['svgd_p0_pram'] and self.env_name in ['Multigoal', 'multigoal-max-entropy']:
            log_p = []
            term1 = []
            term2 = []
            logp_normal = []
            logp_svgd = []
            logp_tanh = []
            mu = []
            sigma = []
            # logp_toy_line1 = []
            # logp_toy_line2 = []
            # logp_toy_line4 = []
            # logp_wrong = []
            for indx, i in enumerate([0, 10, 25]):
                if len(self.episodes_information[-1]['log_p']) > i+1:
                    log_p.append(self.episodes_information[-1]['log_p'][i])
                    mu.append(np.absolute(self.episodes_information[-1]['mu'][i]).mean())
                    sigma.append(np.absolute(self.episodes_information[-1]['sigma'][i]).mean())

                    # term1.append(self.episodes_information[-1]['term1'][i])
                    # term2.append(self.episodes_information[-1]['term2'][i])
                    logp_normal.append(self.episodes_information[-1]['logp_normal'][i])
                    logp_svgd.append(self.episodes_information[-1]['logp_svgd'][i])
                    logp_tanh.append(self.episodes_information[-1]['logp_tanh'][i])
                    # logp_toy_line1.append(self.episodes_information[-1]['logp_toy_line1'][i])
                    # logp_toy_line2.append(self.episodes_information[-1]['logp_toy_line2'][i])
                    # logp_toy_line4.append(self.episodes_information[-1]['logp_toy_line4'][i])
                    # logp_wrong.append(self.episodes_information[-1]['logp_wrong'][i])
            if len(log_p) == 3:
                # print('############################# ', (len(self.episodes_information)- 1))
                # self.add_scalars(tb_path='Entropy/entropy', value={'step_0': log_p[0], 'step_10': log_p[1], 'step_25': log_p[2]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/term1', value={'step_0': term1[0], 'step_10': term1[1], 'step_25': term1[2]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/term2', value={'step_0': term2[0], 'step_10': term2[1], 'step_25': term2[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_0': logp_normal[0], 'step_10': logp_normal[1], 'step_25': logp_normal[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_0': logp_svgd[0], 'step_10': logp_svgd[1], 'step_25': logp_svgd[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_0': logp_tanh[0], 'step_10': logp_tanh[1], 'step_25': logp_tanh[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu', value={'step_0': mu[0], 'step_10': mu[1], 'step_25': mu[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma', value={'step_0': sigma[0], 'step_10': sigma[1], 'step_25': sigma[2]}, itr=self.episode_counter)

                # self.add_scalars(tb_path='Entropy/logp_toy_line1', value={'step_0': logp_toy_line1[0], 'step_10': logp_toy_line1[1], 'step_25': logp_toy_line1[2]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line2', value={'step_0': logp_toy_line2[0], 'step_10': logp_toy_line2[1], 'step_25': logp_toy_line2[2]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line4', value={'step_0': logp_toy_line4[0], 'step_10': logp_toy_line4[1], 'step_25': logp_toy_line4[2]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_wrong', value={'step_0': logp_wrong[0], 'step_10': logp_wrong[1], 'step_25': logp_wrong[2]}, itr=self.episode_counter)


            elif len(log_p) == 2:
                # self.add_scalars(tb_path='Entropy/entropy', value={'step_0': log_p[0], 'step_10': log_p[1]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/term1', value={'step_0': term1[0], 'step_10': term1[1]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/term2', value={'step_0': term2[0], 'step_10': term2[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_0': logp_normal[0], 'step_10': logp_normal[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_0': logp_svgd[0], 'step_10': logp_svgd[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_0': logp_tanh[0], 'step_10': logp_tanh[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu', value={'step_0': mu[0], 'step_10': mu[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma', value={'step_0': sigma[0], 'step_10': sigma[1]}, itr=self.episode_counter)

                # self.add_scalars(tb_path='Entropy/logp_toy_line1', value={'step_0': logp_toy_line1[0], 'step_10': logp_toy_line1[1]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line2', value={'step_0': logp_toy_line2[0], 'step_10': logp_toy_line2[1]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line4', value={'step_0': logp_toy_line4[0], 'step_10': logp_toy_line4[1]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_wrong', value={'step_0': logp_wrong[0], 'step_10': logp_wrong[1]}, itr=self.episode_counter)

            elif len(log_p) == 1:
                # self.add_scalars(tb_path='Entropy/entropy', value={'step_0': log_p[0]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/term1', value={'step_0': term1[0]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/term2', value={'step_0': term2[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_0': logp_normal[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_0': logp_svgd[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_0': logp_tanh[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu', value={'step_0': mu[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma', value={'step_0': sigma[0]}, itr=self.episode_counter)

                # self.add_scalars(tb_path='Entropy/logp_toy_line1', value={'step_0': logp_toy_line1[0]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line2', value={'step_0': logp_toy_line2[0]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_toy_line4', value={'step_0': logp_toy_line4[0]}, itr=self.episode_counter)
                # self.add_scalars(tb_path='Entropy/logp_wrong', value={'step_0': logp_wrong[0]}, itr=self.episode_counter)

            self.episode_counter += 1


    def entropy_plot_v2(self):
        if self.ac.pi.actor in ['svgd_p0_pram'] and self.env_name in ['Hopper-v2']:
            log_p = []
            logp_normal = []
            logp_svgd = []
            logp_tanh = []
            mu_x = []
            sigma_x = []
            mu_y = []
            sigma_y = []
            mu_z = []
            sigma_z = []

            for indx, i in enumerate([10, 100, 500]):
                if len(self.episodes_information[-1]['log_p']) > i+1:
                    log_p.append(self.episodes_information[-1]['log_p'][i])
                    mu_x.append(self.episodes_information[-1]['mu'][i][0][0])
                    sigma_x.append(self.episodes_information[-1]['sigma'][i][0][0])
                    mu_y.append(self.episodes_information[-1]['mu'][i][0][1])
                    sigma_y.append(self.episodes_information[-1]['sigma'][i][0][1])
                    mu_z.append(self.episodes_information[-1]['mu'][i][0][2])
                    sigma_z.append(self.episodes_information[-1]['sigma'][i][0][2])


                    logp_normal.append(self.episodes_information[-1]['logp_normal'][i])
                    logp_svgd.append(self.episodes_information[-1]['logp_svgd'][i])
                    logp_tanh.append(self.episodes_information[-1]['logp_tanh'][i])

            if len(log_p) == 3:

                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_10': logp_normal[0], 'step_100': logp_normal[1], 'step_500': logp_normal[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_10': logp_svgd[0], 'step_100': logp_svgd[1], 'step_500': logp_svgd[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_10': logp_tanh[0], 'step_100': logp_tanh[1], 'step_500': logp_tanh[2]}, itr=self.episode_counter)
                
                self.add_scalars(tb_path='Entropy/mu_x', value={'step_10': mu_x[0], 'step_100': mu_x[1], 'step_500': mu_x[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma_x', value={'step_10': sigma_x[0], 'step_100': sigma_x[1], 'step_500': sigma_x[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu_y', value={'step_10': mu_y[0], 'step_100': mu_y[1], 'step_500': mu_y[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma_y', value={'step_10': sigma_y[0], 'step_100': sigma_y[1], 'step_500': sigma_y[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu_z', value={'step_10': mu_z[0], 'step_100': mu_z[1], 'step_500': mu_z[2]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma_z', value={'step_10': sigma_z[0], 'step_100': sigma_z[1], 'step_500': sigma_z[2]}, itr=self.episode_counter)


            elif len(log_p) == 2:

                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_10': logp_normal[0], 'step_100': logp_normal[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_10': logp_svgd[0], 'step_100': logp_svgd[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_10': logp_tanh[0], 'step_100': logp_tanh[1]}, itr=self.episode_counter)
                
                self.add_scalars(tb_path='Entropy/mu_x', value={'step_10': mu_x[0], 'step_100': mu_x[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma_x', value={'step_10': sigma_x[0], 'step_100': sigma_x[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu_y', value={'step_10': mu_y[0], 'step_100': mu_y[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma_y', value={'step_10': sigma_y[0], 'step_100': sigma_y[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu_z', value={'step_10': mu_z[0], 'step_100': mu_z[1]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma_z', value={'step_10': sigma_z[0], 'step_100': sigma_z[1]}, itr=self.episode_counter)




            elif len(log_p) == 1:
                
                self.add_scalars(tb_path='Entropy/logp_normal', value={'step_10': logp_normal[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_svgd', value={'step_10': logp_svgd[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/logp_tanh', value={'step_10': logp_tanh[0]}, itr=self.episode_counter)
                
                self.add_scalars(tb_path='Entropy/mu_x', value={'step_10': mu_x[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma_x', value={'step_10': sigma_x[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu_y', value={'step_10': mu_y[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma_y', value={'step_10': sigma_y[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/mu_z', value={'step_10': mu_z[0]}, itr=self.episode_counter)
                self.add_scalars(tb_path='Entropy/sigma_z', value={'step_10': sigma_z[0]}, itr=self.episode_counter)



            self.episode_counter += 1
