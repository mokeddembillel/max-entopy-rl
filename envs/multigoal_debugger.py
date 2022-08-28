from gym.utils import EzPickle
from gym import spaces
#from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import Env
from spinup.algos.pytorch.sac.plotter import svgd_plotting 
import numpy as np
import matplotlib.pyplot as plt



class Debuger_MultiGoal():
    def __init__(self,writer): 
        init_state = torch.tensor([0.0,0.0]).to(device)
        a_up = torch.tensor([0.0,0.7]).to(device)
        a_down = torch.tensor([0.0,-0.7]).to(device)
        a_left = torch.tensor([-0.7,0.0]).to(device)
        a_right = torch.tensor([0.7,0.0]).to(device)
        
    @static
    def get_debugging_metrics():
        q_up = ac.q1(init_state,a_up).detach()
        q_down = ac.q1(init_state,a_down).detach()
        q_left = ac.q1(init_state,a_left).detach()
        q_right = ac.q1(init_state,a_right).detach()
        writer.add_scalars('init_state/q_val',{'q_up': q_up, 'q_down':q_down, 'q_left':q_left, 'q_right':q_right}, itr)

        def compute_q(init_s, action):
            return ac.q1(init_s,action)

        # add curvature and gradient
        a_up_ = a_up.requires_grad_(True)
        init_state_ = init_state.requires_grad_(True)
        grad_up_ = torch.autograd.grad(ac.q1(init_state_,a_up_), a_up_,retain_graph=True, create_graph=True)[0]
        grad_up = torch.abs(grad_up_).mean()
        hess_up = ((torch.abs(torch.autograd.grad(grad_up_[0],a_up_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_up_[1],a_up_,retain_graph=True)[0])).sum()/4).detach()
        #hess_up = torch.autograd.functional.hessian(compute_q, (init_state_,a_up_) )
        #hess_up = torch.abs(hess_up[1][1]).mean().detach()
        ###
        a_down_ = a_down.requires_grad_(True)
        grad_down_ = torch.autograd.grad(ac.q1(init_state_,a_down_), a_down_,retain_graph=True, create_graph=True)[0]
        grad_down = torch.abs(grad_down_).mean()
        hess_down = ((torch.abs(torch.autograd.grad(grad_down_[0],a_down_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_down_[1],a_down_,retain_graph=True)[0])).sum()/4).detach()
        #hess_down = torch.autograd.functional.hessian(compute_q, (init_state_,a_down_) )
        #hess_down = torch.abs(hess_down[1][1]).mean().detach()
        ###
        a_left_ = a_left.requires_grad_(True)
        grad_left_ = torch.autograd.grad(ac.q1(init_state_,a_left_), a_left_,retain_graph=True, create_graph=True)[0]
        grad_left = torch.abs(grad_left_).mean()
        hess_left = ((torch.abs(torch.autograd.grad(grad_left_[0],a_left_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_left_[1],a_left_,retain_graph=True)[0])).sum()/4).detach()
        #hess_left = torch.autograd.functional.hessian(compute_q, (init_state_,a_left_) )
        #hess_left = torch.abs(hess_left[1][1]).mean().detach()
        ###
        a_right_ = a_right.requires_grad_(True)
        grad_right_ = torch.autograd.grad(ac.q1(init_state_,a_right_), a_right_,retain_graph=True, create_graph=True)[0]
        grad_right = torch.abs(grad_right_).mean()
        hess_right = ((torch.abs(torch.autograd.grad(grad_right_[0],a_right_,retain_graph=True)[0])+torch.abs(torch.autograd.grad(grad_right_[1],a_right_,retain_graph=True)[0])).sum()/4).detach()
        #hess_right = torch.autograd.functional.hessian(compute_q, (init_state_,a_right_) )
        #hess_right = torch.abs(hess_right[1][1]).mean().detach()
        
        #import pdb; pdb.set_trace()  
        writer.add_scalars('init_state/hessian',{'hess_up': hess_up, 'hess_down':hess_down, 'hess_left':hess_left, 'hess_right':hess_right}, itr)
        writer.add_scalars('init_state/grad',{'grad_up': grad_up, 'grad_down':grad_down, 'grad_left':grad_left, 'grad_right':grad_right}, itr)
        # compute the variance of running svgd
        num_samples = 100

        s_up = a_up.view(-1,1,a_up.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_up.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_up),2)).to(device)
        a_svgd_up, _, _ = ac.svgd_sampler(s_up, a_rand.detach()) 
        q_svgd_up = ac.q1(s_up,a_svgd_up).detach()
        q_svgd_up_var = torch.var(q_svgd_up)
        
        s_down = a_down.view(-1,1,a_down.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_down.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_down),2)).to(device)
        a_svgd_down, _, _ = ac.svgd_sampler(s_down, a_rand.detach()) 
        q_svgd_down = ac.q1(s_down,a_svgd_down).detach()
        q_svgd_down_var = torch.var(q_svgd_down)
        
        s_left = a_left.view(-1,1,a_left.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_left.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_left),2)).to(device)
        a_svgd_left, _, _ = ac.svgd_sampler(s_left, a_rand.detach()) 
        q_svgd_left = ac.q1(s_left,a_svgd_left).detach()
        q_svgd_left_var = torch.var(q_svgd_left)
        
        s_right = a_left.view(-1,1,a_right.size()[-1]).repeat(1,num_samples*num_svgd_particles,1).view(-1,a_right.size()[-1])
        a_rand = torch.normal(0, 1, size=(len(s_right),2)).to(device)
        a_svgd_right, _, _ = ac.svgd_sampler(s_right, a_rand.detach()) 
        q_svgd_right = ac.q1(s_right,a_svgd_right).detach()
        q_svgd_right_var = torch.var(q_svgd_right)

        #if itr==2000:
        #    import pdb; pdb.set_trace()
        writer.add_scalars('init_state/q_var',{'q_up': q_svgd_up_var, 'q_down':q_svgd_down_var, 'q_left':q_svgd_left_var, 'q_right':q_svgd_right_var}, itr)
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
        
        print("saving figure..., epoch=",epoch)
        number_of_hits_mode = env.render_rollouts(paths,num_episodes, plot_svgd_steps, epoch=epoch, eps=eps, fout='loss_p_200k_elu_'+sac_version+"_alphaQ_"+str(alpha_q)+"_alphaP_"+str(alpha_p)+'_svgd_steps_'+str(num_svgd_steps)+'_svgd_particles_'+str(num_svgd_particles)+'_svgd_lr_'+str(svgd_lr)+"_epoch_"+str(epoch)+"_batch_size_"+str(batch_size)+'_lr_p_'+str(lr_p)+"_num_episodes_"+str(num_episodes)+".png" )
        total_number_of_hits_mode = number_of_hits_mode.sum()
        if total_number_of_hits_mode > 0.0:
            m0 = number_of_hits_mode[0]/total_number_of_hits_mode
            m1 = number_of_hits_mode[1]/total_number_of_hits_mode
            m2 = number_of_hits_mode[2]/total_number_of_hits_mode
            m3 = number_of_hits_mode[3]/total_number_of_hits_mode
        else:
            m0, m1, m2, m3 = 0, 0, 0, 0
        ac_hess_list = torch.stack(ac_hess_list)
        ac_score_func_list = torch.stack(ac_score_func_list)
        ac_hess_eig_max = torch.stack(ac_hess_eig_max)

        # 
        writer.add_scalar('smoothness/ac_score/mean', torch.abs(ac_score_func_list).mean() , epoch)
        writer.add_scalar('smoothness/ac_score/std', torch.abs(ac_score_func_list).std() , epoch)
        writer.add_scalar('smoothness/hess/mean', torch.abs(ac_hess_list).mean() , epoch)
        writer.add_scalar('smoothness/hess/std', torch.abs(ac_hess_list).std() , epoch)
        writer.add_scalar('smoothness/hess/max_eigen_val/mean', ac_hess_eig_max.mean() , epoch)
        writer.add_scalar('smoothness/hess/max_eigen_val/std', ac_hess_eig_max.std() , epoch)

        # 
        writer.add_scalar('modes/num_modes',(number_of_hits_mode>0).sum(), epoch)
        writer.add_scalar('modes/total_number_of_hits_mode',total_number_of_hits_mode, epoch)
        writer.add_scalar('modes/prob_mod_0',m0, epoch)
        writer.add_scalar('modes/prob_mod_1',m1, epoch)
        writer.add_scalar('modes/prob_mod_2',m2, epoch)
        writer.add_scalar('modes/prob_mod_3',m3, epoch)

        picks = 0

        #import pdb; pdb.set_trace() 
        if (epoch>200000):
            if ((number_of_hits_mode>0).sum()!=4):
                picks += 1
                stability = 1 - (picks/(epoch-200000))
                writer.add_scalar('modes/stability',stability, epoch)

        


