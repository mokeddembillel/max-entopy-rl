import numpy as np
import torch 
import pandas as pd
import pickle
import matplotlib.pyplot as plt 
from utils import moving_average

def line_plot(data, xlabel, ylabel, title, save_file):
    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    ax = plt.subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(np.arange(len(data)), data)
    plt.savefig(save_file + '.png', dpi=300)
    plt.show()




def plot_results(evaluations, moving_average_window=None):
    for i in range(len(evaluations['experiments'])):
        project_name = evaluations['projects_names'][i]
        title = evaluations['experiments'][i]

        evaluation_data_path = './evaluation_data/' + project_name + '/evaluation_data.pickle' 
        save_path = './STAC/mujoco_plots_/'+ project_name + '/'
        evaluation_data = pickle.load(open(evaluation_data_path, 'rb'))
        # print(evaluation_data.keys())
        train_episodes_return = evaluation_data['train_episodes_return']
        train_episodes_length = evaluation_data['train_episodes_length']
        if moving_average_window:
            line_plot(moving_average(train_episodes_return, moving_average_window), 'episodes', 'Average return', title, save_path + 'moving_average_return_' + project_name + '_plot')
            line_plot(moving_average(train_episodes_length, moving_average_window), 'episodes', 'Episode length', title, save_path + 'moving_episode_length_' + project_name + '_plot')
        else:
            line_plot(train_episodes_return, 'episodes', 'Average return', title, save_path + 'average_return_' + project_name + '_plot')
            line_plot(train_episodes_length, 'episodes', 'Episode length', title, save_path + 'epsidoe_length_' + project_name + '_plot')
        print('Results Saved!')



project_name = 'Sep_24_2022_15_36_16_sac_Ant-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU'
evaluations = {
    'experiments': ['sac_halfcheetah', 'sac_hopper', 'sac_walker2d', 'sac_ant', 'sql_hopper', 'sql_walker2d'],
    'projects_names': [
        'Sep_24_2022_15_36_15_sac_HalfCheetah-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU',
        'Sep_24_2022_15_36_15_sac_Hopper-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU',
        'Sep_24_2022_15_36_15_sac_Walker2d-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU',
        'Sep_24_2022_15_36_16_sac_Ant-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU',
        'Sep_25_2022_19_24_36_svgd_sql_Hopper-v2_alpha_1.0_batch_size_100_lr_critic_0.01_lr_actor_0.001_activation_ReLU',
        'Sep_25_2022_19_24_36_svgd_sql_Walker2d-v2_alpha_1.0_batch_size_100_lr_critic_0.01_lr_actor_0.001_activation_ReLU'
    ]

}


plot_results(evaluations, )
plot_results(evaluations, 500)

