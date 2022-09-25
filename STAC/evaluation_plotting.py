import numpy as np
import torch 
import pandas as pd
import pickle
import matplotlib.pyplot as plt 
from utils import moving_average

def line_plot(data, xlabel, ylabel, title, save_file):
    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    ax = plt.subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(np.arange(len(data)), data)
    plt.savefig(save_file + '.png', dpi=300)
    plt.show()

# project_name = 'Sep_23_2022_20_58_18_sac_Hopper-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU'
# project_name = 'Sep_23_2022_20_59_31_sac_HalfCheetah-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU'
# project_name = 'Sep_23_2022_21_01_11_sac_Ant-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU'
project_name = 'Sep_23_2022_21_02_31_sac_Walker2d-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU'

def plot_results(project_name, moving_average_window=None):
    evaluation_data_path = './evaluation_data/' + project_name + '/evaluation_data.pickle' 
    save_path = './STAC/mujoco_plots/'
    evaluation_data = pickle.load(open(evaluation_data_path, 'rb'))
    # print(evaluation_data.keys())
    train_episodes_return = evaluation_data['train_episodes_return']
    train_episodes_length = evaluation_data['train_episodes_length']
    if moving_average_window:
        line_plot(moving_average(train_episodes_return, moving_average_window), 'episodes', 'Average return', 'Average return over episodes', save_path + 'moving_average_return_' + project_name + '_plot')
        line_plot(moving_average(train_episodes_length, moving_average_window), 'episodes', 'Episode length', 'Episode_length of each episode', save_path + 'moving_episode_length_' + project_name + '_plot')
    else:
        line_plot(train_episodes_return, 'episodes', 'Average return', 'Average return over episodes', save_path + 'average_return_' + project_name + '_plot')
        line_plot(train_episodes_length, 'episodes', 'Episode length', 'Episode_length of each episode', save_path + 'epsidoe_length_' + project_name + '_plot')
    print('Results Saved!')
plot_results(project_name)
plot_results(project_name, 500)

