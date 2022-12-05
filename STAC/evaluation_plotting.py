import numpy as np
import torch 
import pandas as pd
import pickle
import matplotlib.pyplot as plt 
from utils import moving_average
import seaborn as sns

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
        try:
            project_name = evaluations['projects_names'][i]
        except:
            print('hi')
        title = evaluations['experiments'][i]

        evaluation_data_path = './Finished Experiments/evaluation/' + project_name + '/evaluation_data.pickle' 
        save_path = './STAC/mujoco_plots_/'
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
    'experiments': ['sac_halfcheetah', 'sac_hopper', 'sac_walker2d', 'sac_ant'],
    # 'experiments': ['sql_hopper', 'sql_walker2d', 'sql_halfcheetah', 'sql_ant','sql_humanoid'],
    'projects_names': [
        'Sep_24_2022_15_36_15_sac_HalfCheetah-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU',
        'Sep_24_2022_15_36_15_sac_Hopper-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU',
        'Sep_24_2022_15_36_15_sac_Walker2d-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU',
        'Sep_24_2022_15_36_16_sac_Ant-v2_alpha_0.2_batch_size_100_lr_critic_0.001_lr_actor_0.001_activation_ReLU',

        # 'Sep_25_2022_19_24_36_svgd_sql_Hopper-v2_alpha_1.0_batch_size_100_lr_critic_0.01_lr_actor_0.001_activation_ReLU',
        # 'Sep_25_2022_19_24_36_svgd_sql_Walker2d-v2_alpha_1.0_batch_size_100_lr_critic_0.01_lr_actor_0.001_activation_ReLU',
        # 'Sep_25_2022_19_24_36_svgd_sql_HalfCheetah-v2_alpha_1.0_batch_size_100_lr_critic_0.01_lr_actor_0.001_activation_ReLU',
        # 'Sep_25_2022_19_24_36_svgd_sql_Ant-v2_alpha_1.0_batch_size_100_lr_critic_0.01_lr_actor_0.001_activation_ReLU',
        # 'Sep_26_2022_15_33_05_svgd_sql_Humanoid-v2_alpha_1.0_batch_size_100_lr_critic_0.01_lr_actor_0.001_activation_ReLU'
    ]

}


def create_histogram():
    X = ['1', '2', '3']
    y = [47, 161, 132]
    # y = [100, 160, 140]

    X_df = []
    for i in range(len(X)):
        X_df += [[X[i], 1] for _ in range(y[i])]

    df = pd.DataFrame(X_df, columns=['category', 'count'])

    df.value_counts()

    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    ax = plt.subplot(111)
    ax.set_title('STAC')
    ax.set_xlabel('Goals')
    ax.set_ylabel('Count')
    sns.histplot(data=df, x='category', shrink=.7)


create_histogram()
# plot_results(evaluations, )
# plot_results(evaluations, 500)

