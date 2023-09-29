"""
Read all the experimental data and calculate performance for each noise distribution
@author: Lynn Schmittwilken, created Nov 2021
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nF = 0    # number of direction changes to ignore


#########################################
#               Functions               #
#########################################
def get_data(datadir, sess_type):
    all_data = []

    # Read in the experimental data from all the subfolders
    filenames = os.listdir(datadir)
    for filename in filenames:
        if filename.startswith(sess_type + "_"):
            data = pd.read_csv(os.path.join(datadir, filename), delimiter='\t')

            # Add the individual data to the full dataset
            all_data.append(data)

    # Concat dataframes of all subjects:
    all_data = pd.concat(all_data)

    # Let's reset the indices of the full dataframe, so it ranges from 0 to n
    all_data = all_data.reset_index(drop=True)
    return all_data


def get_all_data(folder_dir, subject_dirs, sess_type):
    n_sub = len(subject_dirs)
    data_all = []

    for i in range(n_sub):
        datadir = folder_dir + subject_dirs[i] + '/' + sess_type
        data_sub = get_data(datadir, sess_type)

        # Add a column for the subject ID (here: ranging from 0 to 10)
        vp_id = np.repeat(i, data_sub.shape[0])
        data_sub['vp'] = vp_id

        # Add data of individuals to the full dataset
        data_all.append(data_sub)

    # Concat dataframes of all subjects:
    data_all = pd.concat(data_all)
    return data_all


def plot_stairdata(data):
    noise_contrast = np.unique(data['noise_contrast'])
    edgewidths = np.unique(data['edge_width'])
    noisetypes = np.unique(data['noise'])

    # Get data for the different noise distributions:
    for w in edgewidths:
        data_list = []
        for n in noisetypes:
            data_list.append(data[(data['noise'] == n) & (data['edge_width'] == w)]['edge_contrast'])
        plot_data(data_list, noisetypes)
        plt.title(str(w))
    plt.show()


def plot_data(data_list, label_list):
    # Plot results from each trial:
    plt.figure(figsize=(16, 6))
    for i in range(len(data_list)):
        plt.plot(data_list[i], '.-', label=label_list[i])
        plt.legend()
        plt.title('Consecutive trials')
        plt.xlabel('Trials')
        plt.ylabel('Edge contrasts')


def count_flips(data, nback):
    responses = np.empty(0)
    flip_count = 0
    flip_direction = 1
    for trl in range(len(data)):
        responses = np.append(responses, data['correct'].iloc[trl])
        # Calculate number of flips per noise type
        if (trl > nback) & (responses[trl-nback:trl].sum() == nback) & (responses[trl] == 0.) & (flip_direction==1):
            flip_count += 1
            flip_direction = -1
        elif (trl > nback) & (responses[trl-1] == 0.) & (responses[trl] == 1.) & (flip_direction==-1):
            flip_count += 1
            flip_direction = 1
    return flip_count


def get_flips(data, nback):
    # We need to reset the index to pick contrast values in specific trials:
    data = data.reset_index(drop=True)
    responses = np.empty(0)
    contrast = []
    flip_direction = 1

    # Code to identify trials with flips / direction changes:
    for trl in range(len(data)):
        responses = np.append(responses, data['correct'].iloc[trl])
        # Calculate number of flips per noise type
        if (trl > nback) & (responses[trl-nback:trl].sum() == nback) & (responses[trl] == 0.) & (flip_direction==1):
            flip_direction = -1
            contrast.append(data['edge_contrast'][trl])
        elif (trl > nback) & (responses[trl-1] == 0.) & (responses[trl] == 1.) & (flip_direction==-1):
            flip_direction = 1
            contrast.append(data['edge_contrast'][trl])
    return contrast


def calculate_thresholds(data):
    t_none = get_flips(data[data['noise'] == 'none'])
    t_white = get_flips(data[data['noise'] == 'white'])
    t_pink1 = get_flips(data[data['noise'] == 'pink1'])
    t_pink2 = get_flips(data[data['noise'] == 'pink2'])
    t_lnarrow = get_flips(data[data['noise'] == 'narrow0.5'])
    t_mnarrow = get_flips(data[data['noise'] == 'narrow3'])
    t_hnarrow = get_flips(data[data['noise'] == 'narrow9'])

    # Reduce effect of lapses by removing first nF flips, and calculate average
    m_none = np.mean(t_none[nF::])
    m_white = np.mean(t_white[nF::])
    m_pink1 = np.mean(t_pink1[nF::])
    m_pink2 = np.mean(t_pink2[nF::])
    m_lnarrow = np.mean(t_lnarrow[nF::])
    m_mnarrow = np.mean(t_mnarrow[nF::])
    m_hnarrow = np.mean(t_hnarrow[nF::])

    means = [m_none, m_white, m_pink1, m_pink2, m_lnarrow, m_mnarrow, m_hnarrow]
    titles = ['none', 'white', 'pink', 'brown', 'narrow0.58', 'narrow3', 'narrow9']
    return means, titles


def calculate_thresholds_all(folder_dir, subjects, this_width, sess_type='staircase'):
    n_subjects = len(subjects)
    threshs = []
    for i in range(n_subjects):
        # Load data
        data = get_data(folder_dir + subjects[i] + '/' + sess_type, sess_type)
        edge_widths = np.unique(data['edge_width'])
        data = data[data['edge_width'] == edge_widths[this_width]]

        # Calculate thresholds:
        threshs_ind, titles = calculate_thresholds(data)
        threshs.append(np.array(threshs_ind))
    threshs = np.array(threshs)
    threshs_mean = threshs.mean(0)
    threshs_se = threshs.std(0) / np.sqrt(n_subjects)
    return threshs_mean, threshs_se, titles


def calculate_sensitivity_all(folder_dir, subjects, this_width, sess_type='staircase'):
    n_subjects = len(subjects)
    sens = []
    for i in range(n_subjects):
        # Load data
        data = get_data(folder_dir + subjects[i] + '/' + sess_type, sess_type)
        edge_widths = np.unique(data['edge_width'])
        data = data[data['edge_width'] == edge_widths[this_width]]

        # Calculate sensitivity:
        threshs_ind, titles = calculate_thresholds(data)
        sens.append(1. / np.array(threshs_ind))
    sens = np.array(sens)
    sens_mean = sens.mean(0)
    sens_se = sens.std(0) / np.sqrt(n_subjects)
    return sens_mean, sens_se, titles


def plot_sensitivity(data):
    edge_widths = np.unique(data['edge_width'])
    edge0_means, titles = calculate_thresholds(data[data['edge_width'] == edge_widths[0]])
    edge1_means, _ = calculate_thresholds(data[data['edge_width'] == edge_widths[1]])
    edge2_means, _ = calculate_thresholds(data[data['edge_width'] == edge_widths[2]])

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.plot(np.array(edge0_means), '-o', label=str(edge_widths[0]))
    plt.plot(np.array(edge1_means), '-o', label=str(edge_widths[1]))
    plt.plot(np.array(edge2_means), '-o', label=str(edge_widths[2]))
    plt.ylim(0., 0.03)
    plt.xticks(np.arange(len(titles)), titles)
    plt.legend()
    plt.title('Thresholds')

    plt.subplot(122)
    plt.plot(1. / np.array(edge0_means), '-o', label=str(edge_widths[0]))
    plt.plot(1. / np.array(edge1_means), '-o', label=str(edge_widths[1]))
    plt.plot(1. / np.array(edge2_means), '-o', label=str(edge_widths[2]))
    plt.yscale('log')
    plt.ylim(10., 6000.)
    plt.xticks(np.arange(len(titles)), titles)
    plt.legend()
    plt.title('Sensitivity')



def plot_performance(data, title):
    edge_width = np.unique(data['edge_width'])
    results = np.zeros([len(edge_width), 5, 20])

    # Get data for the different noise distributions:
    plt.figure()
    for i, w in enumerate(edge_width):
        contrasts = np.unique(data[(data['edge_width'] == w)]['edge_contrast'])
        for j, c in enumerate(contrasts):
            results[i, j, :] = data[(data['edge_width'] == w) & (data['edge_contrast'] == c)]['correct']
        means = results[i, :, :].mean(1)
        plt.plot(contrasts, means, label=str(w))
        plt.title(title)
    return


def get_curve(data):
    contrasts = np.unique(data["edge_contrast"])
    n_correct = np.zeros(len(contrasts))
    n_trials = np.zeros(len(contrasts))
    for i in range(len(contrasts)):
        data_correct = data[data["edge_contrast"] == contrasts[i]]["correct"]
        n_correct[i] = np.sum(data_correct)
        n_trials[i] = len(data_correct)
    p_correct = n_correct / n_trials
    return contrasts, p_correct, n_trials


def plot_psychometric_curve(data, f=2):
    edge_widths = np.unique(data['edge_width'])
    fig, axes = plt.subplots(3, 7, sharex=True, sharey=True, figsize=(16, 6))
    for i, w in enumerate(edge_widths):
        data_w = data[data['edge_width'] == w]
        c1, p1, t1 = get_curve(data_w[data_w['noise'] == 'none'])
        c2, p2, t2 = get_curve(data_w[data_w['noise'] == 'white'])
        c3, p3, t3 = get_curve(data_w[data_w['noise'] == 'pink1'])
        c4, p4, t4 = get_curve(data_w[data_w['noise'] == 'pink2'])
        c5, p5, t5 = get_curve(data_w[(data_w['noise'] == 'narrow') & (data_w['sf'] == 0.58)])
        c6, p6, t6 = get_curve(data_w[(data_w['noise'] == 'narrow') & (data_w['sf'] == 3.)])
        c7, p7, t7 = get_curve(data_w[(data_w['noise'] == 'narrow') & (data_w['sf'] == 9.)])

        axes[i, 0].plot(c1, p1)
        axes[i, 1].plot(c2, p2)
        axes[i, 2].plot(c3, p3)
        axes[i, 3].plot(c4, p4)
        axes[i, 4].plot(c5, p5)
        axes[i, 5].plot(c6, p6)
        axes[i, 6].plot(c7, p7)

        axes[i, 0].scatter(c1, p1, t1**f)
        axes[i, 1].scatter(c2, p2, t2**f)
        axes[i, 2].scatter(c3, p3, t3**f)
        axes[i, 3].scatter(c4, p4, t4**f)
        axes[i, 4].scatter(c5, p5, t5**f)
        axes[i, 5].scatter(c6, p6, t6**f)
        axes[i, 6].scatter(c7, p7, t7**f)
    return

#########################################
#                 Main                  #
#########################################
vp_id = 'ls'
datadir = './results/%s/staircase/' % vp_id

# Load all data
data = get_data(datadir, 'staircase')
noisetypes = np.unique(data["noise"])
edgewidths = np.unique(data["edge_width"])
nbacks = np.unique(data["nback"])

# Plot staircase data
plot_stairdata(data[data["nback"] == nbacks[0]])
plot_stairdata(data[data["nback"] == nbacks[1]])


# plot_sensitivity(data)
# plot_psychometric_curve(data)
#plt.savefig(vp_id + '.png', dpi=300)

# Count number of flips:
print_flips = True
if print_flips:
    for w in edgewidths:
        for n in noisetypes:
            for nb in nbacks:
                print(str(w) + n + str(nb))
                print(count_flips(data[(data['noise'] == n) & (data['edge_width'] == w) & (data['nback'] == nb)], nb))


print_combined = False
plot_threholds = False
subs = ['ls', 'mm', 'ga', 'jv', 'sg']
if print_combined:
    for i in range(3):
        plt.figure(1, figsize=(6, 6))
        if plot_threholds:
            t_mean, t_se, titles = calculate_thresholds_all('./results/', subs, i)
            plt.errorbar(titles, t_mean, yerr=t_se, capsize=3, marker='o', linestyle='-')

        else:
            s_mean, s_se, titles = calculate_sensitivity_all('./results/', subs, i)
            plt.errorbar(titles, s_mean, yerr=s_se, capsize=3, marker='o', linestyle='-')
            plt.yscale('log')
    # plt.savefig('results/exp_sensitivity_int2.png', dpi=300)
    # plt.close()


# For non-staircase experiment:
#this_data = data[(data['noise'] == 'pink2')]
#plot_performance(this_data, 'none')


# Calculate bias in %correct between intervall 1 and 2
#data = get_all_data('./results/', subs, 'staircase')
#data = data[(data['noise'] == 'narrow') & (data['sf'] == 9)]
#print(np.mean(data[data['intID'] == 0]['correct']))
#print(np.mean(data[data['intID'] == 1]['correct']))



# Get psychometric curve over all participants
#data = get_all_data('./results/', subs, 'staircase')
#plot_psychometric_curve(data, 1.5)
