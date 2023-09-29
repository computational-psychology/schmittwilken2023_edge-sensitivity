"""
Generates design file for edge detection experiment in different noise distributions
@author: GA July 2021, JV September 2021, LS November 2021
"""

import numpy as np
import pandas as pd
import os
from params import stim_params, exp_params

noise_types = stim_params["noise_types"]
edge_widths = stim_params["edge_widths"]
nmasks = stim_params["n_masks"]
stim_time = exp_params["stim_time"]
nreps = exp_params["n_reps"]
nsteps = exp_params["n_steps"]

# Since the different noise types vary strongly in how effective they are,
# we use different initial edge contrasts for each noise type in the
# staircase procedure and in the warmup (values from pilotting)
xedge1 = stim_params["xedge048_init"]
xedge2 = stim_params["xedge15_init"]
xedge3 = stim_params["xedge95_init"]
data = {
	"noise_type": noise_types * len(edge_widths),
	"edge_width": np.sort(edge_widths * len(noise_types)),
	"init_x": xedge1 + xedge2 + xedge3,
	}
df = pd.DataFrame(data)

xedge1 = []
xedge2 = []
xedge3 = []
noiselist = []
for i in range(len(noise_types)):
    xedge1 += list(np.linspace(stim_params["xedge048_min"][i],
                               stim_params["xedge048_max"][i],
                               nsteps))
    xedge2 += list(np.linspace(stim_params["xedge15_min"][i],
                               stim_params["xedge15_max"][i],
                               nsteps))
    xedge3 += list(np.linspace(stim_params["xedge95_min"][i],
                               stim_params["xedge95_max"][i],
                               nsteps))
    noiselist += [noise_types[i],] * nsteps

data = {
	"noise_type": noiselist * len(edge_widths),
	"edge_width": np.sort(edge_widths * len(noise_types) * nsteps),
	"contrasts": xedge1 + xedge2 + xedge3,
	}

df_exp = pd.DataFrame(data)

# Path to design csv:
designs_dir = os.path.join(".", "design")


def warmup_design_matrix(noise_types=noise_types, nrep=5):
    stim_duration = [0.5] + [stim_time,]*3

    # Create design
    design = [(w, t) for t in stim_duration for w in edge_widths]
    design = np.repeat(design, nrep, axis=0)

    # Shuffle
    for t in stim_duration:
        design[design[:, 1] == t] = np.random.permutation(design[design[:, 1] == t])

    # Create random noiseID vector
    rd_noiseIDs = np.random.randint(0, len(noise_types), len(design))

    full_design = []
    for i in range(len(design)):
        this_id = noise_types[rd_noiseIDs[i]]
        this_w = design[i][0]
        this_d = design[i][1]
        this_c = float(df[(df["noise_type"]==this_id) &  (df["edge_width"]==this_w)]["init_x"])

        full_design.append([this_id, this_w, this_c, this_d])
    return full_design


def staircase_design_matrix(noise_types=noise_types, edge_widths=edge_widths, df=df):
    # Create design
    design = [(i, w) for i in noise_types for w in edge_widths]

    # Add initial edge contrasts depending on noiseID
    full_design = []
    for i in range(len(design)):
        this_id, this_w = design[i][0], design[i][1]

        for j in range(len(noise_types)):
            if this_id == noise_types[j]:
                this_c = float(df[(df["noise_type"]==this_id) &  (df["edge_width"]==this_w)]["init_x"])

        full_design.append([this_id, this_w, this_c])
    return full_design


def experiment_order_matrix(noise_types=noise_types, edge_widths=edge_widths):
    # Create ordeer
    design = [(i, w) for i in noise_types for w in edge_widths]
    return design


def experiment_design_matrix(noise_type, edge_width, df, nreps, nsteps):
    cons = np.array(df[(df["noise_type"]==noise_type) &
                       (df["edge_width"]==edge_width)]["contrasts"])

    # Create design
    cons = np.repeat(cons, nreps, axis=0)
    cons = np.random.permutation(cons)
    maskIDs = [-1,] * 5
    while (len(maskIDs)-5) < len(cons):
        maskID = np.random.randint(0, nmasks)
        # Make sure that there are no repetitions within 5 trials
        while ((maskID == maskIDs[-1]) or
               (maskID == maskIDs[-2]) or
               (maskID == maskIDs[-3]) or
               (maskID == maskIDs[-4]) or
               (maskID == maskIDs[-5])):
            maskID = np.random.randint(0, nmasks)
        maskIDs.append(maskID)
    maskIDs = np.random.permutation(maskIDs[5::])
    
    design = np.array([cons, maskIDs]).transpose()
    return design


def generate_block(design_matrix, headers, shuffle=True):
    # Creates dataframe with all trials
    block = pd.DataFrame(design_matrix, columns=headers[1::])
    block = block[headers[1::]]

    # Shuffle trial order
    if shuffle:
        block = block.reindex(np.random.permutation(block.index))
        block.reset_index(drop=True, inplace=True)
    block.index.name = headers[0]
    return block


def generate_warmup_csv(design_dir):
    filename = 'warmup.csv'
    headers = ['trial', 'noisetype', 'edge_width', 'edge_contrast', 'stim_time']
    design_matrix = warmup_design_matrix()
    block = generate_block(design_matrix, headers, shuffle=False)

    # save in design folder, under block number
    filepath = os.path.join(design_dir, filename)
    block.to_csv(filepath)


def generate_staircase_csv(design_dir):
    filename = 'staircases_order.csv'
    headers = ['number', 'noisetype', 'edge_width', 'init_edge_contrast']
    design_matrix = staircase_design_matrix()
    block = generate_block(design_matrix, headers)

    # save in design folder, under block number
    filepath = os.path.join(design_dir, filename)
    block.to_csv(filepath)


def generate_experiment_order(design_dir):
    filename = 'experiment_order.csv'
    headers = ['number', 'noisetype', 'edge_width']
    design_matrix = experiment_order_matrix()
    block = generate_block(design_matrix, headers)

    # save in design folder, under block number
    filepath = os.path.join(design_dir, filename)
    block.to_csv(filepath)


def generate_experiment_designs(noise_types, edge_widths, df, nreps, nsteps):
    headers = ['trial', 'edge_contrast', 'maskID']
    for n in noise_types:
        for e in edge_widths:
            filename = "experiment_%s_%f.csv" % (n, e)
            design_matrix = experiment_design_matrix(n, e, df, nreps, nsteps)
            block = generate_block(design_matrix, headers, shuffle=False)

            # save in design folder
            filepath = os.path.join(design_dir, filename)
            block.to_csv(filepath)


if __name__ == "__main__":
    obsname = input("Please enter participant initials: ")  # also folder name where to save

    # does the design folder already exist?
    design_dir = os.path.join(designs_dir, obsname)
    if os.path.exists(design_dir) == 1:
        raise NameError('Design folder exists. Use different initials or delete existing folder.')
    else:
        os.makedirs(design_dir)

    print("Generating warm-up block")
    generate_warmup_csv(design_dir)

    # Create file with block order for staircase
    print("Generating staircase design")
    generate_staircase_csv(design_dir)
    
    # Create file with block order for experiment
    print("Generating experiment order")
    generate_experiment_order(design_dir)
    generate_experiment_designs(noise_types, edge_widths, df_exp, nreps, nsteps)
