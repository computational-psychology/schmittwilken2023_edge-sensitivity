# -*- coding: utf-8 -*-
"""
Edge detection experiment for Cornsweet edge in different noise distributions
Uses HRL on python 2
@author: Lynn Schmittwilken Nov 2021, adapted from template experiment MLCM
"""

from socket import gethostname
import os
import numpy as np
from hrl import HRL
import pygame

from experiment_functions import show_continue, present_trial, show_instruction, \
    get_block_info, measure_refresh_rate, staircase
from helper_functions import make_sound
from stimulus_functions import get_stim_params, prepare_stim
from params import bg, lut, stim_time, n_reversals, n_adapt, n_masks


# Log file name and location, decides which design matrix is loaded
vp_id = input('Please input the observer name (e.g. demo): ')

# Flag to switch between running locally or running in the lab (inlab = True)
inlab = True if 'vlab' in gethostname() else False

# Size of Siemens monitor / Center of screen
WIDTH, HEIGHT = 1024, 768
wh, hh = WIDTH / 2.0, HEIGHT / 2.0

# Sound to be played at each interval
f1, f2, f3, f4 = 500, 600, 800, 300
pygame.mixer.pre_init(44100, -16, 1)

# Create HRL object
if inlab:
    hrl = HRL(graphics='datapixx', inputs='responsepixx', photometer=None, wdth=WIDTH,
              hght=HEIGHT, bg=bg, scrn=1, lut=lut, db=True, fs=True)
    monitor_rate = 130

else:
    pygame.init()
    hrl = HRL(graphics='gpu', inputs='keyboard', photometer=None, wdth=WIDTH,
              hght=HEIGHT, bg=bg, scrn=1, lut=lut, db=True, fs=False)
    monitor_rate = 60

# Tolerance rate before throwing an error:
tol_rate = monitor_rate*0.01

# initializing sounds
sound1 = pygame.sndarray.make_sound(make_sound(f1))
sound2 = pygame.sndarray.make_sound(make_sound(f2))
sound3 = pygame.sndarray.make_sound(make_sound(f3))
sound4 = pygame.sndarray.make_sound(make_sound(f4))
sounds = [sound1, sound2, sound3, sound4]


# Function that defines how to run a trial:
def run_trial(hrl, edge_contrast, params_static, params_dynamic):
    stim_time = params_static["stim_time"]
    edge_width = params_static["edge_width"]
    noise_type = params_static["noise_type"]
    maskIDs = params_dynamic["maskIDs"]
    stairID = params_dynamic["stairID"]
    nback = params_dynamic["nback"]
    trl = params_dynamic["trl"]

    # Functions cannot handle zero-contrast. Use a very small contrast instead.
    eps = 0.00001
    if edge_contrast < eps:
        edge_contrast = eps

    # Randomize polarity and translation of edge stimulus
    rotID = np.random.randint(0, 2)
    transID = np.random.randint(0, 2)

    # Randomly pull a noise mask. Make sure to not pull the same mask within 5 trials.
    maskID = np.random.randint(0, n_masks)

    while ((maskIDs[-1] == maskID) or
           (maskIDs[-2] == maskID) or
           (maskIDs[-3] == maskID) or
           (maskIDs[-4] == maskID) or
           (maskIDs[-5] == maskID)):
        maskID = np.random.randint(0, n_masks)
    maskIDs.append(maskID)

    # Create stim params object (sp) and create stimulus
    sp = get_stim_params(edge_width, edge_contrast, noise_type, stim_time, rotID, transID,
                         maskID, monitor_rate, refreshThreshold)
    stim = prepare_stim(sp)

    print('--------------------------------------')
    print("TRIAL =", trl)
    print('Noise type: %s, contrast: %f' % (noise_type, sp.noise_contrast))
    print('Edge contrast: %f, width: %f' % (edge_contrast, sp.edge_width))
    print('Mean lum: %f, stim duration: %f' % (sp.mean_lum, stim_time))
    print('Translation: %d, rotation: %d' % (transID, rotID))

    # Present trial
    resp, rtime, dropframes = present_trial(hrl, stim, sp, [wh, hh], sounds, inlab)

    # Calculate response so that 0:incorrect and 1:correct
    correct = np.abs(resp - transID)
    correct = np.abs(correct - 1)
    print(['Incorrect!', 'Correct!'][correct])

    # Save trial data
    data_row = (trl, noise_type, sp.noise_contrast, edge_contrast, sp.edge_width, sp.mean_lum,
                stim_time, rotID, transID, maskID, stairID, nback, resp, correct, rtime, dropframes)
    rfl.write('%d\t%s\t%f\t%f\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%f\t%d\n' % data_row)
    rfl.flush()
    return correct, {"maskIDs": maskIDs}


############################################
#                Run main                  #
############################################
if __name__ == '__main__':
    result_headers = ['trial', 'noise', 'noise_contrast', 'edge_contrast', 'edge_width', 'mean_lum',
                      'stim_time', 'rotID', 'transID', 'maskID', 'stairID', 'nback','resp', 'correct',
                      'resptime', 'dropframes']

    # Create results folder for participant
    results_dir = './results/%s/staircase/' % vp_id
    if os.path.exists(results_dir) == 0:
        os.makedirs(results_dir)

    # Get information about which blocks need to be run
    order_path = './design/%s/staircases_order.csv' % vp_id
    done_path = results_dir + '/staircases_done.txt'
    blockstorun, blocksdone = get_block_info(order_path, done_path, 'staircase', sep=',')

    # Opens block file to write
    bfl = open(done_path, 'a')
    if blocksdone is None:
        bfl.write('\t'.join(['number', 'noisetype', 'edge_width']) + '\n')
        bfl.flush()

    # Check refresh rate of monitor
    rate, refreshThreshold = measure_refresh_rate(hrl, monitor_rate)
    if inlab:
        # if refresh rate is out of range, through error and exits
        if rate < (monitor_rate-tol_rate) or rate > (monitor_rate+tol_rate):
            raise(RuntimeError, 'Monitor refresh rate is out of tolerated range')

    # Display instructions
    show_instruction(hrl, 'staircase')

    # Run staircase for all staircase blocks
    for i in range(len(blockstorun['number'])):
        # Check refresh rate of monitor
        rate, refreshThreshold = measure_refresh_rate(hrl, monitor_rate)
        if inlab:
            # if refresh rate is out of range, through error and exits
            if rate < (monitor_rate-tol_rate) or rate > (monitor_rate+tol_rate):
                raise(RuntimeError, 'Monitor refresh rate is out of tolerated range')

        sess = int(blockstorun['number'][i])                  # Get block number
        n = str(blockstorun['noisetype'][i])                  # Get noisetype
        edge_width = float(blockstorun['edge_width'][i])      # Get edge width
        x_init = float(blockstorun['init_edge_contrast'][i])  # Get initial contrast
        rfl = open(results_dir + 'staircase_%s_%f.txt' % (n, edge_width), 'w')
        rfl.write('\t'.join(result_headers)+'\n')
        print("------------------------------------------------")
        print("Running block", sess)
        print("------------------------------------------------")

        # Params that dont change during the staircase
        params_dict = {"stim_time": stim_time,
                       "edge_width": edge_width,
                       "noise_type": n,
                       }

        # Initiate two staircases
        nback1 = 3
        stair_3down = staircase(hrl=hrl,
                                n_back=nback1,
                                n_reversals=n_reversals,
                                x_init=x_init,
                                run_trial=run_trial,
                                xmin=0,
                                n_adapt=n_adapt,
                                params_dict=params_dict
                                )

        nback2 = 6
        stair_6down = staircase(hrl=hrl,
                                n_back=nback2,
                                n_reversals=n_reversals,
                                x_init=x_init,
                                run_trial=run_trial,
                                xmin=0,
                                n_adapt=n_adapt,
                                params_dict=params_dict
                                )

        # Initiate relevant variables
        trl1, trl2 = 0, 0
        responses1, responses2 = np.empty(0), np.empty(0)
        r_count1, r_count2 = 0, 0
        r_direction1, r_direction2 = 1, 1
        r_vals1, r_vals2 = np.empty(0), np.empty(0)
        r_left1, r_left2 = 1, 1
        x1, x2 = x_init, x_init
        params_dynamic = {"maskIDs": [-1,]*5}

        # Run staircases, until both are finished:
        while (r_left1 > 0) or (r_left2 > 0):
            # Randomly choose one of the two staircases
            rnd = np.random.randint(0, 2)
            params_dynamic["stairID"] = rnd
            params_dynamic["trl"] = trl1 + trl2

            if rnd == 0:
                params_dynamic["nback"] = nback1
                stair_out1 = stair_3down.run(trl1,
                                             x1,
                                             responses1,
                                             r_count1,
                                             r_direction1,
                                             r_vals1,
                                             params_dynamic,
                                             )
                trl1 = stair_out1[0]
                x1 = stair_out1[1]
                responses1 = stair_out1[2]
                r_count1 = stair_out1[3]
                r_direction1 = stair_out1[4]
                r_vals1 = stair_out1[5]
                params_dynamic = stair_out1[6]
                r_left1 = stair_out1[7]

            elif rnd == 1:
                params_dynamic["nback"] = nback2
                stair_out2 = stair_6down.run(trl2,
                                             x2,
                                             responses2,
                                             r_count2,
                                             r_direction2,
                                             r_vals2,
                                             params_dynamic,
                                             )
                trl2 = stair_out2[0]
                x2 = stair_out2[1]
                responses2 = stair_out2[2]
                r_count2 = stair_out2[3]
                r_direction2 = stair_out2[4]
                r_vals2 = stair_out2[5]
                params_dynamic = stair_out2[6]
                r_left2 = stair_out2[7]

            else:
                raise ValueError("Only have two staircases")

        # Close result file for this block
        rfl.close()

        # Save the block just done
        bfl.write('%d\t%s\t%f\n' % (sess, n, edge_width))
        bfl.flush()

        # Continue?
        btn = show_continue(hrl, i + 1, len(blockstorun['number']))
        print("continue screen, pressed ", btn)
        if btn == 'Left':
            break

    # Close remaining files and finish everything
    bfl.close()
    hrl.close()
    print("Session finished")
