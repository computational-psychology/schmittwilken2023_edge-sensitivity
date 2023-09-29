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

from experiment_functions import read_design, present_trial, show_instruction, \
    measure_refresh_rate, get_block_info, show_continue
from helper_functions import make_sound
from stimulus_functions import get_stim_params, prepare_stim
from params import lut, bg, stim_time, start_easy


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
              hght=HEIGHT, bg=bg, scrn=0, lut=lut, db=True, fs=False)
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
def run_trial(hrl, trl):
    # Trial design variables
    edge_contrast = float(design['edge_contrast'][trl])
    maskID = int(float(design['maskID'][trl]))

    # Randomize polarity and translation of edge stimulus
    rotID = np.random.randint(0, 2)
    transID = np.random.randint(0, 2)

    # Create stim params object (sp) and create stimulus
    sp = get_stim_params(edge_width, edge_contrast, n, stim_time,
                         rotID, transID, maskID, monitor_rate, refreshThreshold)
    stim = prepare_stim(sp)

    print('--------------------------------------')
    print("TRIAL =", trl)
    print('Noise type: %s, contrast: %f' % (n, sp.noise_contrast))
    print('Edge contrast: %f, width: %f' % (edge_contrast, sp.edge_width))
    print('Mean lum: %f, stim duration: %f' % (sp.mean_lum, stim_time))
    print('Translation: %d, rotation: %d' % (transID, rotID))

    # Present trial
    resp, rtime, dropframes = present_trial(hrl, stim, sp, [wh, hh], sounds, inlab, feedback=False)

    # Calculate response so that 0:incorrect and 1:correct
    correct = np.abs(resp - transID)
    correct = np.abs(correct - 1)
    print(['Incorrect!', 'Correct!'][correct])

    # Save trial data
    data_row = (trl, n, sp.noise_contrast, edge_contrast, sp.edge_width, sp.mean_lum,
                stim_time, rotID, transID, maskID, resp, correct, rtime, dropframes)
    rfl.write('%d\t%s\t%f\t%f\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%f\t%d\n' % data_row)
    rfl.flush()
    return


def run_easy_trial(hrl, trl):
    # Trial design variables
    max_contrast = max(np.array(design['edge_contrast']).astype(float))
    maskID = int(float(design['maskID'][-trl-1]))

    # Randomize polarity and translation of edge stimulus
    rotID = np.random.randint(0, 2)
    transID = np.random.randint(0, 2)

    # Create stim params object (sp) and create stimulus
    sp = get_stim_params(edge_width, max_contrast, n, stim_time,
                         rotID, transID, maskID, monitor_rate, refreshThreshold)
    stim = prepare_stim(sp)

    print('--------------------------------------')
    print("TRIAL = easy start " + str(trl+1))
    print('Noise type: %s, contrast: %f' % (n, sp.noise_contrast))
    print('Edge contrast: %f, width: %f' % (max_contrast, sp.edge_width))
    print('Mean lum: %f, stim duration: %f' % (sp.mean_lum, stim_time))
    print('Translation: %d, rotation: %d' % (transID, rotID))

    # Present trial
    resp, rtime, dropframes = present_trial(hrl, stim, sp, [wh, hh], sounds, inlab, feedback=False)

    # Calculate response so that 0:incorrect and 1:correct
    correct = np.abs(resp - transID)
    correct = np.abs(correct - 1)
    print(['Incorrect!', 'Correct!'][correct])
    return


############################################
#                Run main                  #
############################################
if __name__ == '__main__':
    result_headers = ['trial', 'noise', 'noise_contrast', 'edge_contrast', 'edge_width', 'mean_lum',
                      'stim_time', 'rotID', 'transID', 'maskID', 'resp', 'correct', 'resptime',
                      'dropframes']

    # Create results folder for participant
    results_dir = './results/%s/experiment/' % vp_id
    if os.path.exists(results_dir) == 0:
        os.makedirs(results_dir)
    
    # Get information about which blocks need to be run
    order_path = './design/%s/experiment_order.csv' % vp_id
    done_path = results_dir + '/experiments_done.txt'
    blockstorun, blocksdone = get_block_info(order_path, done_path, 'experiment', sep=',')

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
        
        design = read_design('./design/%s/experiment_%s_%f.csv' % (vp_id, n, edge_width), ',')
        end_trl = len(design['trial'])  # Get end trial

        rfl = open(results_dir + 'experiment_%s_%f.txt' % (n, edge_width), 'w')
        rfl.write('\t'.join(result_headers)+'\n')
        print("------------------------------------------------")
        print("Running block", sess)
        print("------------------------------------------------")
        
        # Start with three easy trials to make sure that participants know
        # what they are looking for in this block
        if start_easy:
            for easy in range(3):
                run_easy_trial(hrl, easy)

        # Loop over one all trials:
        for trl in np.arange(end_trl):
            # Run trial
            run_trial(hrl, trl)

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
