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

from experiment_functions import read_design, present_trial, show_instruction, measure_refresh_rate
from helper_functions import make_sound
from stimulus_functions import get_stim_params, prepare_stim
from params import lut, bg, n_masks


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
    # pygame.init()
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
    edge_width = float(design['edge_width'][trl])
    edge_contrast = float(design['edge_contrast'][trl])
    stim_time = float(design['stim_time'][trl])
    noise_type = str(design['noisetype'][trl])
    maskID = np.random.randint(0, n_masks)

    # Randomize polarity and translation of edge stimulus
    rotID = np.random.randint(0, 2)
    transID = np.random.randint(0, 2)

    # Create stim params object (sp) and create stimulus
    sp = get_stim_params(edge_width, edge_contrast, noise_type, stim_time,
                         rotID, transID, maskID, monitor_rate, refreshThreshold)
    stim = prepare_stim(sp)

    print('--------------------------------------')
    print("TRIAL =", trl)
    print('Noise type: %s, contrast: %f' % (noise_type, sp.noise_contrast))
    print('Edge contrast: %f, width: %f' % (edge_contrast, sp.edge_width))
    print('Mean lum: %f, stim duration: %f' % (sp.mean_lum, stim_time))
    print('Translation: %d, rotation: %d' % (transID, rotID))

    # Present trial
    resp, rtime, dropframes = present_trial(hrl, stim, sp, [wh, hh], sounds, inlab, feedback=True)

    # Calculate response so that 0:incorrect and 1:correct
    correct = np.abs(resp - transID)
    correct = np.abs(correct - 1)
    print(['Incorrect!', 'Correct!'][correct])

    # Save trial data
    data_row = (trl, noise_type, sp.noise_contrast, edge_contrast, sp.edge_width, sp.mean_lum,
                stim_time, rotID, transID, maskID, resp, correct, rtime, dropframes)
    rfl.write('%d\t%s\t%f\t%f\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%f\t%d\n' % data_row)
    rfl.flush()


############################################
#                Run main                  #
############################################
if __name__ == '__main__':
    result_headers = ['trial', 'noise', 'noise_contrast', 'edge_contrast', 'edge_width', 'mean_lum',
                      'stim_time', 'rotID', 'transID', 'maskID', 'resp', 'correct', 'resptime',
                      'dropframes']

    # Create results folder for participant
    results_dir = './results/%s/' % vp_id
    if os.path.exists(results_dir) == 0:
        os.makedirs(results_dir)

    # Log file name and location
    design = read_design('./design/%s/warmup.csv' % vp_id, ',')
    end_trl = len(design['trial'])  # Get end trial
    rfl = open(results_dir + '/warmup.txt', 'a')
    rfl.write('\t'.join(result_headers) + '\n')

    # Check refresh rate of monitor
    rate, refreshThreshold = measure_refresh_rate(hrl, monitor_rate)
    if inlab:
        # if refresh rate is out of range, through error and exits
        if rate < (monitor_rate-tol_rate) or rate > (monitor_rate+tol_rate):
            raise(RuntimeError, 'Monitor refresh rate is out of tolerated range')

    # Display instructions
    show_instruction(hrl, 'warmup')

    # Loop over one all warmup trials:
    for trl in np.arange(end_trl):
        # Run trial
        run_trial(hrl, trl)

    # Close result file for this block
    rfl.close()

    # Finishes everything
    hrl.close()
    print("Session finished")
