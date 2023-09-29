"""
Some functions used during experiments.
For HRL on Python 2

Functions written over the years at the TU Vision group.
@authors: MM, KZ, CW, TB, GA and LS
"""

from PIL import Image, ImageFont, ImageDraw
from hrl.graphics import graphics
import numpy as np
import sys
import time
from params import bg
import clock
defaultClock = clock.monotonicClock


# Function to read in response:
def read_response(hrl):
    btn = None
    while btn is None or btn == 'Left' or btn == 'Right':
        (btn, t1) = hrl.inputs.readButton()
        if btn == 'Up':
            response = 1
        elif btn == 'Down':
            response = 0
        elif btn == 'Escape' or btn == 'Space':
            print('Escape pressed, exiting experiment!!')
            hrl.close()
            sys.exit(0)
    return response, btn, t1


def read_design(fname, sep=None):
    """
    Reads a design file in sep separated format and stores the design matrix in a dictionary.
    Use sep=None to read from txt and sep=',' to read from csv
    @author: MM.
    """
    design = open(fname)
    header = design.readline().strip('\n').replace('"', '').split(sep)
    data = design.readlines()
    new_data = {}

    for k in header:
        new_data[k] = []
    for line in data:
        curr_line = line.strip('\n').replace('"', '').split(sep)
        for j, k in enumerate(header):
            new_data[k].append(curr_line[j])
    return new_data


def get_block_info(order_file, done_file, sess_type, sep=None):
    blockorder = read_design(order_file, sep)

    # Reads blocks that are already done
    try:
        blocksdone = read_design(done_file)

        # Determine which blocks are left to run
        next = len(blocksdone['number'])
        print(next)
        if sess_type == 'experiment':
            blockstorun = {'number': blockorder['number'][next:],
                           'noisetype': blockorder['noisetype'][next:],
                           'edge_width': blockorder['edge_width'][next:]}
        elif sess_type == 'staircase':
            blockstorun = {'number': blockorder['number'][next:],
                           'noisetype': blockorder['noisetype'][next:],
                           'edge_width': blockorder['edge_width'][next:],
                           'init_edge_contrast': blockorder['init_edge_contrast'][next:]}

    except IOError:
        blocksdone = None
        blockstorun = blockorder  # all blocks to run

    # If all is done
    if len(blockstorun['number']) == 0:
        print("All BLOCKS are DONE, exiting.")
    return blockstorun, blocksdone


def draw_text(text, bg=bg, text_color=0, fontsize=48):
    """
    Create a numpy array containing the string text as an image.
    @author: TB
    """
    bg *= 255
    text_color *= 255
    font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf", fontsize, encoding="unic")
    text_width, text_height = font.getsize(text)
    im = Image.new("L", (text_width, text_height), int(bg))
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), text, fill=text_color, font=font)
    return np.array(im) / 255.0


# Function to read in last trial that was done:
def get_last_trial(filepath):
    try:
        rfl = open(filepath, 'r')
    except IOError:
        print('Result file not found')
        return 0

    line_count = 0
    for line in rfl:
        if line != "\n":
            line_count += 1

    rfl.close()
    return line_count


# Show pause script and wait for continuation:
def show_continue(hrl, b, nb):
    # Function from Torsten
    hrl.graphics.flip(clr=True)
    lines = [u'You can take a break now.',
             u' ',
             u'You have completed %d out of %d blocks.' % (b, nb),
             u' ',
             u'To continue, press the right button,',
             u'to finish, press the left button.'
             ]

    for line_nr, line in enumerate(lines):
        textline = hrl.graphics.newTexture(draw_text(line, fontsize=36, bg=bg))
        textline.draw(((1024 - textline.wdth) / 2,
                       (768 / 2 - (4 - line_nr) * (textline.hght + 10))))
    hrl.graphics.flip(clr=True)
    btn = None
    while not (btn == 'Left' or btn == 'Right'):
        (btn, t1) = hrl.inputs.readButton()

    # clean text
    graphics.deleteTextureDL(textline._dlid)
    return btn


def show_instruction(hrl, sess_type):
    # Function from Torsten
    hrl.graphics.flip(clr=True)

    if sess_type == 'warmup':
        lines = [u'This is the warm-up phase.',
                 u'In each trial, you see an image with an edge.',
                 u'Decide whether it is above or below center via button-press.',
                 u' ',
                 u'Up button: Edge is above the center.',
                 u'Down button: Edge is below the center.',
                 u' ',
                 u'Press right button to continue.',
                 ]
    elif sess_type == 'staircase':
        lines = [u'This is the second phase.',
                 u'In each trial, you see an image with an edge.',
                 u'Decide whether it is above or below center via button-press.',
                 u' ',
                 u'Up button: Edge is above the center.',
                 u'Down button: Edge is below the center.',
                 u' ',
                 u'Press right button to continue.',
                 ]

    for line_nr, line in enumerate(lines):
        textline = hrl.graphics.newTexture(draw_text(line, fontsize=36, bg=bg))
        textline.draw(((1024 - textline.wdth) / 2,
                       (768 / 2 - (4 - line_nr) * (textline.hght + 10))))
    hrl.graphics.flip(clr=True)
    btn = None
    while not (btn == 'Right'):
        (btn, t1) = hrl.inputs.readButton()

    # clean text
    graphics.deleteTextureDL(textline._dlid)
    return btn


def timing(lastFrameT, firsttimeon, frameIntervals, nDroppedFrames, refreshThreshold):
    """ Logs current time and checks if frames have been dropped. It also
    accummulates the frame intervals for later used if needed
    """
    frameTime = defaultClock.getTime()

    if firsttimeon:
        firsttimeon = False
        lastFrameT = frameTime
    else:
        deltaT = frameTime - lastFrameT
        lastFrameT = frameTime
        frameIntervals.append(deltaT)
        # throw warning if dropped frame
        if deltaT > refreshThreshold:
            nDroppedFrames += 1
            msg = 'Drop frame detected. t since last frame: %.2f ms' % (deltaT*1000)
            print(msg)
    return (lastFrameT, firsttimeon, frameIntervals, nDroppedFrames)


def measure_refresh_rate(hrl, monitor_rate):
    """ Measures refresh rate """
    # measuring frame rate
    nIdentical = 20
    nMaxFrames = 2 * monitor_rate
    nWarmUpFrames = monitor_rate
    threshold = 1
    refreshThreshold = 1.0

    # run warm-ups
    for frameN in range(nWarmUpFrames):
        hrl.graphics.flip()

    # run test frames
    firsttimeon = True
    frameIntervals = []
    rate = 0
    for frameN in range(nMaxFrames):

        hrl.graphics.flip(clr=True)

        frameTime = defaultClock.getTime()

        if firsttimeon:
            firsttimeon = False
            lastFrameT = frameTime

        else:
            deltaT = frameTime - lastFrameT
            lastFrameT = frameTime
            frameIntervals.append(deltaT)

        if (len(frameIntervals) >= nIdentical and
                (np.std(frameIntervals[-nIdentical:]) < (threshold / 1000.0))):
            rate = 1.0 / np.mean(frameIntervals[-nIdentical:])

    # print frameIntervals
    val_mean = 1. / (np.mean(frameIntervals[-nIdentical:]))
    val_std = np.std(1. / np.array(frameIntervals[-nIdentical:]))
    print("Measured refresh rate")
    print("%f +- %f (mean +- 1 SD)" % (val_mean, val_std))

    # setting threshold for dropped frames detection
    refreshThreshold = 1.0 / rate * 1.2

    return (rate, refreshThreshold)


def present_trial(hrl, stim, sp, coords, sounds, inlab, feedback=False):
    # timing variables
    dropFrames = 0
    firstOn = True
    frameInts = []
    lastFrame = None
    refreshThresh = sp.refreshThreshold

    # Fixation dots and texture creation in buffer:
    fixdot1 = hrl.graphics.newTexture(np.ones((10, 10)) * 0.0)
    fixdot2 = hrl.graphics.newTexture(np.ones((10, 10)) * 0.6)

    # Add markers to indicate center:
    stim = np.pad(stim, 10, mode="constant", constant_values=sp.mean_int)
    stim[int(stim.shape[0]/2), 0:10] = 0.
    stim[int(stim.shape[0]/2), -10:-1] = 0.

    # Pre-stimulus interval:
    if inlab:
        for t in range(int(sp.prestim * sp.monitor_rate)):
            # Draw fixation dot in the middle
            fixdot1.draw((coords[0] - fixdot1.wdth/2, coords[1] - fixdot1.hght/2))
            hrl.graphics.flip(clr=True)

    else:
        fixdot1.draw((coords[0] - fixdot1.wdth/2, coords[1] - fixdot1.hght/2))
        hrl.graphics.flip()
        time.sleep(sp.prestim + 0.2)
        hrl.graphics.flip(clr=True)

    # Create all stimulus textures.
    # NOTE: since this might take some time, we already show the fixation dot prior to this step
    if inlab:
        stims = []
        hann3d = np.expand_dims(np.expand_dims(sp.hann, 0), 0)
        stim = np.expand_dims(stim-stim.mean(), -1) * hann3d + stim.mean()
        for t in range(len(sp.hann)):
            stims.append(hrl.graphics.newTexture(stim[:, :, t]))
    else:
        stims = hrl.graphics.newTexture(stim)

    # Stimulus interval
    sounds[0].play(loops=0, maxtime=int(sp.stim_time*1000))
    if inlab:
        for t in range(len(sp.hann)):
            # Fade-in stimulus
            stims[t].draw((coords[0] - stims[t].wdth / 2, coords[1] - stims[t].hght / 2))
            hrl.graphics.flip(clr=True)
            lastFrame, firstOn, frameInts, dropFrames = timing(lastFrame, firstOn, frameInts, dropFrames, refreshThresh)
        for t in range(int(sp.stim_time * sp.monitor_rate)):
            # Present full-contrast stimulus for stim_time
            stims[-1].draw((coords[0] - stims[-1].wdth / 2, coords[1] - stims[-1].hght / 2))
            hrl.graphics.flip(clr=True)
        for t in range(len(sp.hann)):
            # Fade-out stimulus
            stims[-t-1].draw((coords[0] - stims[-t-1].wdth / 2, coords[1] - stims[-t-1].hght / 2))
            hrl.graphics.flip(clr=True)
            lastFrame, firstOn, frameInts, dropFrames = timing(lastFrame, firstOn, frameInts, dropFrames, refreshThresh)

    else:
        # Show stimulus for duration length
        stims.draw((coords[0] - stims.wdth / 2, coords[1] - stims.hght / 2))
        hrl.graphics.flip(clr=True)
        time.sleep(sp.stim_time)
        hrl.graphics.flip(clr=True)

    # Response interval
    # Loop for waiting for a response + show a bright fixation dot during that time
    fixdot2.draw((coords[0] - fixdot2.wdth/2, coords[1] - fixdot2.hght/2))
    hrl.graphics.flip()
    no_resp = True
    while no_resp:  # as long as no_resp TRUE
        resp, btn, resp_time = read_response(hrl)
        print("Response:", btn)
        if resp is not None:
            no_resp = False
    hrl.graphics.flip(clr=True)

    # Provide feedback:
    if feedback:
        if resp == sp.transID:
            sounds[2].play(loops=0, maxtime=100)  # maxtime in ms
        else:
            sounds[3].play(loops=0, maxtime=100)  # maxtime in ms

    # Intertrial interval
    if inlab:
        for t in range(int((sp.intertrial+0.25) * sp.monitor_rate)):
            pass
    else:
        time.sleep(sp.intertrial + 0.4)
        hrl.graphics.flip(clr=True)

    # Clean textures from buffer
    graphics.deleteTextureDL(fixdot1._dlid)
    graphics.deleteTextureDL(fixdot2._dlid)
    graphics.deleteTextureDL(fixdot1._txid)
    graphics.deleteTextureDL(fixdot2._txid)
    if inlab:
        for t in range(len(sp.hann)):
            graphics.deleteTextureDL(stims[t]._dlid)
            graphics.deleteTexture(int(stims[t]._txid))
    else:
        graphics.deleteTextureDL(stims._dlid)
        graphics.deleteTexture(stims._txid)

    # Subtract number of to-be-expected dropframes
    if inlab:
        dropFrames = dropFrames - 1
    return resp, resp_time, dropFrames


class staircase:
    def __init__(self,
                 hrl,
                 n_back,
                 n_reversals,
                 x_init,
                 run_trial,
                 xmin=None,
                 xmax=None,
                 n_adapt=None,
                 params_dict=None):

        self.hrl = hrl
        self.n_back = n_back             # nback for n-down-1up staircase
        self.n_reversals = n_reversals   # number of reversals until finished
        self.x_init = x_init
        self.step_size = x_init / 10     # initial step size
        self.run_trial = run_trial       # Function definition to run trial
        self.xmin = xmin                 # min intensity to use
        self.xmax = xmax                 # max intensity to use
        self.n_adapt = n_adapt           # number of reversals after which to divide step size
        self.params_dict = params_dict   # additional params for running trials

    def run(self, trl, x, responses, r_count, r_direction, r_vals, params_dynamic=None):
        r, params_dynamic = self.run_trial(self.hrl, x, self.params_dict, params_dynamic)
        responses = np.append(responses, r)

        # If reversal occurs, increase r_counter and save r_value
        if (
                (trl > self.n_back) &
                (responses[trl-self.n_back:trl].sum() == self.n_back) &
                (responses[trl] == 0.) & (r_direction == 1)
                ):
            r_count += 1
            r_direction = -1
            r_vals = np.append(r_vals, x)

            # After n_adapt reversals, divide step size by two
            if self.n_adapt is not None:
                if (r_count % self.n_adapt == 0) & (r_count != 0):
                    self.step_size = self.step_size / 2.

        elif (
                (trl > self.n_back) &
                (responses[trl - 1] == 0.) &
                (responses[trl] == 1.) &
                (r_direction == -1)
                ):
            r_count += 1
            r_direction = 1
            r_vals = np.append(r_vals, x)

            # After n_adapt reversals, divide step size by two
            if self.n_adapt is not None:
                if (r_count % self.n_adapt == 0) & (r_count != 0):
                    self.step_size = self.step_size / 2.

        # If response was correct for at least n_back times in a row, reduce the threshold
        if (trl < self.n_back) or (responses[trl-self.n_back+1:trl+1].mean() == 1.):
            x -= self.step_size
        # Otherwise, increase the threshold
        else:
            x += self.step_size

        # x should be between given limits
        if (self.xmin is not None):
            if (x < self.xmin):
                x = self.xmin
        if (self.xmax is not None):
            if (x > self.xmax):
                x = self.xmax

        # Increase trial counter
        trl += 1
        print('Trial: %d, Response: %d, new x: %f' % (trl, r, x))

        r_left = self.n_reversals - r_count
        return trl, x, responses, r_count, r_direction, r_vals, params_dynamic, r_left
