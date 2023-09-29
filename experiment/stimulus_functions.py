"""
Util functions to create different noise distributions and to create the Cornsweet edge
@author: Lynn Schmittwilken, Nov 2021
"""

import numpy as np
from params import stim_params, exp_params, lut_params
from helper_functions import load_mask


###################################
#        Helper functions         #
###################################
# Class with stimulus parameters
class get_stim_params(object):
    def __init__(self, edge_width=None, edge_contrast=None, noise_type=None, stim_time=None,
                 rotID=None, transID=None, maskID=None, monitor_rate=None, refreshThreshold=None):
        # Constant stimulus params
        self.stim_size = stim_params["stim_size"]
        self.ppd = stim_params["ppd"]
        self.mean_lum = stim_params["mean_lum"]
        self.mean_int = stim_params["mean_int"]
        self.edge_exponent = stim_params["edge_exponent"]
        self.noise_contrast = stim_params["noise_contrast"]
        self.mask_path = str(stim_params["mask_path"]) + "/"

        # Variable stimulus params
        self.edge_width = edge_width
        self.edge_contrast = edge_contrast
        self.noise_type = noise_type
        self.rotID = rotID
        self.transID = transID
        self.maskID = maskID

        # Experimental params
        self.trans_amount = exp_params["trans_amount"]
        self.prestim = exp_params["prestim"]
        self.stimfading = exp_params["stimfading"]
        if stim_time is None:
            self.stim_time = exp_params["stim_time"]
        else:
            self.stim_time = stim_time
        self.intertrial = exp_params["intertrial"]
        if monitor_rate is not None:
            hann = np.hanning(self.stimfading*monitor_rate)
            self.hann = hann[0:int(self.stimfading*monitor_rate/2.)]
        
        # Other information
        self.monitor_rate = monitor_rate
        self.refreshThreshold = refreshThreshold


# Flip or rotate the stimulus:
def rotate_stimulus(stim, rotID):
    if rotID == 0:
        pass
    elif rotID == 1:
        stim = np.flipud(stim)
    elif rotID == 2:
        stim = np.rot90(stim)
    elif rotID == 3:
        stim = np.flipud(stim)
        stim = np.rot90(stim)
    else:
        raise ValueError("rotID does not exist")
    return stim


def translate_stimulus(stim, trans_ID, amount):
    if trans_ID == 0:
        # Translate up
        stim = np.roll(stim, int(amount), axis=0)
    elif trans_ID == 1:
        # Translate down
        stim = np.roll(stim, int(-amount), axis=0)
    else:
        raise ValueError("trans_ID does not exist")
    return stim


def clip_values(array, amin, amax):
    array[array < amin] = amin
    array[array > amax] = amax
    return array


def pull_noise(sp):
    if sp.noise_type == 'none':
        noise = np.zeros([int(sp.stim_size * sp.ppd), int(sp.stim_size * sp.ppd)])
    elif sp.noise_type == 'white':
        noise, _ = load_mask(sp.mask_path + "white/" + str(sp.maskID) + ".pickle")
    elif sp.noise_type == 'pink1':
        noise, _ = load_mask(sp.mask_path + "pink1/" + str(sp.maskID) + ".pickle")
    elif sp.noise_type == 'pink2':
        noise, _ = load_mask(sp.mask_path + "pink2/" + str(sp.maskID) + ".pickle")
    elif sp.noise_type == 'narrow0.5':
        noise, _ = load_mask(sp.mask_path + "narrow0.5/" + str(sp.maskID) + ".pickle")
    elif sp.noise_type == 'narrow3':
        noise, _ = load_mask(sp.mask_path + "narrow3/" + str(sp.maskID) + ".pickle")
    elif sp.noise_type == 'narrow9':
        noise, _ = load_mask(sp.mask_path + "narrow9/" + str(sp.maskID) + ".pickle")
    else:
        raise ValueError("noise_type unknown")
    return noise


# Prepare stimulus for experiment (sp: stim_params, lp: lut_params)
def prepare_stim(sp):
    # Create Cornsweet edge (on the fly because of variable contrast)
    stim = cornsweet_illusion(size=sp.stim_size,
                              ppd=sp.ppd,
                              contrast_type='rms',
                              contrast=sp.edge_contrast,
                              ramp_width=sp.edge_width,
                              mean_lum=sp.mean_int,
                              exponent=sp.edge_exponent)

    # Adapt the intensities to compensate that the monitor cannot display 0 luminance
    stim = stim - lut_params.b / lut_params.a

    # By default, rotate stimuli from vertical to horizontal:
    stim = np.rot90(stim)

    # Randomly flip polarity and show stimulus below or above the center
    stim = rotate_stimulus(stim, sp.rotID)
    stim = translate_stimulus(stim, sp.transID, sp.trans_amount)

    # Pull noise_mask (constant contrast)
    noise = pull_noise(sp)

    # Sum edge+noise into final stimulus
    edge_stim = stim + noise

    # In rare occasions, single pixels can have negative luminances.
    # To compensate, clip pixels equally in both directions
    edge_stim = clip_values(edge_stim, 0, sp.mean_int*2)

    # Double-check that stimulus intensities are inside 0-1 range
    if (edge_stim.max() > 1.) or (edge_stim.min() < 0.):
        raise RuntimeError('stimulus values out of 0-1 range')
    return edge_stim


###################################
#         Noise functions         #
###################################
def bandpass_filter(fx, fy, fcenter, sigma):
    """Function to create a bandpass filter

    Parameters
    ----------
    fx
        Array with frequencies in x-direction.
    fy
        Array with frequencies in y-direction.
    fcenter
        Center frequency of the bandpass filter
    sigma
        Sigma that defines the spread of the Gaussian in deg.

    Returns
    -------
    bandpass
        2D bandpass filter in frequency domain.

    """
    # Calculate the distance of each 2d spatial frequency from requested center frequency
    distance = np.abs(fcenter - np.sqrt(fx**2. + fy**2.))

    # Create bandpass filter:
    bandpass = 1. / (np.sqrt(2.*np.pi) * sigma) * np.exp(-(distance**2.) / (2.*sigma**2.))
    bandpass = bandpass / bandpass.max()
    return bandpass


def randomize_sign(array):
    """Helper function that randomizes the sign of values in an array.

    Parameters
    ----------
    array
        N-dimensional array

    Returns
    -------
    array
        Same array with randomized signs

    """
    sign = np.random.rand(*array.shape) - 0.5
    sign[sign <= 0.] = -1.
    sign[sign > 0.] = 1.
    array = array * sign
    return array


def pseudo_white_noise_patch(shape, A):
    """Helper function used to generate pseudorandom white noise patch.

    Parameters
    ----------
    shape
        Shape of noise patch
    A
        Amplitude of each (pos/neg) frequency component = A/2

    Returns
    -------
    output
        Pseudorandom white noise patch

    """
    Re = np.random.rand(*shape) * A - A/2.
    Im = np.sqrt((A/2.)**2 - Re**2)
    Im = randomize_sign(Im)
    output = Re+Im*1j
    return output


def pseudo_white_noise(n, A=2.):
    """Function to create pseudorandom white noise. Code translated and adapted
    from Matlab scripts provided by T. Peromaa

    Parameters
    ----------
    n
        Even-numbered size of output
    A
        Amplitude of noise power spectrum

    Returns
    -------
    spectrum
        Shifted 2d complex number spectrum. DC = 0.
        Amplitude of each (pos/neg) frequency component = A/2
        Power of each (pos/neg) frequency component = (A/2)**2

    """
    # We divide the noise spectrum in four quadrants with pseudorandom white noise
    quadrant1 = pseudo_white_noise_patch((int(n/2)-1, int(n/2)-1), A)
    quadrant2 = pseudo_white_noise_patch((int(n/2)-1, int(n/2)-1), A)
    quadrant3 = quadrant2[::-1, ::-1].conj()
    quadrant4 = quadrant1[::-1, ::-1].conj()

    # We place the quadrants in the spectrum to eventuate that each frequency component has
    # an amplitude of A/2
    spectrum = np.zeros([n, n], dtype=complex)
    spectrum[1:int(n/2), 1:int(n/2)] = quadrant1
    spectrum[1:int(n/2), int(n/2)+1:n] = quadrant2
    spectrum[int(n/2+1):n, 1:int(n/2)] = quadrant3
    spectrum[int(n/2+1):n, int(n/2+1):n] = quadrant4

    # We need to fill the rows / columns that the quadrants do not cover
    # Fill first row:
    row = pseudo_white_noise_patch((1, n), A)
    apu = np.fliplr(row)
    row[0, int(n/2+1):n] = apu[0, int(n/2):n-1].conj()
    spectrum[0, :] = np.squeeze(row)

    # Fill central row:
    row = pseudo_white_noise_patch((1, n), A)
    apu = np.fliplr(row)
    row[0, int(n/2+1):n] = apu[0, int(n/2):n-1].conj()
    spectrum[int(n/2), :] = np.squeeze(row)

    # Fill first column:
    col = pseudo_white_noise_patch((n, 1), A)
    apu = np.flipud(col)
    col[int(n/2+1):n, 0] = apu[int(n/2):n-1, 0].conj()
    spectrum[:, int(n/2)] = np.squeeze(col)

    # Fill central column:
    col = pseudo_white_noise_patch((n, 1), A)
    apu = np.flipud(col)
    col[int(n/2+1):n, 0] = apu[int(n/2):n-1, 0].conj()
    spectrum[:, 0] = np.squeeze(col)

    # Set amplitude at filled-corners to A/2:
    spectrum[0, 0] = -A/2 + 0j
    spectrum[0, int(n/2)] = -A/2 + 0j
    spectrum[int(n/2), 0] = -A/2 + 0j

    # Set DC = 0:
    spectrum[int(n/2), int(n/2)] = 0 + 0j
    return spectrum


def create_whitenoise(size, rms_contrast=0.2, pseudo_noise=True):
    """Function to create white noise.

    Parameters
    ----------
    size
        Size of noise image in pixels.
    rms_contrast
        rms contrast of noise.
    pseudo_noise
        Bool, if True generate pseudorandom noise with smooth power spectrum.

    Returns
    -------
    white_noise
        2D array with white noise.

    """
    size = int(size)

    if pseudo_noise:
        # Create white noise with frequency amplitude of 1 everywhere
        white_noise_fft = pseudo_white_noise(size)

        # ifft
        white_noise = np.fft.ifft2(np.fft.ifftshift(white_noise_fft))
        white_noise = np.real(white_noise)
    else:
        # Create white noise and fft
        white_noise = np.random.rand(size, size) * 2. - 1.

    # Adjust noise rms contrast:
    white_noise = rms_contrast * white_noise / white_noise.std()
    return white_noise


def create_narrownoise(size, noisefreq, ppd=60., rms_contrast=0.2, pseudo_noise=True):
    """Function to create narrowband noise.

    Parameters
    ----------
    size
        Size of noise image in pixels.
    noisefreq
        Noise center frequency in cpd.
    ppd
        Spatial resolution (pixels per degree).
    rms_contrast
        rms contrast of noise.
    pseudo_noise
        Bool, if True generate pseudorandom noise with smooth power spectrum.

    Returns
    -------
    narrow_noise
        2D array with narrowband noise.

    """
    size = int(size)

    # We calculate sigma to eventuate a ratio bandwidth of 1 octave
    sigma = noisefreq / (3.*np.sqrt(2.*np.log(2.)))

    # Prepare spatial frequency axes and create bandpass filter:
    fs = np.fft.fftshift(np.fft.fftfreq(size, d=1./ppd))
    fx, fy = np.meshgrid(fs, fs)
    bp_filter = bandpass_filter(fx, fy, noisefreq, sigma)

    if pseudo_noise:
        # Create white noise with frequency amplitude of 1 everywhere
        white_noise_fft = pseudo_white_noise(size)
    else:
        # Create white noise and fft
        white_noise = np.random.rand(size, size) * 2. - 1.
        white_noise_fft = np.fft.fftshift(np.fft.fft2(white_noise))

    # Filter white noise with bandpass filter
    narrow_noise_fft = white_noise_fft * bp_filter

    # ifft
    narrow_noise = np.fft.ifft2(np.fft.ifftshift(narrow_noise_fft))
    narrow_noise = np.real(narrow_noise)

    # Adjust noise rms contrast:
    narrow_noise = rms_contrast * narrow_noise / narrow_noise.std()
    return narrow_noise


def create_pinknoise(size, ppd=60., rms_contrast=0.2, exponent=2., pseudo_noise=True):
    """Function to create 1/f**exponent noise.

    Parameters
    ----------
    size
        Size of noise image in pixels.
    ppd
        Spatial resolution (pixels per degree).
    rms_contrast
        rms contrast of noise.
    exponent
        Exponent used to create 1/f**exponent noise
    pseudo_noise
        Bool, if True generate pseudorandom noise with smooth power spectrum.

    Returns
    -------
    pink_noise
        2D array with 1/f**exponent noise.

    """
    size = int(size)

    # Prepare spatial frequency axes and create bandpass filter:
    fs = np.fft.fftshift(np.fft.fftfreq(size, d=1./ppd))
    fx, fy = np.meshgrid(fs, fs)

    # Needed to create 2d 1/f**exponent noise. Prevent division by zero.
    # Note: The noise amplitude at DC is 0.
    f = np.sqrt(fx**2. + fy**2.)
    f = f**exponent
    f[f == 0.] = 1.

    if pseudo_noise:
        # Create white noise with frequency amplitude of 1 everywhere
        white_noise_fft = pseudo_white_noise(size)
    else:
        # Create white noise and fft
        white_noise = np.random.rand(size, size) * 2. - 1.
        white_noise_fft = np.fft.fftshift(np.fft.fft2(white_noise))

    # Create 1/f noise:
    pink_noise_fft = white_noise_fft / f

    # ifft
    pink_noise = np.fft.ifft2(np.fft.ifftshift(pink_noise_fft))
    pink_noise = np.real(pink_noise)

    # Adjust noise rms contrast:
    pink_noise = rms_contrast * pink_noise / pink_noise.std()
    return pink_noise


###################################
#          Edge stimulus          #
###################################
def cornsweet_illusion(size=10., ppd=10., contrast_type='mc', contrast=0.1, ramp_width=2.,
                       exponent=2., mean_lum=.5):
    """Function to create 1/f**exponent noise.

    Parameters
    ----------
    size
        Size of noise image in degrees visual angle.
    ppd
        Spatial resolution (pixels per degree).
    contrast_type : str
        either mc for Michelson contrast or rms for rms-contrast
    contrast
        contrast of noise.
    ramp_width
        Width of ramp in degrees visual angle
    exponent
        exponent used to define the ramp. An exponent of 1 generates a linear ramp.
    mean_lum
        Mean luminance used.

    Returns
    -------
    img
        2D array with Cornsweet illusion.

    """
    size = int(size*ppd)
    img = np.ones([size, size]) * mean_lum
    dist = np.arange(size / 2.)
    dist = dist / (ramp_width*ppd)
    dist[dist > 1.] = 1.
    profile = (1. - dist) ** exponent * mean_lum * contrast
    img[:, :int(np.ceil(size/2.))] += profile[::-1]
    img[:, size // 2:] -= profile

    if contrast_type == 'rms':
        # Adjust rms contrast (std divided by mean_lum):
        img = img - img.mean()
        img = contrast * mean_lum * img / img.std() + mean_lum
    return img
