"""
Scripts to create noises for frozen noise experiment
@author: Lynn Schmittwilken, Nov 2022
"""

from params import mask_path, stim_params
from stimulus_functions import create_whitenoise, create_pinknoise, create_narrownoise
from helper_functions import create_directory, save_mask


if __name__ == "__main__":
    sp = stim_params
    stim_size = sp["stim_size"]
    ppd = sp["ppd"]
    rms = sp["noise_contrast"] * sp["mean_int"]
    n_masks = sp["n_masks"]
    file_path = mask_path
    create_directory(file_path)

    # Create and save white noise
    noise_dir = file_path + sp["noise_types"][1] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        noise = create_whitenoise(size=stim_size*ppd, rms_contrast=rms)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")

    # Create pink1 noise
    noise_dir = file_path + sp["noise_types"][2] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        noise = create_pinknoise(size=stim_size*ppd, ppd=ppd, rms_contrast=rms, exponent=1.)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")

    # Create pink2 / brown noise
    noise_dir = file_path + sp["noise_types"][3] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        noise = create_pinknoise(size=stim_size*ppd, ppd=ppd, rms_contrast=rms, exponent=2.)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")

    # Create narrowband noise with center frequency of 0.5 cpd
    noise_dir = file_path + sp["noise_types"][4] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        noise = create_narrownoise(size=stim_size*ppd, noisefreq=0.5, ppd=ppd, rms_contrast=rms)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")

    # Create narrowband noise with center frequency of 3. cpd
    noise_dir = file_path + sp["noise_types"][5] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        noise = create_narrownoise(size=stim_size*ppd, noisefreq=3., ppd=ppd, rms_contrast=rms)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")

    # Create narrowband noise with center frequency of 9. cpd
    noise_dir = file_path + sp["noise_types"][6] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        noise = create_narrownoise(size=stim_size*ppd, noisefreq=9., ppd=ppd, rms_contrast=rms)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")
