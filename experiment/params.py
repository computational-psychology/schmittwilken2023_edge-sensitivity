"""
Overview file with all parameters
@author: Lynn Schmittwilken, Nov 2022
"""

from pathlib import Path
from helper_functions import get_lut_params, lum_to_intensity


###################################
#            General              #
###################################
lut = Path(__file__).parents[0] / "lut_20220425.csv"
mask_path = Path(__file__).parents[0] / "noise_masks/"


###################################
#           Stimulus              #
###################################
stim_size = 4.               # in deg
ppd = 44.                    # pixels per degree (at 100cm)
mean_lum = 100.              # in cd/m**2
edge_widths = [0.048,
               0.15,
               0.95]         # of Cornsweet edges in deg
edge_exponent = 1.           # of Cornsweet edges
noise_contrast = 0.2
noise_types = ["none",
               "white",
               "pink1",
               "pink2",
               "narrow0.5",
               "narrow3",
               "narrow9",
               ]
n_masks = 40

# Staircase
xedge048_init = [0.01, 0.03, 0.03, 0.015, 0.01, 0.02, 0.03]
xedge15_init = [0.01, 0.02, 0.03, 0.015, 0.015, 0.025, 0.015]
xedge95_init = [0.01, 0.02, 0.06, 0.02, 0.015, 0.05, 0.02]


# Experiment
xedge048_min = [0.00001,] * 7
xedge048_max = [0.003, 0.015, 0.015, 0.004, 0.005, 0.006, 0.015]

xedge15_min = [0.00001,] * 7
xedge15_max = [0.004, 0.0125, 0.02, 0.006, 0.003, 0.015, 0.0075]

xedge95_min = [0.00001,] * 7
xedge95_max = [0.005, 0.015, 0.05, 0.01, 0.0075, 0.03, 0.0075]


###################################
#           Experiment            #
###################################
trans_amount = int(ppd * 0.5)   # amount by which edges are translated

# Stimulus timing (in s)
prestim = 0.5
stimfading = 0.2
stim_time = 0.2
intertrial = 0.75

# Staircase
n_reversals = 18
n_adapt = 6

# Experiment
n_reps = 20    # Number of repetitions per contrast
n_steps = 5    # Number of different contrasts
start_easy = True  # if True, add 3 easy trials at the start of each block (not included in data sheet)


###################################
#              Lut                #
###################################
lut_params = get_lut_params(lut)

# Calculate background intensity value properly
bg = lum_to_intensity(mean_lum + lut_params.b, lut_params.a, lut_params.b) - lut_params.b / lut_params.a

# Calculate mean intensity value properly
mean_int = lum_to_intensity(mean_lum + lut_params.b, lut_params.a, lut_params.b)


###################################
#           Summarize             #
###################################
stim_params = {
    "stim_size": stim_size,
    "ppd": ppd,
    "mean_lum": mean_lum,
    "mean_int": mean_int,
    "edge_widths": edge_widths,
    "edge_exponent": edge_exponent,
    "noise_contrast": noise_contrast,
    "noise_types": noise_types,
    "n_masks": n_masks,
    "xedge048_init": xedge048_init,
    "xedge15_init": xedge15_init,
    "xedge95_init": xedge95_init,
    "mask_path": mask_path,
    "xedge048_min": xedge048_min,
    "xedge048_max": xedge048_max,
    "xedge15_min": xedge15_min,
    "xedge15_max": xedge15_max,
    "xedge95_min": xedge95_min,
    "xedge95_max": xedge95_max,
    }

exp_params = {
    "trans_amount": trans_amount,
    "prestim": prestim,
    "stimfading": stimfading,
    "stim_time": stim_time,
    "intertrial": intertrial,
    "n_reversals": n_reversals,
    "n_adapt": n_adapt,
    "n_reps": n_reps,
    "n_steps": n_steps,
    }
