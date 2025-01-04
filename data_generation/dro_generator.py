import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
import itertools
import json

# Set of Default Parameters
default_parameters = {
    # Size Features
    "mean_radius": [100, 100, 1],
    # Shape Features
    "x_deformation": [1, 1, 1],
    "y_deformation": [1, 1, 1],
    "z_deformation": [1, 1, 1],
    "surface_frequency": [0, 0, 1],
    "surface_amplitude": [0, 0, 1],
    # Intensity Features
    "mean_intensity": [100, 100, 1],
    # Texture Features
    "texture_wavelength": [0, 0, 1],
    "texture_amplitude": [0, 0, 1],
    # Margin Features
    "gaussian_standard_deviation": [0, 0, 1],
}

# Keys in the Order for the DRO Name
ordered_keys = [
    "mean_radius",
    "x_deformation",
    "y_deformation",
    "z_deformation",
    "surface_frequency",
    "surface_amplitude",
    "mean_intensity",
    "texture_wavelength",
    "texture_amplitude",
    "gaussian_standard_deviation",
]


# expand_range
# Takes:    dictionary of parameters
# Does:     expands the min, max, number of values into array of values at equal intervals
# Returns:  dictionary of parameters with full arrays of values
def expand_range(dic):
    expanded = {}
    for key in dic.keys():
        kmax = dic[key][0]
        kmin = dic[key][1]
        knum = dic[key][2]
        expanded[key] = frange(kmin, kmax, knum)
    return expanded


# generate_params
# Takes:    dictionary of parameters with full arrays of values
# Does:     find all combinations of parameters of all ranges of values
# Returns:  array of all combinations of parameters
def generate_params(dic):
    params = []
    for key in ordered_keys:
        params.append(dic[key])
    params = list(itertools.product(*params))
    params = [list(p) for p in params]
    return params


def get_single_dro(arguments):
    arguments = [float(arg) for arg in arguments]
    global r, xx, yy, zz, shape_freq, shape_amp, avg, text_wav, text_amp, decay
    r, xx, yy, zz, shape_freq, shape_amp, avg, text_wav, text_amp, decay = arguments
    mask, output_array = generate_dro()
    return output_array, mask


def get_all_dros(params) -> list[tuple[np.ndarray, np.ndarray]]:
    """Creates DROs without writing anything to disk

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: [(DRO, mask)]
    """
    return [get_single_dro(param) for param in params]


# generate_dro
# Takes:    nothing
# Does:     generate dro from its mathematical definition
# Return:   image array embedding the object and mask for the object
def generate_dro():
    n = 300
    s = 512
    # Make 3D Grid
    x = np.linspace(-s / 2, s / 2, s)
    y = np.linspace(-s / 2, s / 2, s)
    z = np.linspace(-n / 2, n / 2, n)
    xt, yt, zt = np.meshgrid(x, y, z, sparse=True)  # xt stands for "x-true"
    if xx != 1 or yy != 1 or zz != 1:
        xs, ys, zs = np.meshgrid(
            1 / float(xx) * x, 1 / float(yy) * y, 1 / float(zz) * z, sparse=True
        )  # xs stands for "x stretch"
    else:
        xs, ys, zs = xt, yt, zt
    # Calculate distance to origin of each point then compare to the shape of the object
    origin = np.sqrt(xs * xs + ys * ys + zs * zs)
    rp = r
    if shape_amp != 0.0 and shape_freq != 0.0:
        rp = r * (
            1
            + shape_amp
            * np.sin(shape_freq * np.arccos(zs / origin))
            * np.cos(shape_freq * np.arctan2(ys, xs))
        )
    mask = rp >= origin
    # Apply Texture
    texture = np.full_like(mask, 1024, dtype=float)
    if text_amp != 0.0 and text_wav != 0.0:
        variation = avg + text_amp * np.cos((1 / text_wav) * 2 * np.pi * xt) * np.cos(
            (1 / text_wav) * 2 * np.pi * yt
        ) * np.cos((1 / text_wav) * 2 * np.pi * zt)
        texture += variation
    else:
        texture += avg
    # Add blurred edge
    if decay != 0:
        big = binary_dilation(mask, iterations=10)
        texture[~big] = 0
        inside = np.copy(texture)
        inside[~mask] = 0
        texture = gaussian_filter(texture, sigma=decay)
        output_array = texture
        texture[mask] = 0
        output_array = inside + texture
    else:
        texture[~mask] = 0
        output_array = texture
    return mask, output_array


# Create a numpy range
def frange(start, stop, step):
    return np.linspace(start, stop, num=step).tolist()


def read_json_cfg(path: str) -> dict:
    params = default_parameters
    with open(path, "r", encoding="utf-8") as cfg_file:
        for key, val in json.load(cfg_file).items():
            params[key] = [val, val, 1]
    return params


def generate_phantom(cfg_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Generates a phantom.

    Args:
        cfg_path (str): path to JSON with custom config.

    Returns:
        tuple[np.ndarray, np.ndarray]: phantom and it's mask.
    """

    full_param_list = generate_params(expand_range(read_json_cfg(cfg_path)))
    return get_single_dro(full_param_list[0])
