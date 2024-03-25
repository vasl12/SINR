""" Dataset helper functions """
import numpy as np
import torch
from utils.image import crop_and_pad, normalise_intensity
from utils.image_io import load_nifti


def _to_tensor(data_dict):
    # cast to Pytorch Tensor
    for name, data in data_dict.items():
        data_dict[name] = torch.from_numpy(data).float()
    return data_dict


def _crop_and_pad(data_dict, crop_size):
    # cropping and padding
    for name, data in data_dict.items():
        data_dict[name] = crop_and_pad(data, new_size=crop_size)
    return data_dict


def _normalise_intensity(data_dict, keys=None, vmin=0., vmax=1.):
    """ Normalise intensity of data in `data_dict` with `keys` """
    if keys is None:
        keys = {'fixed', 'moving', 'fixed_original'}

    # images in one pairing should be normalised using the same scaling
    vmin_in = np.amin(np.array([data_dict[k] for k in keys]))
    vmax_in = np.amax(np.array([data_dict[k] for k in keys]))

    for k, x in data_dict.items():
        if k in keys:
            data_dict[k] = normalise_intensity(x,
                                               min_in=vmin_in, max_in=vmax_in,
                                               min_out=vmin, max_out=vmax,
                                               mode="minmax", clip=True)
    return data_dict


def _load3d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
        data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
    return data_dict

