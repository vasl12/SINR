import nibabel as nib
import numpy as np


def load_nifti(path, data_type=np.float32, nim=False):
    xnim = nib.load(path)
    x = np.asanyarray(xnim.dataobj).astype(data_type)
    if nim:
        return x, xnim
    else:
        return x


def save_nifti(x, path, nim=None, verbose=False):
    """
    Save a numpy array to a nifti file

    Args:
        x: (numpy.ndarray) data
        path: destination path
        nim: Nibabel nim object, to provide the nifti header
        verbose: (boolean)

    Returns:
        N/A
    """
    if nim is not None:
        nim_save = nib.Nifti1Image(x, nim.affine, nim.header)
    else:
        nim_save = nib.Nifti1Image(x, np.eye(4))
    nib.save(nim_save, path)

    if verbose:
        print("Nifti saved to: {}".format(path))
