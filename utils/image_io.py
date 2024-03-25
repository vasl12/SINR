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


def split_volume_idmat(image_path, output_prefix, data_type=np.float32):
    """ Split an image volume into slices with identity matrix as I2W transformation
    This is to by-pass the issue of not accumulating displacement in z-direction
     in MIRTK's `convert-dof` function
     """
    # image data saved in shape (H, W, N)
    nim = nib.load(image_path)
    Z = nim.header['dim'][3]
    image = np.asanyarray(nim.dataobj).astype(data_type)

    for z in range(Z):
        image_slice = image[:, :, z]
        image_slice = np.expand_dims(image_slice, axis=2)
        nim2 = nib.Nifti1Image(image_slice, np.eye(4))
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_prefix, z))