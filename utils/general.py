import numpy as np
import torch
import omegaconf
import matplotlib.pyplot as plt


def normalise_intensity(x,
                        mode="minmax",
                        min_in=0.0,
                        max_in=255.0,
                        min_out=0.0,
                        max_out=1.0,
                        clip=False,
                        clip_range_percentile=(0.05, 99.95),
                        ):
    """
    Intensity normalisation (& optional percentile clipping)
    for both Numpy Array and Pytorch Tensor of arbitrary dimensions.

    The "mode" of normalisation indicates different ways to normalise the intensities, including:
    1) "meanstd": normalise to 0 mean 1 std;
    2) "minmax": normalise to specified (min, max) range;
    3) "fixed": normalise with a fixed ratio

    Args:
        x: (ndarray / Tensor, shape (N, *size))
        mode: (str) indicate normalisation mode
        min_in: (float) minimum value of the input (assumed value for fixed mode)
        max_in: (float) maximum value of the input (assumed value for fixed mode)
        min_out: (float) minimum value of the output
        max_out: (float) maximum value of the output
        clip: (boolean) value clipping if True
        clip_range_percentile: (tuple of floats) percentiles (min, max) to determine the thresholds for clipping

    Returns:
        x: (same as input) in-place op on input x
    """

    # determine data dimension
    dim = x.ndim - 1
    image_axes = tuple(range(1, 1 + dim))  # (1,2) for 2D; (1,2,3) for 3D

    # for numpy.ndarray
    if type(x) is np.ndarray:
        # Clipping
        if clip:
            # intensity clipping
            clip_min, clip_max = np.percentile(x, clip_range_percentile, axis=image_axes, keepdims=True)
            x = np.clip(x, clip_min, clip_max)

        # Normalise meanstd
        if mode == "meanstd":
            mean = np.mean(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            std = np.std(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        # Normalise minmax
        elif mode == "minmax":
            min_in = np.amin(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            max_in = np.amax(x, axis=image_axes, keepdims=True)  # (N, *range(dim)))
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12) + min_out # (!) multiple broadcasting)

        # Fixed ratio
        elif mode == "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError("Intensity normalisation mode not understood."
                             "Expect either one of: 'meanstd', 'minmax', 'fixed'")

        # cast to float 32
        x = x.astype(np.float32)

    # for torch.Tensor
    elif type(x) is torch.Tensor:
        # todo: clipping not supported at the moment (requires Pytorch version of the np.percentile()

        # Normalise meanstd
        if mode == "meanstd":
            mean = torch.mean(x, dim=image_axes, keepdim=True)  # (N, *range(dim))
            std = torch.std(x, dim=image_axes, keepdim=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        # Normalise minmax
        elif mode == "minmax":
            # get min/max across dims by flattening first
            min_in = x.flatten(start_dim=1, end_dim=-1).min(dim=1)[0].view(-1, *(1,)*dim)  # (N, (1,)*dim)
            max_in = x.flatten(start_dim=1, end_dim=-1).max(dim=1)[0].view(-1, *(1,)*dim)  # (N, (1,)*dim)
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12) + min_out  # (!) multiple broadcasting)

        # Fixed ratio
        elif mode == "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError("Intensity normalisation mode not recognised."
                             "Expect: 'meanstd', 'minmax', 'fixed'")

        # cast to float32
        x = x.float()

    else:
        raise TypeError("Input data type not recognised, support numpy.ndarray or torch.Tensor")
    return x




def crop_and_pad(x, new_size=192, mode="constant", **kwargs):
    """
    Crop and/or pad input to new size.
    (Adapted from DLTK: https://github.com/DLTK/DLTK/blob/master/dltk/io/preprocessing.py)

    Args:
        x: (np.ndarray) input array, shape (N, H, W) or (N, H, W, D)
        new_size: (int or tuple/list) new size excluding the batch size
        mode: (string) padding value filling mode for numpy.pad() (compulsory in Numpy v1.18)
        kwargs: additional arguments to be passed to np.pad

    Returns:
        (np.ndarray) cropped and/or padded input array
    """
    assert isinstance(x, (np.ndarray, np.generic))
    new_size = param_ndim_setup(new_size, ndim=x.ndim - 1)

    dim = x.ndim - 1
    sizes = x.shape[1:]

    # Initialise padding and slicers
    to_padding = [[0, 0] for i in range(x.ndim)]
    slicer = [slice(0, x.shape[i]) for i in range(x.ndim)]

    # For each dimensions except the dim 0, set crop slicers or paddings
    for i in range(dim):
        if sizes[i] < new_size[i]:
            to_padding[i+1][0] = (new_size[i] - sizes[i]) // 2
            to_padding[i+1][1] = new_size[i] - sizes[i] - to_padding[i+1][0]
        else:
            # Create slicer object to crop each dimension
            crop_start = int(np.floor((sizes[i] - new_size[i]) / 2.))
            crop_end = crop_start + new_size[i]
            slicer[i+1] = slice(crop_start, crop_end)

    return np.pad(x[tuple(slicer)], to_padding, mode=mode, **kwargs)

def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.

    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension

    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param

def visualise_results(fixed_image, moving_image, transformed_image, def_coords, title="Validation intensities", show=True, cmap='gray'):

    image_size = fixed_image.shape
    field = def_coords.reshape(*image_size, 3).cpu() * image_size[0]/2
    field = np.transpose(field)
    field = field.permute(0, 3, 2, 1)

    fig, ax = plt.subplots(3, 6, figsize=(15, 15))

    slices = None
    if slices is None:
        slices = [int(image_size[0] / 2), int(image_size[1] / 2), int(image_size[2] / 2)]

    ne_disps = []
    for a in range(0, 3):
        axes = [0, 1, 2]
        axes.remove(a)
        # z = 64
        z = slices[a]

        fixedAx = torch.index_select(fixed_image.cpu().clone().reshape(image_size), dim=a, index=torch.tensor([z])).squeeze().numpy()
        movingAx = torch.index_select(moving_image.clone().cpu().reshape(image_size), dim=a, index=torch.tensor([z])).squeeze().numpy()
        warpedAx = torch.index_select(transformed_image.detach().cpu().clone().reshape(image_size), dim=a, index=torch.tensor([z])).squeeze().numpy()

        s1 = ax[a, 0].imshow(fixedAx, cmap=cmap)
        plt.colorbar(s1, ax=ax[a, 0], fraction=0.045)
        ax[a, 0].set_title('Fixed')
        s2 = ax[a, 1].imshow(movingAx, cmap=cmap)
        ax[a, 1].set_title('Moving')
        plt.colorbar(s2, ax=ax[a, 1], fraction=0.045)
        s3 = ax[a, 2].imshow(warpedAx, cmap=cmap)
        ax[a, 2].set_title('Warped')
        plt.colorbar(s3, ax=ax[a, 2], fraction=0.045)
        s4 = ax[a, 3].imshow(movingAx - fixedAx, cmap='seismic')
        plt.colorbar(s4, ax=ax[a, 3], fraction=0.045)
        ax[a, 3].set_title('Error before')
        s4.set_clim(-1.0, 1.0)
        # plt.show()
        s5 = ax[a, 4].imshow(warpedAx - fixedAx, cmap='seismic')
        plt.colorbar(s5, ax=ax[a, 4], fraction=0.045)
        ax[a, 4].set_title('Error after')
        s5.set_clim(-1.0, 1.0)
        fieldAx = torch.index_select(field[axes, ...], dim=a+1, index=torch.tensor([z])).squeeze().numpy()
        plot_warped_grid(ax[a, 5], fieldAx, None, interval=5, title=f"axis {a}", fontsize=20)

        ne_disps.append(torch.tensor(fieldAx).permute(1, 2, 0).numpy()[::4, ::4]/4)

    if show:
        plt.show()

    plt.close('all')


def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="$\mathcal{T}_\phi$", fontsize=30, color='c'):
    """disp shape (2, H, W)"""
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": 1.5, "color": color}
    # matplotlib.plot() uses CV x-y indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output

def fast_nearest_interpolation(input_array, x_indices, y_indices, z_indices):

    # voxel displacement in image coordinate system
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    # get neighboring voxels in all three dimensions
    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # border handling
    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    # new position relative to neighboring voxels
    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    # nearest neighbor interpolation
    xN = torch.where(x > 0.5, x1, x0)
    yN = torch.where(y > 0.5, y1, y0)
    zN = torch.where(z > 0.5, z1, z0)

    output = input_array[xN, yN, zN]
    return output



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coordinate_slice(dims=(28, 28), dimension=0, slice_pos=0, gpu=True):
    """Make a coordinate tensor."""

    dims = list(dims)
    dims.insert(dimension, 1)

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor[dimension] = torch.linspace(slice_pos, slice_pos, 1)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_coordinate_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_masked_coordinate_tensor(mask, dims=(28, 28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor



def set_kernel(stride):
    kernels = list()
    for s in stride:
        # 1d cubic b-spline kernels
        kernels += [cubic_bspline1d(s)]
    return kernels



def cubic_bspline1d(stride, derivative: int = 0, dtype=None, device= None) -> torch.Tensor:
    r"""Cubic B-spline kernel for specified control point spacing.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    if dtype is None:
        dtype = torch.float
    if not isinstance(stride, int):
        (stride,) = stride
    kernel = torch.ones(4 * stride - 1, dtype=dtype)
    radius = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        kernel[i] = cubic_bspline_value((i - radius) / stride, derivative=derivative)
    if device is None:
        device = kernel.device
    return kernel.to(device)


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    r"""Evaluate 1-dimensional cubic B-spline."""
    t = abs(x)
    # outside local support region
    if t >= 2:
        return 0
    # 0-th order derivative
    if derivative == 0:
        if t < 1:
            return 2 / 3 + (0.5 * t - 1) * t ** 2
        return -((t - 2) ** 3) / 6
    # 1st order derivative
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2) ** 2
        return -0.5 * (t - 2) ** 2
    # 2nd oder derivative
    if derivative == 2:
        if t < 1:
            return 3 * t - 2
        return -t + 2


