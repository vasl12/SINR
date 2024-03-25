import os
import omegaconf


def setup_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


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
