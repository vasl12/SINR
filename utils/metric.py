import numpy as np
import torch
import deepali
from utils.image import bbox_from_mask, bbox_crop


def measure_disp_metrics(metric_data):
    """
    Calculate DVF-related metrics.
    If roi_mask is given, the disp is masked and only evaluate in the bounding box of the mask.

    Args:
        metric_data: (dict)

    Returns:
        metric_results: (dict)
    """
    # new object to avoid changing data in metric_data
    disp_pred = metric_data['disp_pred']
    if 'disp_gt' in metric_data.keys():
        disp_gt = metric_data['disp_gt']

    # mask the disp with roi mask if given
    if 'roi_mask' in metric_data.keys():
        roi_mask = metric_data['roi_mask']  # (N, 1, *(sizes))

        # find roi mask bbox mask
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])

        # mask and bbox crop dvf gt and pred by roi_mask
        disp_pred = disp_pred * roi_mask
        disp_pred = bbox_crop(disp_pred, mask_bbox)

        if 'disp_gt' in metric_data.keys():
            disp_gt = disp_gt * roi_mask
            disp_gt = bbox_crop(disp_gt, mask_bbox)

    # Regularity (Jacobian) metrics
    folding_ratio, mag_det_jac_det = calculate_jacobian_metrics(disp_pred)

    disp_metric_results = dict()
    disp_metric_results.update({'folding_ratio': folding_ratio,
                               'mag_det_jac_det': mag_det_jac_det})

    # DVF accuracy metrics if ground truth is available
    if 'disp_gt' in metric_data.keys():
        disp_metric_results.update({'aee': calculate_aee(disp_pred, disp_gt),
                                   'rmse_disp': calculate_rmse_disp(disp_pred, disp_gt)})
    return disp_metric_results



def measure_seg_metrics(metric_data):
    """ Calculate segmentation """

    if isinstance(metric_data['fixed_seg'], np.ndarray):
        seg_gt = torch.tensor(metric_data['fixed_seg']).unsqueeze(0).unsqueeze(0)
        seg_pred = torch.tensor(metric_data['warped_seg']).unsqueeze(0).unsqueeze(0)
    else:
        seg_gt = metric_data['fixed_seg'].unsqueeze(0).unsqueeze(0)
        seg_pred = metric_data['warped_seg'].unsqueeze(0).unsqueeze(0)

    assert seg_gt.ndim == seg_pred.ndim

    results = dict()
    for label_cls in torch.unique(seg_gt):
        # calculate DICE score for each class
        if label_cls == 0:
            # skip background
            continue
        results[f'dice_class_{label_cls}'] = calculate_dice(seg_gt, seg_pred, label_class=label_cls)

    # calculate mean dice
    results['mean_dice'] = np.mean([dice.cpu().numpy() for k, dice in results.items()])
    return results

def calculate_dice(mask1, mask2, label_class=0):
    """
    Dice score of a specified class between two label masks.
    (classes are encoded but by label class number not one-hot )

    Args:
        mask1: (numpy.array, shape (N, 1, *sizes)) segmentation mask 1
        mask2: (numpy.array, shape (N, 1, *sizes)) segmentation mask 2
        label_class: (int or float)

    Returns:
        volume_dice
    """
    mask1_pos = (mask1 == label_class).type(torch.cuda.FloatTensor)
    mask2_pos = (mask2 == label_class).type(torch.cuda.FloatTensor)

    assert mask1.ndim == mask2.ndim
    axes = tuple(range(2, mask1.ndim))
    pos1and2 = torch.sum(mask1_pos * mask2_pos, axis=axes)
    pos1 = torch.sum(mask1_pos, axis=axes)
    pos2 = torch.sum(mask2_pos, axis=axes)
    return torch.mean(2 * pos1and2 / (pos1 + pos2 + 1e-7)).cpu()

"""
Functions calculating individual metrics
"""


def calculate_aee(x, y):
    """
    Average End point Error (AEE, mean over point-wise L2 norm)
    Input DVF shape: (N, dim, *(sizes))
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1)).mean()


def calculate_rmse_disp(x, y):
    """
    RMSE of DVF (square root over mean of sum squared)
    Input DVF shape: (N, dim, *(sizes))
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1).mean())



def calculate_jacobian_metrics(disp):
    """
    Calculate Jacobian related regularity metrics.

    Args:
        disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field

    Returns:
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
    """


    folding_ratio = []
    mag_grad_jac_det = []
    # for n in range(disp.shape[0]):
    #     disp_n = np.moveaxis(disp[n, ...], 0, -1)  # (*sizes, ndim)
        # jac_det_n = calculate_jacobian_det(disp_n)
    if not isinstance(disp, torch.Tensor):
        disp = torch.tensor(disp)
    jac_det_n = deepali.core.flow.jacobian_det(disp)
    folding_ratio += [(jac_det_n < 0).sum() / np.prod(jac_det_n.shape)]
    mag_grad_jac_det += [np.abs(np.gradient(jac_det_n[0][0])).mean()]
    return np.mean(folding_ratio), np.mean(mag_grad_jac_det)





