# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose, Activations
from monai.data import decollate_batch
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch


def compute_surface_voxels(binary_mask: np.ndarray):
    if binary_mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.int32)
    eroded = ndimage.binary_erosion(binary_mask)
    surface = binary_mask ^ eroded
    coords = np.array(np.nonzero(surface)).T  # z,y,x order
    return coords


def compute_hd95(gt_mask: np.ndarray, pred_mask: np.ndarray):
    if gt_mask.sum() == 0 and pred_mask.sum() == 0:
        return 0.0
    if gt_mask.sum() == 0 or pred_mask.sum() == 0:
        return float('nan')

    gt_surf = compute_surface_voxels(gt_mask)
    pred_surf = compute_surface_voxels(pred_mask)
    if gt_surf.shape[0] == 0 or pred_surf.shape[0] == 0:
        return float('nan')

    tree_pred = cKDTree(pred_surf)
    dists_gt_to_pred, _ = tree_pred.query(gt_surf, k=1)
    tree_gt = cKDTree(gt_surf)
    dists_pred_to_gt, _ = tree_gt.query(pred_surf, k=1)

    hd95 = max(np.percentile(dists_gt_to_pred, 95), np.percentile(dists_pred_to_gt, 95))
    return float(hd95)


def dice(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    intersection = np.logical_and(im1, im2)
    return 2. * (intersection.sum()) / (im1.sum() + im2.sum() + 1e-8)


def lesion_wise_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray, dil_factor=3):
    """
    Implements lesion-wise logic exactly as specified in the senior's script.
    """
    # connectivity=26 structure
    struct = ndimage.generate_binary_structure(3, 3) 
    dilation_struct = ndimage.generate_binary_structure(3, 2)

    gt_mat_cc, num_gt = ndimage.label(gt_mask, structure=struct)
    pred_mat_cc, num_pred = ndimage.label(pred_mask, structure=struct)

    # 1. Combine GT lesions by dilation
    gt_mat_dilation = ndimage.binary_dilation(gt_mask, structure=dilation_struct, iterations=dil_factor)
    gt_mat_dilation_cc, num_gt_dil = ndimage.label(gt_mat_dilation, structure=struct)
    
    gt_label_cc = np.zeros_like(gt_mat_cc)
    for comp in range(1, num_gt_dil + 1):
        gt_d_tmp = (gt_mat_dilation_cc == comp)
        # Components of original GT that fall into this dilated component
        gt_components = np.unique(gt_mat_cc[gt_d_tmp])
        gt_components = gt_components[gt_components != 0]
        for c in gt_components:
            gt_label_cc[gt_mat_cc == c] = comp
    
    num_merged_gt = num_gt_dil
    
    # 2. Match Prediction components to (dilated) GT components
    tp_pred_indices = set()
    metric_pairs = [] # List of (gt_vol, dice, hd95)

    for gtcomp in range(1, num_merged_gt + 1):
        gt_tmp = (gt_label_cc == gtcomp).astype(np.uint8)
        if gt_tmp.sum() == 0: continue
        
        gt_vol = gt_tmp.sum() # Assuming 1x1x1 mm3
        
        # Dilation of THIS specific merged GT lesion
        gt_tmp_dilation = ndimage.binary_dilation(gt_tmp, structure=dilation_struct, iterations=dil_factor)
        
        # Intersecting prediction components
        intersecting_cc = np.unique(pred_mat_cc[gt_tmp_dilation > 0])
        intersecting_cc = intersecting_cc[intersecting_cc != 0]
        
        for cc in intersecting_cc:
            tp_pred_indices.add(cc)
            
        # Isolate matching pred components
        # Note: the senior script calculates dice between (all matched pred CCs) and (this single GT CC)
        pred_matched = np.isin(pred_mat_cc, intersecting_cc).astype(np.uint8)
        
        d_score = dice(pred_matched, gt_tmp)
        h_score = compute_hd95(gt_tmp, pred_matched)
        
        metric_pairs.append((float(gt_vol), float(d_score), float(h_score)))

    # 3. Identify FP components
    # FP are pred components that didn't touch any dilated GT component
    fp_indices = [i for i in range(1, num_pred + 1) if i not in tp_pred_indices]
    num_fp = len(fp_indices)
    
    return metric_pairs, num_fp


class BTCVMetrics:

    def __init__(self):
        self.post_label = AsDiscrete(to_onehot=14)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=14)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    def __call__(self, pred, target):
        target_list = decollate_batch(target)
        target_list = [self.post_label(t) for t in target_list]
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        self.dice_metric(y_pred=pred_list, y=target_list)
        self.dice_metric_batch(y_pred=pred_list, y=target_list)

        avg_dice = self.dice_metric.aggregate().item()
        class_dice = self.dice_metric_batch.aggregate()
        class_dice = [d.item() for d in class_dice]

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return avg_dice, class_dice


class BraTSMetrics:

    def __init__(self):
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
        self.hd95_metric_batch = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")

    def __call__(self, pred, target):
        pred_list = decollate_batch(pred)
        pred_list = [self.post_pred(p) for p in pred_list]

        # Standard Dice and HD95
        self.dice_metric(y_pred=pred_list, y=target)
        self.dice_metric_batch(y_pred=pred_list, y=target)
        self.hd95_metric(y_pred=pred_list, y=target)
        self.hd95_metric_batch(y_pred=pred_list, y=target)

        avg_dice = self.dice_metric.aggregate().item()
        class_dice = self.dice_metric_batch.aggregate()

        avg_hd95 = self.hd95_metric.aggregate().item()
        class_hd95 = self.hd95_metric_batch.aggregate()

        # Lesion-wise metrics
        all_lesion_data = [] # List of list of metrics per class
        all_fp_counts = []

        target_np = target.detach().cpu().numpy()
        for b in range(len(pred_list)):
            case_lesion_data = []
            case_fp_counts = []
            pred_np = pred_list[b].detach().cpu().numpy()
            for c in range(pred_np.shape[0]):
                l_metrics, l_fp = lesion_wise_metrics(target_np[b, c], pred_np[c])
                case_lesion_data.append(l_metrics)
                case_fp_counts.append(l_fp)
            all_lesion_data.append(case_lesion_data)
            all_fp_counts.append(case_fp_counts)

        self.dice_metric.reset()
        self.dice_metric_batch.reset()
        self.hd95_metric.reset()
        self.hd95_metric_batch.reset()

        return {
            "avg_dice": avg_dice,
            "class_dice": [d.item() for d in class_dice],
            "avg_hd95": avg_hd95,
            "class_hd95": [d.item() for d in class_hd95],
            "lesion_data": all_lesion_data, # [case][class] -> list of (vol, dice, hd95)
            "fp_counts": all_fp_counts      # [case][class] -> int
        }


class LASEGMetrics(BTCVMetrics):

    def __init__(self):
        super().__init__()
        self.post_label = AsDiscrete(to_onehot=2)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)


class ClassificationMetrics:
    def __init__(self):
        pass

    def __call__(self, pred, target):
        # pred: [B, num_classes], target: [B]
        # Calculate accuracy
        preds = torch.argmax(pred, dim=1)
        acc = (preds == target).float().mean()
        return acc.item()


def get_metric(dataset_name, task_type="segmentation"):
    if task_type == "classification":
        return ClassificationMetrics()
        
    if dataset_name == "BTCV":
        return BTCVMetrics()
    elif dataset_name == "BraTS":
        return BraTSMetrics()
    elif dataset_name == "LA-SEG":
        return LASEGMetrics()
    elif dataset_name == "TDSC-ABUS":
        return LASEGMetrics()  # same as LA-SEG
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
