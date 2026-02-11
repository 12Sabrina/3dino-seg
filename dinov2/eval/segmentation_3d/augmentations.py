# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    MapTransform,
    EnsureTyped,
    RandSpatialCropSamplesd,
    RandScaleIntensityd,
    ConcatItemsd,
    DeleteItemsd,
    SpatialPadd,
    Lambdad
)
import torch
import numpy as np


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the necrotic and non-enhancing tumor core (NCR)
    label 2 is the peritumoral edema (ED)
    label 3 is GD-enhancing tumor (ET)
    label 4 is the resection cavity (RC)

    The possible classes are TC (Tumor core), WT (Whole tumor), ET (Enhancing tumor), and RC.
    """

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            result = []
            # TC: NCR + ET (1 + 3/4)
            result.append(torch.logical_or(d[key] == 1, torch.logical_or(d[key] == 3, d[key] == 4)))
            # WT: NCR + ED + ET (1 + 2 + 3/4)
            result.append(torch.logical_or(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1), d[key] == 4))
            # ET: 3 or 4
            result.append(torch.logical_or(d[key] == 3, d[key] == 4))
            
            d[key] = torch.cat(result, dim=0).float()
        return d



def make_transforms(dataset_name, image_size, resize_scale, min_int, task_type="segmentation", input_channels=4):

    if dataset_name == 'BraTS':
        keys = [f"image{i+1}" for i in range(input_channels)]
        if task_type == "segmentation":
            keys.append("label")

        import monai.transforms as mt
        if input_channels > 1:
             # Repacking the list for BraTS with multiple channels
             base_train = [
                LoadImaged(keys=keys, ensure_channel_first=True, allow_missing_keys=True),
                DeleteItemsd(keys=["image"]), # Delete the original path list to avoid name conflict
                ConcatItemsd(keys=[f"image{i+1}" for i in range(input_channels)], name='image', dim=0),
                DeleteItemsd(keys=[f"image{i+1}" for i in range(input_channels)]),
                EnsureTyped(keys=["image"] + (["label"] if task_type == "segmentation" else [])),
            ]
        else:
             base_train = [
                LoadImaged(keys=keys, ensure_channel_first=True, allow_missing_keys=True),
                DeleteItemsd(keys=["image"]), # Delete the original path list to avoid name conflict
                mt.CopyItemsd(keys=["image1"], names=["image"]),
                DeleteItemsd(keys=["image1"]),
                EnsureTyped(keys=["image"] + (["label"] if task_type == "segmentation" else [])),
            ]
        
        if task_type == "segmentation":
            base_train.append(ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]))
            
        base_train.extend([
            Orientationd(keys=["image"] + (["label"] if task_type == "segmentation" else []), axcodes="RAS"),
            Spacingd(
                keys=["image"] + (["label"] if task_type == "segmentation" else []),
                pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                mode=("bilinear", "nearest" if task_type == "segmentation" else None),
                allow_missing_keys=True
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
            ),
            SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
        ])
        
        if task_type == "segmentation":
             base_train.append(SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.))

        # Add RandCrop/SpatialCrop
        if task_type == "segmentation":
            base_train.append(RandSpatialCropSamplesd(
                keys=["image", "label"], num_samples=4, roi_size=(image_size, image_size, image_size), random_size=False
            ))
        else:
            base_train.append(RandSpatialCropSamplesd(
                keys=["image"], num_samples=1, roi_size=(image_size, image_size, image_size), random_size=False
            ))

        train_tail = [
            RandFlipd(keys=["image"] + (["label"] if task_type == "segmentation" else []), prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=["image"] + (["label"] if task_type == "segmentation" else []), prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandFlipd(keys=["image"] + (["label"] if task_type == "segmentation" else []), prob=0.5, spatial_axis=2, allow_missing_keys=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
        
        train_transforms = Compose(base_train + train_tail)
        
        # Val transforms
        if input_channels > 1:
            base_val = [
                LoadImaged(keys=keys, ensure_channel_first=True, allow_missing_keys=True),
                DeleteItemsd(keys=["image"]), # Delete the original path list to avoid name conflict
                ConcatItemsd(keys=[f"image{i+1}" for i in range(input_channels)], name='image', dim=0),
                DeleteItemsd(keys=[f"image{i+1}" for i in range(input_channels)]),
                EnsureTyped(keys=["image"] + (["label"] if task_type == "segmentation" else [])),
            ]
        else:
            base_val = [
                LoadImaged(keys=keys, ensure_channel_first=True, allow_missing_keys=True),
                DeleteItemsd(keys=["image"]), # Delete the original path list to avoid name conflict
                mt.CopyItemsd(keys=["image1"], names=["image"]),
                DeleteItemsd(keys=["image1"]),
                EnsureTyped(keys=["image"] + (["label"] if task_type == "segmentation" else [])),
            ]
            
        if task_type == "segmentation":
            base_val.append(ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]))
            
        base_val.extend([
            Orientationd(keys=["image"] + (["label"] if task_type == "segmentation" else []), axcodes="RAS"),
            Spacingd(
                keys=["image"] + (["label"] if task_type == "segmentation" else []),
                pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                mode=("bilinear", "nearest" if task_type == "segmentation" else None),
                allow_missing_keys=True
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
            ),
            SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
        ])
        
        if task_type == "segmentation":
            base_val.append(SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.))
            
        val_transforms = Compose(base_val)
        
        return train_transforms, val_transforms

    if dataset_name == 'BTCV':


        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["label"], func=lambda x: (x == 255).astype(np.uint8)),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 0.5 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    elif dataset_name == 'TDSC-ABUS':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(image_size, image_size, image_size),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=min_int,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0 / resize_scale, 1.0 / resize_scale, 1.0 / resize_scale),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                SpatialPadd(keys=["label"], spatial_size=(image_size, image_size, image_size), value=0.),
            ]
        )

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Flatten transforms to allow caching of non-random transforms if needed
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()

    return train_transforms, val_transforms
