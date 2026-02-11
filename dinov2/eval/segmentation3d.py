# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from dinov2.data.loaders import make_segmentation_dataset_3d
from dinov2.data import SamplerType, make_data_loader
from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead, LinearDecoderHead, ViTAdapterUNETRHead
from dinov2.eval.setup import get_args_parser, setup_and_build_model_3d
from dinov2.eval.segmentation_3d.augmentations import make_transforms
from dinov2.eval.segmentation_3d.metrics import get_metric

import torch
import json
import numpy as np
import os
from functools import partial
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.data.utils import list_data_collate
from monai.optimizers import WarmupCosineSchedule
from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead, LinearDecoderHead, ViTAdapterUNETRHead, ClassificationHead


def add_seg_args(parser):
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of finetuning dataset",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="segmentation",
        choices=["segmentation", "classification"],
        help="Task type: segmentation or classification",
    )
    parser.add_argument(
        "--datalist-path",
        type=str,
        help="Path to the datalist JSON file",
    )
    parser.add_argument(
        "--dataset-percent",
        type=int,
        help="Percent of finetuning dataset to use",
        default=100
    )
    parser.add_argument(
        "--base-data-dir",
        type=str,
        help="Base data directory for finetuning dataset",
    )
    parser.add_argument(
        "--segmentation-head",
        type=str,
        help="Segmentation head",
    )
    parser.add_argument(
        "--train-feature-model",
        action="store_true",
        help="Freeze feature model or not",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Total epochs",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        default=24,
        help="Iterations to perform per epoch",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=24,
        help="Iterations to perform per evaluation",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image side length",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=1.0,
        help="Scale factor for resizing images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="path to cache directory for monai persistent dataset"
    )

    return parser


def train_iter(model, batch, optimizer, scheduler, loss_function, scaler, task_type="segmentation"):
    x = batch["image"].cuda()
    if task_type == "segmentation":
        y = batch["label"].cuda()
    else:
        y = batch["class"].cuda()
        
    logits = model(x)
    loss = loss_function(logits, y)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item()


def val_iter(model, batch, metric, image_size, batch_size, overlap=0.5, task_type="segmentation"):
    x = batch["image"].cuda()
    if task_type == "segmentation":
        y = batch["label"].cuda()
        logits = sliding_window_inference(x, image_size, batch_size, model, overlap=overlap)
        iter_metric = metric(logits, y)
    else:
        y = batch["class"].cuda()
        logits = model(x)
        # For classification, we might just return the cross entropy loss or accuracy
        # But let's assume metric is a simple accuracy or similar
        iter_metric = metric(logits, y)
        
    return iter_metric

def aggregate_metrics(metrics_list):
    """Aggregate a list of metric dicts from multiple steps/batches."""
    if not metrics_list:
        return {}
    
    # Check if first element is a dict (segmentation) or a scalar (classification)
    if isinstance(metrics_list[0], dict):
        agg = {
            "avg_dice": [],
            "class_dice": [],
            "avg_hd95": [],
            "class_hd95": [],
            "lesion_data": [],
            "fp_counts": []
        }
        for m in metrics_list:
            agg["avg_dice"].append(m["avg_dice"])
            agg["class_dice"].append(m["class_dice"])
            agg["avg_hd95"].append(m["avg_hd95"])
            agg["class_hd95"].append(m["class_hd95"])
            # lesion_data is list(cases) -> list(classes) -> list(metrics)
            agg["lesion_data"].extend(m["lesion_data"])
            agg["fp_counts"].extend(m["fp_counts"])
            
        # Aggregate Dice/HD95
        class_dices = np.array(agg["class_dice"])
        mean_class_dice = np.nanmean(class_dices, axis=0).tolist()
        
        class_hd95s = np.array(agg["class_hd95"])
        # Treat sentinel large values (used to replace inf/nan) as missing
        # and exclude GT-all-zero classes from HD95 averages.
        class_hd95s = np.where(np.isfinite(class_hd95s) & (class_hd95s < 374.0), class_hd95s, np.nan)
        mean_class_hd95 = np.nanmean(class_hd95s, axis=0).tolist()

        # Aggregate lesion-wise (Senior Script Logic)
        num_classes = len(mean_class_dice)
        avg_lesion_dice_cls = []
        avg_lesion_hd95_cls = []
        
        # BraTS threshold (usually 50 for PED/SSA)
        vol_thresh = 50 
        
        for c in range(num_classes):
            total_dice = 0.0
            total_hd95 = 0.0
            total_gt_lesions = 0
            total_fp = 0
            
            for case_idx in range(len(agg["lesion_data"])):
                l_metrics = agg["lesion_data"][case_idx][c] # list of (vol, dice, hd95)
                l_fp = agg["fp_counts"][case_idx][c]
                total_fp += l_fp
                for vol, d, h in l_metrics:
                    if vol > vol_thresh:
                        total_dice += d
                        h_val = 374.0 if (np.isinf(h) or np.isnan(h)) else h
                        total_hd95 += h_val
                        total_gt_lesions += 1
                total_hd95 += l_fp * 374.0
            
            denom = total_gt_lesions + total_fp
            if denom > 0:
                l_dice_score = total_dice / denom
                l_hd95_score = total_hd95 / denom
            else:
                l_dice_score = 1.0 # Logic: if no lesions and no FPs, score is perfectly clean
                # For HD95, mark as NaN so it will be skipped in averages
                l_hd95_score = np.nan
                
            avg_lesion_dice_cls.append(l_dice_score)
            avg_lesion_hd95_cls.append(l_hd95_score)

        # Convert batch-level avg HD95 sentinel values to NaN before averaging
        avg_hd95_arr = np.array(agg["avg_hd95"])
        avg_hd95_arr = np.where(np.isfinite(avg_hd95_arr) & (avg_hd95_arr < 374.0), avg_hd95_arr, np.nan)

        return {
            "dice": np.nanmean(agg["avg_dice"]),
            "class_dice": mean_class_dice,
            "hd95": np.nanmean(avg_hd95_arr),
            "class_hd95": mean_class_hd95,
            "lesion_dice": np.mean(avg_lesion_dice_cls),
            "lesion_hd95": np.nanmean(avg_lesion_hd95_cls),
            "lesion_dice_cls": avg_lesion_dice_cls,
            "lesion_hd95_cls": avg_lesion_hd95_cls
        }
    else:
        # Classification aggregation
        return {"accuracy": np.mean(metrics_list)}

def run_eval(model, loader, metric, image_size, batch_size, task_type="segmentation", overlap=0.0):
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for batch in loader:
            m = val_iter(model, batch, metric, image_size, batch_size, overlap=overlap, task_type=task_type)
            metrics_list.append(m)
    return aggregate_metrics(metrics_list)


def do_finetune(feature_model, autocast_dtype, args):
    # Quick check for input channels before building transforms
    datalist_file = args.datalist_path if args.datalist_path else args.base_data_dir
    tmp_channels = 4
    if args.dataset_name == 'BraTS' and datalist_file and os.path.exists(datalist_file):
        try:
            with open(datalist_file, 'r') as f:
                tmp_dl = json.load(f)
                training_data = tmp_dl.get('training', tmp_dl.get('data', []))
                if len(training_data) > 0:
                    img = training_data[0].get('image', [])
                    if isinstance(img, list):
                        tmp_channels = len(img)
                    else:
                        tmp_channels = 1
                    print(f"Pre-detected {tmp_channels} channels for BraTS")
        except Exception as e:
            print(f"Warning: Could not pre-detect channels: {e}")

    # get transforms, dataset, dataloaders
    train_transforms, val_transforms = make_transforms(
        args.dataset_name,
        args.image_size,
        args.resize_scale,
        min_int=-1.0,
        task_type=args.task_type,
        input_channels=tmp_channels
    )
    
    # Use user-specified datalist path if provided
    datalist_file = args.datalist_path if args.datalist_path else args.base_data_dir
    
    if args.task_type == 'segmentation':
        train_ds, val_ds, test_ds, input_channels, num_classes = make_segmentation_dataset_3d(
            args.dataset_name,
            args.dataset_percent,
            args.base_data_dir,
            train_transforms,
            val_transforms,
            args.cache_dir,
            args.batch_size,
            datalist_filename=datalist_file if args.datalist_path else None
        )
    else:
        # Classification
        from dinov2.data.loaders import make_classification_dataset_3d
        train_ds, val_ds, test_ds, num_classes = make_classification_dataset_3d(
            args.dataset_name,
            args.dataset_percent,
            datalist_file,
            train_transforms,
            val_transforms,
            args.cache_dir,
            dataset_seed=0
        )
        input_channels = 4 # Default for BraTS modals
    
    train_loader = make_data_loader(
        dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=True,
        collate_fn=list_data_collate
    )
    # Val loader for metrics (no shuffling)
    metric_train_loader = make_data_loader(
        dataset=train_ds,
        batch_size=1, # Eval one by one
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=list_data_collate
    )
    val_loader = make_data_loader(
        dataset=val_ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=list_data_collate
    )
    test_loader = make_data_loader(
        dataset=test_ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=list_data_collate
    )

    # get model
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    scaler = torch.cuda.amp.GradScaler()
    
    if args.task_type == "segmentation":
        if args.segmentation_head == 'UNETR':
            model = UNETRHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
        elif args.segmentation_head == 'Linear':
            model = LinearDecoderHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
        elif args.segmentation_head == 'ViTAdapterUNETR':
            model = ViTAdapterUNETRHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)
        else:
            raise ValueError(f"Unknown segmentation head: {args.segmentation_head}")
        loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    else:
        # Classification
        model = ClassificationHead(feature_model, input_channels, num_classes, autocast_ctx)
        loss_fn = torch.nn.CrossEntropyLoss()

    if args.train_feature_model:
        if hasattr(model, "feature_model") and hasattr(model.feature_model, "vit_model"):
            model.feature_model.vit_model.train()
        else:
            model.feature_model.train()
    else:
        if hasattr(model, "feature_model") and hasattr(model.feature_model, "vit_model"):
            model.feature_model.vit_model.eval()
            for param in model.feature_model.vit_model.parameters():
                param.requires_grad = False
        else:
            model.feature_model.eval()
            for param in model.feature_model.parameters():
                param.requires_grad = False

    model.cuda()
    loss_fn.cuda()

    # Dynamic epoch length based on actual loader size
    args.epoch_length = len(train_loader)
    print(f"Setting epoch_length to {args.epoch_length}")

    # get optimizer, scheduler, metric
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=args.learning_rate)
    max_iter = args.epochs * args.epoch_length
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=args.warmup_iters,
        t_total=max_iter
    )

    dice_metric = get_metric(args.dataset_name, task_type=args.task_type)

    best_val_score = -1
    train_loss_sum = 0
    
    results = {
        'epoch_logs': []
    }

    import time
    
    it = 0
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        if not args.train_feature_model:
            if hasattr(model, "feature_model") and hasattr(model.feature_model, "vit_model"):
                model.feature_model.vit_model.eval()
            else:
                model.feature_model.eval()

        for iter_in_epoch, train_data in enumerate(train_loader):
            start_time = time.time()
            # train for one iteration
            train_loss = train_iter(
                model=model,
                batch=train_data,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_function=loss_fn,
                scaler=scaler,
                task_type=args.task_type
            )
            train_loss_sum += train_loss
            elapsed_time = time.time() - start_time
            it += 1

            print(f"Train epoch [{epoch}/{args.epochs}]({iter_in_epoch}/{len(train_loader)}):  loss: {train_loss:.4f} time {elapsed_time:.2f}s", flush=True)

        # End of epoch evaluation
        avg_train_loss = train_loss_sum / len(train_loader)
        train_loss_sum = 0
        
        # Validation
        model.eval()
        val_metrics_list = []
        with torch.no_grad():
            for v_it, val_data in enumerate(val_loader):
                v_start = time.time()
                m = val_iter(model, val_data, dice_metric, (args.image_size,) * 3, args.batch_size, overlap=0.0, task_type=args.task_type)
                val_metrics_list.append(m)
                v_elapsed = time.time() - v_start
                print(f"Valid epoch [{epoch}/{args.epochs}]({v_it}/{len(val_loader)}):  loss: 0.0000 time {v_elapsed:.2f}s", flush=True)

        val_results = aggregate_metrics(val_metrics_list)
        
        # Test Evaluation also
        test_metrics_list = []
        with torch.no_grad():
            for t_it, test_data in enumerate(test_loader):
                m = val_iter(model, test_data, dice_metric, (args.image_size,) * 3, args.batch_size, overlap=0.0, task_type=args.task_type)
                test_metrics_list.append(m)
        test_results = aggregate_metrics(test_metrics_list)

        # Print in the requested format
        if args.task_type == "segmentation":
            v_dice = val_results.get('dice', 0)
            v_class_dice = val_results.get('class_dice', [])
            v_hd95 = val_results.get('hd95', 0)
            v_l_dice = val_results.get('lesion_dice', 0)
            v_l_hd95 = val_results.get('lesion_hd95', 0)

            t_dice = test_results.get('dice', 0)
            t_class_dice = test_results.get('class_dice', [])
            t_hd95 = test_results.get('hd95', 0)

            cls_dice_str = ", ".join([f"{i}: {d:.5f}" for i, d in enumerate(v_class_dice)])
            print(f"validation {epoch} epoch:  AVG: {v_dice:.5f},  {cls_dice_str}", flush=True)
            print(f"validation {epoch} epoch summary: AVG Dice: {v_dice:.5f}, AVG HD95: {v_hd95:.5f}", flush=True)
            
            print(f"[Epoch {epoch:03d}] VALIDATION metrics -> class Dice {v_class_dice}, avg Dice {v_dice:.4f}, HD95 {v_hd95:.4f}, lesion Dice {v_l_dice:.4f}, lesion HD95 {v_l_hd95:.4f}", flush=True)
            print(f"[Epoch {epoch:03d}] TEST metrics -> class Dice {t_class_dice}, avg Dice {t_dice:.4f}, HD95 {t_hd95:.4f}", flush=True)
            print(f"[Epoch {epoch:03d}] SUMMARY -> val avg Dice {v_dice:.4f}, test avg Dice {t_dice:.4f}, val HD95 {v_hd95:.4f}, test HD95 {t_hd95:.4f}", flush=True)
        else:
            # Classification logging
            v_acc = val_results.get('accuracy', 0)
            t_acc = test_results.get('accuracy', 0)
            print(f"validation {epoch} epoch:  AVG Accuracy: {v_acc:.5f}", flush=True)
            print(f"[Epoch {epoch:03d}] SUMMARY -> validation avg Accuracy {v_acc:.4f}, test avg Accuracy {t_acc:.4f}", flush=True)

        score = val_results.get('dice', 0) if args.task_type == "segmentation" else val_results.get('accuracy', 0)
        
        if score > best_val_score:
            best_val_score = score
            print(f"Saving {epoch} epoch with best validation metrics {best_val_score}", flush=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        if it >= max_iter:
            break

    # Final test
    if os.path.exists(os.path.join(args.output_dir, "best_model.pth")):
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    test_results = run_eval(model, test_loader, dice_metric, (args.image_size,) * 3, args.batch_size, task_type=args.task_type, overlap=0.75)
    
    print("\n--- Final Test Results ---")
    print(f"Test Dice: {test_results.get('dice', 0):.4f}")
    if args.task_type == "segmentation":
        print(f"Test HD95: {test_results.get('hd95', 0):.4f}")
        print(f"Test Lesion Dice: {test_results.get('lesion_dice', 0):.4f}, HD95: {test_results.get('lesion_hd95', 0):.4f}")

    results['final_test'] = test_results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=4)

def main(args):
    from dinov2.eval.setup import setup_and_build_model_3d
    feature_model, autocast_dtype = setup_and_build_model_3d(args)
    do_finetune(feature_model, autocast_dtype, args)


if __name__ == "__main__":
    from dinov2.eval.setup import get_args_parser
    args = add_seg_args(get_args_parser(add_help=True)).parse_args()
    main(args)

