# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn


logger = logging.getLogger("dinov2")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if not pretrained_weights:
        logger.info("No pretrained weights provided, skipping loading.")
        return
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    # 3D Position embedding interpolation if size mismatch
    if "pos_embed" in state_dict and "pos_embed" in model.state_dict():
        ckpt_pos_embed = state_dict["pos_embed"]
        model_pos_embed = model.state_dict()["pos_embed"]
        if ckpt_pos_embed.shape != model_pos_embed.shape:
            logger.info(f"Interpolating pos_embed from {ckpt_pos_embed.shape} to {model_pos_embed.shape}")
            cls_token = ckpt_pos_embed[:, :1, :]
            patch_pos_embed = ckpt_pos_embed[:, 1:, :]

            target_num_patches = model_pos_embed.shape[1] - 1
            ckpt_num_patches = patch_pos_embed.shape[1]

            # Assume cube structure (cube root of patches)
            ckpt_side = int(round(ckpt_num_patches ** (1/3)))
            target_side = int(round(target_num_patches ** (1/3)))
            embedding_dim = ckpt_pos_embed.shape[-1]

            patch_pos_embed = patch_pos_embed.reshape(1, ckpt_side, ckpt_side, ckpt_side, embedding_dim).permute(0, 4, 1, 2, 3)
            patch_pos_embed = torch.nn.functional.interpolate(
                patch_pos_embed.float(),
                size=(target_side, target_side, target_side),
                mode='trilinear',
                align_corners=False
            ).to(patch_pos_embed.dtype)
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).reshape(1, -1, embedding_dim)

            state_dict["pos_embed"] = torch.cat((cls_token, patch_pos_embed), dim=1)

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
