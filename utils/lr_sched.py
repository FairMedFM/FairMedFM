# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math


def adjust_learning_rate(optimizer, epoch, opt):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < opt["warmup_epochs"]:
        lr = opt["lr"] * epoch / opt["warmup_epochs"]
    else:
        if opt["fixed_lr"]:
            lr = opt["lr"]
        else:
            lr = opt["min_lr"] + (opt["blr"] - opt["min_lr"]) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - opt["warmup_epochs"]) / (opt["total_epochs"] - opt["warmup_epochs"]))
            )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
