"""Learning-rate schedule of the benchmark trainers: linear warmup, then half-cosine decay to ``min_lr``."""
import math


def scheduled_lr(epoch, args):
    """Learning rate at a (possibly fractional) epoch."""
    if epoch < args.warmup_epochs:
        return args.lr * epoch / args.warmup_epochs
    if args.fixed_lr:
        return args.lr
    progress = (epoch - args.warmup_epochs) / (args.total_epochs - args.warmup_epochs)
    return args.min_lr + (args.blr - args.min_lr) * (1 + math.cos(math.pi * progress)) / 2


def adjust_learning_rate(optimizer, epoch, args):
    """Set each parameter group's learning rate for ``epoch`` (scaled by its ``lr_scale``, if any); return it."""
    lr = scheduled_lr(epoch, args)
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)
    return lr
