# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch
import paddle
from paddle.optimizer import lr
from paddle.optimizer.lr import LRScheduler


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupMultiStepLRPaddle(object):
    """
    Piecewise learning rate decay
    Args:
        step_each_epoch(int): steps each epoch
        learning_rate (float): The initial learning rate. It is a python float number.
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 milestones,
                 step_each_epoch,
                 epochs,
                 gamma=0.1,
                 warmup_iters=0,
                 warmup_factor=1.0/3,
                 last_epoch=-1,
                 **kwargs):
        super().__init__()
        # if warmup_epoch >= epochs:
        #     msg = f"When using warm up, the value of \"Global.epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
        #     logger.warning(msg)
        #     warmup_epoch = epochs
        self.step_size = step_each_epoch * step_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_steps = warmup_iters
        self.warmup_factor = warmup_factor
        self.milestones = milestones

    def __call__(self):
        lrs = lr.MultiStepDecay(
            learning_rate=self.learning_rate,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_epoch)
        if self.warmup_steps > 0:
            lrs = lr.LinearWarmup(
                learning_rate=lrs,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_factor * self.learning_rate,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return lrs
