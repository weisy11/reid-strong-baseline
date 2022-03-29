# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch
import paddle

from lr_scheduler import WarmupMultiStepLRPaddle


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer


def make_optimizer_paddle(cfg, model, len_dataloader):
    base_lr = cfg.SOLVER.BASE_LR
    lr = WarmupMultiStepLRPaddle(base_lr, cfg.SOLVER.STEPS, len_dataloader, cfg.SOLVER.MAX_EPOCHS,
                                 cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_FACTOR)
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        momentum = cfg.SOLVER.MOMENTUM
        optimizer = getattr(paddle.optimizer, "Momentum")(lr, momentum, model.parameters(), weight_decay=weight_decay)
    else:
        optimizer = getattr(paddle.optimizer, cfg.SOLVER.OPTIMIZER_NAME)(lr, parameters=model.parameters(),
                                                                         weight_decay=weight_decay)
    return optimizer


def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center


def make_optimizer_with_center_paddle(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"parameters": [value], "learning_rate": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(paddle.optimizer, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(paddle.optimizer, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = paddle.optimizer.SGD(learning_rate=cfg.SOLVER.CENTER_LR,
                                            parameters=center_criterion.parameters())
    return optimizer, optimizer_center
