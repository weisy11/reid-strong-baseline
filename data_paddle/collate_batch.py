# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import paddle


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = paddle.to_tensor(pids, dtype=paddle.int64)
    return paddle.stack(imgs, axis=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return paddle.stack(imgs, axis=0), pids, camids
