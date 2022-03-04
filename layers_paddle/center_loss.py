from __future__ import absolute_import

import paddle
from paddle import nn


class CenterLoss(nn.Layer):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = paddle.create_parameter([self.num_classes, self.feat_dim], "float32")
        else:
            self.centers = paddle.create_parameter([self.num_classes, self.feat_dim], "float32")

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.shape[0] == labels.shape[0], "features.size(0) is not equal to labels.size(0)"

        batch_size = x.shape[0]
        distmat = paddle.pow(x, 2).sum(axis=1, keepdim=True).expand((batch_size, self.num_classes)) +\
                  paddle.pow(self.centers, 2).sum(axis=1, keepdim=True).expand((self.num_classes, batch_size)).t()
        reprod_logger.add("distmat", distmat.numpy())
        distmat = distmat - 2 * paddle.matmul(x, self.centers.t())

        classes = paddle.arange(self.num_classes)
        labels = labels.unsqueeze(1).expand((batch_size, self.num_classes))
        mask = labels.equal(classes.expand((batch_size, self.num_classes)))

        dist = distmat * mask
        loss = dist.clip(min=1e-12, max=1e+12).sum() / batch_size
        return loss


if __name__ == '__main__':
    import numpy as np
    import torch
    from reprod_log import ReprodLogger

    reprod_logger = ReprodLogger()
    use_gpu = False
    center_loss = CenterLoss(use_gpu=use_gpu)
    center_loss.eval()
    torch_weight = torch.load("center_loss_weight.npy")
    paddle_weight = {"centers": torch_weight["centers"].detach().numpy()}
    center_loss.set_state_dict(paddle_weight)
    features = np.load("features_input.npy")
    features = paddle.to_tensor(features)
    targets = paddle.to_tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4])
    loss = center_loss(features, targets)
    print(loss)
    print(center_loss.state_dict().keys())
    reprod_logger.add("output", loss.numpy())
    reprod_logger.save("paddle_centerloss.npy")
