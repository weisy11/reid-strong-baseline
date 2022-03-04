import torch
from reprod_log import ReprodDiffHelper
import numpy as np


def compare():
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("torch_center_loss.npy")
    paddle_info = diff_helper.load_info("paddle_center_loss.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="forward_diff.log")


if __name__ == '__main__':
    compare()
