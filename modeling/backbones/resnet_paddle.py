#! encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

# import torch
import paddle
from paddle import nn
from reprod_log import ReprodLogger


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        reprod_logger.add("layer1", x.numpy())
        x = self.layer2(x)
        x = self.layer3(x)
        reprod_logger.add("layer3", x.numpy())

        x = self.layer4(x)
        reprod_logger.add("layer4", x.numpy())


        return x

    def load_param(self, model_path):
        param_dict = paddle.load(model_path)
        self.set_state_dict(param_dict)
        # for i in param_dict:
        #     if 'fc' in i:
        #         continue
        #     self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def convert_params(torch_params):
    convert_dict = {
        "running_mean": "_mean",
        "running_var": "_variance"
    }
    paddle_params = {}
    for src_key in torch_params:
        dst_key = src_key
        for torch_kw in convert_dict:
            if torch_kw in src_key:
                dst_key = dst_key.replace(torch_kw, convert_dict[torch_kw])
        paddle_params[dst_key] = torch_params[src_key].detach().numpy()
    return paddle_params


if __name__ == '__main__':
    import numpy as np
    reprod_logger = ReprodLogger()
    input_data = np.load("test_input.npy")
    resnet = ResNet()
    resnet.eval()
    import torch
    torch_params = torch.load("resnet50-0676ba61.pth")
    paddle_params = convert_params(torch_params)
    # print(resnet.state_dict().keys())
    resnet.set_state_dict(paddle_params)
    input_tensor = paddle.to_tensor(input_data)
    output = resnet(input_tensor)
    reprod_logger.add("output", output.numpy())
    reprod_logger.save("paddle_resnet.npy")
