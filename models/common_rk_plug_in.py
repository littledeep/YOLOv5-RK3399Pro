# This file contains modules common to various models

import torch
import torch.nn as nn
from models.common import Conv


class surrogate_silu(nn.Module):
    """docstring for surrogate_silu"""
    def __init__(self):
        super(surrogate_silu, self).__init__()
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x*self.act(x)


class surrogate_hardswish(nn.Module):
    """docstring for surrogate_hardswish"""
    def __init__(self):
        super(surrogate_hardswish, self).__init__()
        self.relu6 = nn.ReLU()

    def forward(self, x):
        return x *(self.relu6(torch.add(x, 3))/6)


class surrogate_focus(nn.Module):
    # surrogate_focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(surrogate_focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

        with torch.no_grad():
            self.conv1 = nn.Conv2d(3, 3, (2, 2), groups=3, bias=False, stride=(2, 2))
            self.conv1.weight[:, :, 0, 0] = 1
            self.conv1.weight[:, :, 0, 1] = 0
            self.conv1.weight[:, :, 1, 0] = 0
            self.conv1.weight[:, :, 1, 1] = 0

            self.conv2 = nn.Conv2d(3, 3, (2, 2), groups=3, bias=False, stride=(2, 2))
            self.conv2.weight[:, :, 0, 0] = 0
            self.conv2.weight[:, :, 0, 1] = 0
            self.conv2.weight[:, :, 1, 0] = 1
            self.conv2.weight[:, :, 1, 1] = 0

            self.conv3 = nn.Conv2d(3, 3, (2, 2), groups=3, bias=False, stride=(2, 2))
            self.conv3.weight[:, :, 0, 0] = 0
            self.conv3.weight[:, :, 0, 1] = 1
            self.conv3.weight[:, :, 1, 0] = 0
            self.conv3.weight[:, :, 1, 1] = 0

            self.conv4 = nn.Conv2d(3, 3, (2, 2), groups=3, bias=False, stride=(2, 2))
            self.conv4.weight[:, :, 0, 0] = 0
            self.conv4.weight[:, :, 0, 1] = 0
            self.conv4.weight[:, :, 1, 0] = 0
            self.conv4.weight[:, :, 1, 1] = 1

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], 1))


class preprocess_conv_layer(nn.Module):
    """docstring for preprocess_conv_layer"""
    #   input_module 为输入模型，即为想要导出模型
    #   mean_value 的值可以是 [m1, m2, m3] 或 常数m
    #   std_value 的值可以是 [s1, s2, s3] 或 常数s
    #   BGR2RGB的操作默认为首先执行，既替代的原有操作顺序为 
    #       BGR2RGB -> minus mean -> minus std (与rknn config 设置保持一致) -> nhwc2nchw
    #
    #   使用示例-伪代码：
    #       from add_preprocess_conv_layer import preprocess_conv_layer
    #       model_A = create_model()
    #       model_output = preprocess_co_nv_layer(model_A, mean_value, std_value, BGR2RGB)
    #       onnx_export(model_output)
    #
    #   量化时：
    #       rknn.config的中 channel_mean_value 、reorder_channel 均不赋值。
    #
    #   部署代码：
    #       rknn_input 的属性
    #           pass_through = 1
    #
    #   另外：
    #       由于加入permute操作，c端输入为opencv mat(hwc格式)即可，无需在外部将hwc改成chw格式。
    #

    def __init__(self, input_module, mean_value, std_value, BGR2RGB=False):
        super(preprocess_conv_layer, self).__init__()
        if isinstance(mean_value, int):
            mean_value = [mean_value for i in range(3)]
        if isinstance(std_value, int):
            std_value = [std_value for i in range(3)]

        assert len(mean_value) <= 3, 'mean_value should be int, or list with 3 element'
        assert len(std_value) <= 3, 'std_value should be int, or list with 3 element'

        self.input_module = input_module

        with torch.no_grad():
            self.conv1 = nn.Conv2d(3, 3, (1, 1), groups=1, bias=True, stride=(1, 1))

            if BGR2RGB is False:
                self.conv1.weight[:, :, :, :] = 0
                self.conv1.weight[0, 0, :, :] = 1/std_value[0]
                self.conv1.weight[1, 1, :, :] = 1/std_value[1]
                self.conv1.weight[2, 2, :, :] = 1/std_value[2]
            elif BGR2RGB is True:
                self.conv1.weight[:, :, :, :] = 0
                self.conv1.weight[0, 2, :, :] = 1/std_value[0]
                self.conv1.weight[1, 1, :, :] = 1/std_value[1]
                self.conv1.weight[2, 0, :, :] = 1/std_value[2]

            self.conv1.bias[0] = -mean_value[0]/std_value[0]
            self.conv1.bias[1] = -mean_value[1]/std_value[1]
            self.conv1.bias[2] = -mean_value[2]/std_value[2]

        self.conv1.eval()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW, apply for rknn_pass_through
        x = self.conv1(x)
        return self.input_module(x)