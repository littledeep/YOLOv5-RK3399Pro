"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r'D:\workspace\LMO\opengit\yolov5\runs\train\exp\weights\best.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--rknn_mode', action='store_true', help='export rknn-friendly onnx model')
    parser.add_argument('--ignore_output_permute', action='store_true', help='export model without permute layer,which can be used for rknn_yolov5_demo c++ code')
    parser.add_argument('--add_image_preprocess_layer', action='store_true', help='add image preprocess layer, benefit for decreasing rknn_input_set time-cost')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Update model
    if opt.rknn_mode != True:
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, models.common.Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            # elif isinstance(m, models.yolo.Detect):
            #     m.forward = m.forward_export  # assign forward (optional)

    else:
        # Update model
        from models.common_rk_plug_in import surrogate_silu
        from models import common
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, common.Conv):  # assign export-friendly activations
                if isinstance(m.act, torch.nn.Hardswish):
                    m.act = torch.nn.Hardswish()
                elif isinstance(m.act, torch.nn.SiLU):
                    # m.act = SiLU()
                    m.act = surrogate_silu()
            # elif isinstance(m, models.yolo.Detect):
            #     m.forward = m.forward_export  # assign forward (optional)

            if isinstance(m, models.common.SPP):  # assign export-friendly activations
                ### best
                # tmp = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(2)])
                # m.m[0] = tmp
                # m.m[1] = tmp
                # m.m[2] = tmp
                ### friendly to origin config
                tmp = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(2)])
                m.m[0] = tmp
                tmp = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(4)])
                m.m[1] = tmp
                tmp = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(6)])
                m.m[2] = tmp

        ### use deconv2d to surrogate upsample layer.
        # replace_one = torch.nn.ConvTranspose2d(model.model[10].conv.weight.shape[0],
        #                                        model.model[10].conv.weight.shape[0],
        #                                        (2, 2),
        #                                        groups=model.model[10].conv.weight.shape[0],
        #                                        bias=False,
        #                                        stride=(2, 2))
        # replace_one.weight.data.fill_(1)
        # replace_one.eval()
        # temp_i = model.model[11].i
        # temp_f = model.model[11].f
        # model.model[11] = replace_one
        # model.model[11].i = temp_i
        # model.model[11].f = temp_f

        # replace_one = torch.nn.ConvTranspose2d(model.model[14].conv.weight.shape[0],
        #                                        model.model[14].conv.weight.shape[0],
        #                                        (2, 2),
        #                                        groups=model.model[14].conv.weight.shape[0],
        #                                        bias=False,
        #                                        stride=(2, 2))
        # replace_one.weight.data.fill_(1)
        # replace_one.eval()
        # temp_i = model.model[11].i
        # temp_f = model.model[11].f
        # model.model[15] = replace_one
        # model.model[15].i = temp_i
        # model.model[15].f = temp_f

        ### use conv to surrogate slice operator
        from models.common_rk_plug_in import surrogate_focus
        surrogate_focous = surrogate_focus(int(model.model[0].conv.conv.weight.shape[1]/4),
                                           model.model[0].conv.conv.weight.shape[0],
                                           k=tuple(model.model[0].conv.conv.weight.shape[2:4]),
                                           s=model.model[0].conv.conv.stride,
                                           p=model.model[0].conv.conv.padding,
                                           g=model.model[0].conv.conv.groups,
                                           act=True)
        surrogate_focous.conv.conv.weight = model.model[0].conv.conv.weight
        surrogate_focous.conv.conv.bias = model.model[0].conv.conv.bias
        surrogate_focous.conv.act = model.model[0].conv.act
        temp_i = model.model[0].i
        temp_f = model.model[0].f

        model.model[0] = surrogate_focous
        model.model[0].i = temp_i
        model.model[0].f = temp_f
        model.model[0].eval()

    model.model[-1].export = not opt.grid  # set Detect() layer grid export

    if opt.ignore_output_permute is True:
        model.model[-1].ignore_permute_layer = True

    if opt.add_image_preprocess_layer is True:
        from models.common_rk_plug_in import preprocess_conv_layer
        model = preprocess_conv_layer(model, 0, 255, True)
        img = torch.zeros(opt.batch_size, *opt.img_size, 3).to(device)
    else:
        img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)

    y = model(img)  # dry run
    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=10, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                        'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)
