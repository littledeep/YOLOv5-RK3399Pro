## Description：

Original repo：https://github.com/ultralytics/yolov5

https://github.com/EASY-EAI/yolov5

Requirement：python version >= 3.6

Training：python3 train.py

Model exporting：python3 models/export.py --rknn_mode

Model inference：python3 detect.py --rknn_mode



## Explanation：

The activation layer in the common file is changed to ReLU, and the model structure, training, testing and other operations are the same as the original version of Yolov5(4.0release). Add rknn_mode mode for model testing and exporting to export rknn-friendly models. (Based on opset_version=10, rknn_toolkit_1.6.0 test passed)

Detail：

- onnx.slice reports errors when loading models in rknn_toolkit_1.6.0. Add equivalent replacement operation. (convolution)



***4.29 update***：

- Export models can optionally remove the trailing permute layer to be compatible with **rknn_yolov5_demo's c++ deployment code**

  `python3 models/export.py --rknn_mode --ignore_output_permute`

- The export model can optionally add a **image preprocessing layer**, which can **effectively reduce the time consumption of the deployment segment rknn_input_set**. See the description of preprocess_conv_layer in models/common_rk_plug_in.py for details on how to use it.

  `python3 models/export.py --rknn_mode --add_image_preprocess_layer`

  （rknn_mode、ignore_output_permute、add_image_preprocess_layer . The three are not mutually exclusive and can be used simultaneously）

- Add onnx->rknn model export tool, see rknn_convert_tools folder for details.

- Add pre-compiled code to reduce the loading time of the model

- Add model inference code in RK3399Pro

##### *5.12 update*：

- When exporting the model using `--rknn_mode`, the large `maxpool` is equivalently replaced by multiple smaller `maxpools` by default, which has no effect on the computational results, but can significantly improve the inference speed on rknpu.



## Description of known problems (have no affect on current use)：

- onnx.opset_version=12 does not support SiLU activation layer, add equivalent alternative model to solve it. (x* sigmoid(x)) But rknn_toolkit_1_6_0 works fine in simulations, deploying to the board side will cause an exception. Default is not used for now, waiting for **rockchip** to fix.
-  onnx.upsample.opset_version=12 Implementation in rknn_toolkit_1.6.0 Temporarily problematic, add equivalence replacement model. (deconvolution). rknn_toolkit_1_6_0 works fine in simulation, deploying to the board side results in an exception. Default is not used for now, waiting for **rockchip.inc** to fix.



------

### rk_npu speed test<sup>[4](#脚注4)</sup> (ms)：

| Model(416x416 input) | rknn_toolkit_1.6.0 Simulators (800MHZ)_rv1109 | rv1109<sup>[3](#脚注3)</sup> | rv1126  | rv1126(Model pre-compiling) | rknn_toolkit_1.6.0 Simulators (800MHZ)_rk1808 | rk1808 | rk1808(Model pre-compiling) |
| :---------------------- | :-----------------------------------------: | :-------: | :-----: | :----------------: | :-----------------------------------------: | :----: | :----------------: |
| yolov5s_int8<sup>[1](#脚注1)</sup> |                     92                      |    113    |   80    |         77         |                     89                      |   83   |         81         |
| yolov5s_int8_optimize<sup>[2](#脚注2)</sup> |                   **18**                    |  **45**   | **36**  |       **33**       |                   **15**                    | **30** |       **29**       |
| yolov5s_int16           |                     149                     |    160    |   110   |        108         |                     106                     |  178   |        174         |
| yolov5s_int16_optimize  |                     76                      |    90     |   67    |         64         |                     32                      |  126   |        122         |
| yolov5m_int8            |                     158                     |    192    |   132   |        120         |                     144                     |  132   |        123         |
| yolov5m_int8_optimize   |                   **47**                    |  **88**   | **66**  |       **55**       |                   **33**                    | **54** |       **45**       |
| yolov5m_int16           |                     312                     |    302    |   212   |        202         |                     187                     |  432   |        418         |
| yolov5m_int16_optimize  |                     202                     |    198    |   147   |        137         |                     76                      |  354   |        344         |
| yolov5l_int8            |                     246                     |    293    |   199   |                    |                     214                     |  192   |                    |
| yolov5l_int8_optimize   |                   **98**                    |  **155**  | **110** |                    |                   **66**                    | **88** |                    |
| yolov5l_int16           |                     577                     |    522    |   362   |                    |                     301                     |  697   |                    |
| yolov5l_int16_optimize  |                     432                     |    384    |   275   |                    |                     154                     |  592   |                    |

<a name="脚注1">1</a>: is based on the original yaml configuration, with the activation layer modified to relu.

<a name="脚注2">2</a>: optimize means to optimize the large size maxpool when exporting the model, now open source, used by default when exporting the parameter --rknn_mode. It does not affect the accuracy.

<a name="脚注3">3</a>: Statistical time includes **rknn_inputs_set**, **rknn_run**, **rknn_outputs_get** three parts of time, excluding post-processing time on the cpu side. This principle is followed for the tests on other platforms in this table except for the simulator evaluation.

<a name="脚注4">4</a>: This test is for reference only, the test is a single-threaded loop execution timing, only test npu efficiency. The actual use should consider the post-processing time.



## Chinese Blog(中文博客)

https://blog.csdn.net/weixin_42200352/article/details/119772850



## Reference：

https://github.com/soloIife/yolov5_for_rknn

https://github.com/ultralytics/yolov5

https://github.com/EASY-EAI/yolov5

RKNN QQ group:  1025468710





