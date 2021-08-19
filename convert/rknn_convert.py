import yaml
from rknn.api import RKNN
import cv2

_model_load_dict = {
    'caffe': 'load_caffe',
    'tensorflow': 'load_tensorflow',
    'tflite': 'load_tflite',
    'onnx': 'load_onnx',
    'darknet': 'load_darknet',
    'pytorch': 'load_pytorch',
    'mxnet': 'load_mxnet',
    'rknn': 'load_rknn',
    }

yaml_file = './config.yaml'


def main():
    with open(yaml_file, 'r') as F:
        config = yaml.load(F)
    # print('config is:')
    # print(config)

    model_type = config['running']['model_type']
    print('model_type is {}'.format(model_type))#检查模型的类型

    rknn = RKNN(verbose=True)



#配置文件
    print('--> config model')
    rknn.config(**config['config'])
    print('done')


    print('--> Loading model')
    load_function = getattr(rknn, _model_load_dict[model_type])
    ret = load_function(**config['parameters'][model_type])
    if ret != 0:
        print('Load yolo failed! Ret = {}'.format(ret))
        exit(ret)
    print('done')

    ####
    #print('hybrid_quantization')
    #ret = rknn.hybrid_quantization_step1(dataset=config['build']['dataset'])


    if model_type != 'rknn':
        print('--> Building model')
        ret = rknn.build(**config['build'])
        print('acc_eval')
        rknn.accuracy_analysis(inputs='./dataset1.txt', target='rk3399pro')
        print('acc_eval done!')

        if ret != 0:
            print('Build yolo failed!')
            exit(ret)
    else:
        print('--> skip Building model step, cause the model is already rknn')


#导出RKNN模型
    if config['running']['export'] is True:
        print('--> Export RKNN model')
        ret = rknn.export_rknn(**config['export_rknn'])
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
    else:
        print('--> skip Export model')


#初始化
    print('--> Init runtime environment')
    ret = rknn.init_runtime(**config['init_runtime'])
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')


    print('--> load img')
    img = cv2.imread(config['img']['path'])
    print('img shape is {}'.format(img.shape))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = [img]
    print(inputs[0][0:10,0,0])
#推理
    if config['running']['inference'] is True:
        print('--> Running model')
        config['inference']['inputs'] = inputs
        #print(config['inference'])
        outputs = rknn.inference(inputs)
        #outputs = rknn.inference(config['inference'])
        print('len of output {}'.format(len(outputs)))
        print('outputs[0] shape is {}'.format(outputs[0].shape))
        print(outputs[0][0][0:2])
    else:
        print('--> skip inference')
#评价
    if config['running']['eval_perf'] is True:
        print('--> Begin evaluate model performance')
        config['inference']['inputs'] = inputs
        perf_results = rknn.eval_perf(inputs=[img])
    else:
        print('--> skip eval_perf')


if __name__ == '__main__':
    main()
