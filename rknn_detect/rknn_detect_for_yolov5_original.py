import cv2
import time
import random
import numpy as np
from rknn.api import RKNN

"""
yolov5 官方原版 预测脚本 for rknn
"""


def get_max_scale(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    return scale


def get_new_size(img, scale):
    return tuple(map(int, np.array(img.shape[:2][::-1]) * scale))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, conf_thres):
    box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
    box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引
    box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
    pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
    # pos = box_class_scores >= OBJ_THRESH  # 找出概率大于阈值的item
    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]
    return boxes, classes, scores


def nms_boxes(boxes, scores, iou_thres):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def auto_resize(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    new_size = tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    return cv2.resize(img, new_size), scale


def letterbox(img, new_wh=(416, 416), color=(114, 114, 114)):
    new_img, scale = auto_resize(img, *new_wh)
    shape = new_img.shape
    new_img = cv2.copyMakeBorder(new_img, 0, new_wh[1] - shape[0], 0, new_wh[0] - shape[1], cv2.BORDER_CONSTANT,
                                 value=color)
    return new_img, (new_wh[0] / scale, new_wh[1] / scale)


def load_model(model_path, npu_id):
    rknn = RKNN()
    devs = rknn.list_devices()
    device_id_dict = {}
    for index, dev_id in enumerate(devs[-1]):
        if dev_id[:2] != 'TS':
            device_id_dict[0] = dev_id
        if dev_id[:2] == 'TS':
            device_id_dict[1] = dev_id
    print('-->loading model : ' + model_path)
    rknn.load_rknn(model_path)
    print('--> Init runtime environment on: ' + device_id_dict[npu_id])
    ret = rknn.init_runtime(device_id=device_id_dict[npu_id])
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


class Detector:
    def __init__(self, opt):
        self.opt = opt

        model = opt['model']
        wh = opt['size']
        masks = opt['masks']
        anchors = opt['anchors']
        names = opt['names']
        conf_thres = opt['conf_thres']
        iou_thres = opt['iou_thres']
        platform = opt['platform']

        self.wh = wh
        self.size = wh
        self._masks = masks
        self._anchors = anchors
        self.names = list(
            filter(lambda a: len(a) > 0, map(lambda x: x.strip(), open(names, "r").read().split()))) if isinstance(
            names, str) else names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        if isinstance(model, str):
            model = load_model(model, platform)
        self._rknn = model
        self.draw_box = False

    def _predict(self, img_src, img, gain):
        src_h, src_w = img_src.shape[:2]
        # _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        img = img[..., ::-1]  #
        img = np.concatenate([img[::2, ::2], img[1::2, ::2], img[::2, 1::2], img[1::2, 1::2]], 2)
        t0 = time.time()
        pred_onx = self._rknn.inference(inputs=[img])
        print("inference time:\t", time.time() - t0)
        boxes, classes, scores = [], [], []
        for t in range(3):
            input0_data = sigmoid(pred_onx[t][0])
            input0_data = np.transpose(input0_data, (1, 2, 0, 3))
            grid_h, grid_w, channel_n, predict_n = input0_data.shape
            anchors = [self._anchors[i] for i in self._masks[t]]
            box_confidence = input0_data[..., 4]
            box_confidence = np.expand_dims(box_confidence, axis=-1)
            box_class_probs = input0_data[..., 5:]
            box_xy = input0_data[..., :2]
            box_wh = input0_data[..., 2:4]
            col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
            row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
            col = col.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            row = row.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)
            box_xy = box_xy * 2 - 0.5 + grid
            box_wh = (box_wh * 2) ** 2 * anchors
            box_xy /= (grid_w, grid_h)  # 计算原尺寸的中心
            box_wh /= self.wh  # 计算原尺寸的宽高
            box_xy -= (box_wh / 2.)  # 计算原尺寸的中心
            box = np.concatenate((box_xy, box_wh), axis=-1)
            res = filter_boxes(box, box_confidence, box_class_probs, self.conf_thres)
            boxes.append(res[0])
            classes.append(res[1])
            scores.append(res[2])
        boxes, classes, scores = np.concatenate(boxes), np.concatenate(classes), np.concatenate(scores)
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = nms_boxes(b, s, self.iou_thres)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        if len(nboxes) < 1:
            return [], []
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        label_list = []
        box_list = []
        for (x, y, w, h), score, cl in zip(boxes, scores, classes):
            x *= gain[0]
            y *= gain[1]
            w *= gain[0]
            h *= gain[1]
            x1 = max(0, np.floor(x).astype(int))
            y1 = max(0, np.floor(y).astype(int))
            x2 = min(src_w, np.floor(x + w + 0.5).astype(int))
            y2 = min(src_h, np.floor(y + h + 0.5).astype(int))
            # label_list.append(self.names[cl])
            label_list.append(cl)
            box_list.append((x1, y1, x2, y2))
            if self.draw_box:
                plot_one_box((x1, y1, x2, y2), img_src, label=self.names[cl])
        return label_list, np.array(box_list)

    def detect_resize(self, img_src):
        """
        预测一张图片，预处理使用resize
        return: labels,boxes
        """
        _img = cv2.resize(img_src, self.wh)
        gain = img_src.shape[:2][::-1]
        return self._predict(img_src, _img, gain)

    def detect(self, img_src):
        """
        预测一张图片，预处理保持宽高比
        return: labels,boxes
        """
        _img, gain = letterbox(img_src, self.wh)
        return self._predict(img_src, _img, gain)

    def close(self):
        self._rknn.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def test_video(det, video_path):
    reader = cv2.VideoCapture()
    reader.open(video_path)
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        t0 = time.time()
        det.detect(frame)
        print("total time", time.time() - t0)
        cv2.imshow("res", auto_resize(frame, 1200, 600)[0])
        cv2.waitKey(1)


if __name__ == '__main__':
    import yaml
    import cv2

    image = cv2.imread("data/images/bus.jpg")
    with open("models/yolov5_rknn_640x640.yaml", "rb") as f:
        cfg = yaml.load(f, yaml.FullLoader)
        d = Detector(cfg["opt"])
        d.draw_box = True
        d.detect(image)
        cv2.imshow("res", image)
        cv2.waitKey()
    cv2.destroyAllWindows()
