import time
import logging

import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, time_synchronized
from utils.augmentations import letterbox


class Terminal:
    def __init__(self,
                 weights='yolov5s.pt',  # model.pt path(s)
                 imgsz=640,  # inference size (pixels)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 update=False,  # update all models
                 half=False,  # use FP16 half-precision inference
                 log=True,  # logging print
                 ):
        # Initialize Variable
        self.imgsz = imgsz
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det

        # Initialize
        if log:
            set_logging()
        self.device = select_device(device)
        half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        self.half_ = half

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

        if half:
            self.model.half()  # to FP16

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    def forward(self, path):
        t0 = time.time()
        img0 = cv2.imread(path)

        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half_ else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        t2 = time_synchronized()
        boxes = list()
        logging.info('Model Time: inference cost %.3fms, NMS cost %.3fms, total cost %.3fms' %
                     ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3))

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # translate box
                for *xyxy, conf, cls in reversed(det):
                    item = dict()
                    item['label'] = self.names[int(cls)]
                    item['confidence'] = float(conf)
                    item['offset'] = [(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))]
                    boxes.append(item)
        return boxes


if __name__ == '__main__':
    m = Terminal(weights='weights/GapYolov5s.pt', imgsz=160, conf_thres=0.7, iou_thres=0.45)
    boxes = m.forward('data/demo/1001.png')
    for box in boxes:
        print(box)
