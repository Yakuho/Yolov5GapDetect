# -*- coding: utf-8 -*-
# Author: Yakuho
# Date  : 2021/7/13
import xml.etree.ElementTree as xmlTree
import json
import os


def convert_(dw, dh, x0, y0, x1, y1, w=None, h=None):
    x_center = (x0 + x1) / 2.0
    y_center = (y0 + y1) / 2.0
    if w and h:
        pass
    else:
        w = x1 - x0
        h = y1 - y0
    return x_center / dw, y_center / dh, w / dw, h / dh


def voc2yolo(file, labels):
    data = list()
    tree = xmlTree.parse(file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for item in root.iter('object'):
        label = item.find('name').text
        box = item.find('bndbox')
        x0 = float(box.find('xmin').text)
        y0 = float(box.find('ymin').text)
        x1 = float(box.find('xmax').text)
        y1 = float(box.find('ymax').text)
        x_nor, y_nor, w_nor, h_nor = convert_(w, h, x0, y0, x1, y1)
        label_code = labels.index(label)
        data.append([label_code, x_nor, y_nor, w_nor, h_nor])
    return data


def coco2yolo(text, labels):
    data = list()
    json_dict = json.loads(text)
    w = json_dict['imageWidth']
    h = json_dict['imageHeight']
    for item in json_dict['shapes']:
        label = item['label']
        box = item['points']
        x0 = min(box[0][0], box[1][0])
        y0 = min(box[0][1], box[1][1])
        x1 = max(box[0][0], box[1][0])
        y1 = max(box[0][1], box[1][1])
        x_nor, y_nor, w_nor, h_nor = convert_(w, h, x0, y0, x1, y1)
        label_code = labels.index(label)
        data.append([label_code, x_nor, y_nor, w_nor, h_nor])
    return data


def all_convert(src, des, dataset_type, labels):
    total_type = ['voc', 'coco']
    input_type = [0, 1]     # 0: file   1: text
    if dataset_type.lower() not in total_type:
        raise ValueError('Having no code type of %s.' % dataset_type)
    else:
        open_type = total_type.index(dataset_type)
        convert = eval('%s2yolo' % dataset_type)
    files = os.listdir(src)
    print('found %s target' % len(files))
    if len(files) > 0:
        pass
    else:
        return 0
    for file in files:
        name, _ = file.split('.')
        with open(os.path.join(src, file), 'r') as f:
            if input_type[open_type]:
                input_data = f.read()
            else:
                input_data = f
            items = convert(input_data, labels)
        with open(os.path.join(des, '%s.txt' % name), 'w') as f:
            for item in items:
                context = '%s %s %s %s %s\n' % tuple(item)
                f.write(context)
    print('%s target convert success' % len(files))
    return 1
