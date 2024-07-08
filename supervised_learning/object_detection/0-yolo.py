#!/usr/bin/env python3
"""a class Yolo that uses the Yolo v3 algorithm to perform object detection"""
from tensorflow import keras as K


class Yolo:
    """yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
