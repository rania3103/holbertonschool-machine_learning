#!/usr/bin/env python3
"""a class Yolo that uses the Yolo v3 algorithm to perform object detection"""
from tensorflow import keras as K
import tensorflow as tf
import numpy as np


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

    def process_outputs(self, outputs, image_size):
        """Returns a tuple of (boxes, box_confidences, box_class_probs)"""
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []
        for output in outputs:
            grid_height, grid_width, anchor_boxes, value = output.shape
            box_confidence = tf.sigmoid(output[..., 4:5])
            box_class_prob = tf.sigmoid(output[..., 5:])
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)
            t_xy = tf.sigmoid(output[..., :2])
            t_wh = tf.exp(output[..., 2:4])
            for g_h in range(grid_height):
                for g_w in range(grid_width):
                    for anchor in range(anchor_boxes):
                        center_x = (
                            g_w + t_xy[g_h, g_w, anchor, 0]) / grid_width
                        center_y = (
                            g_h + t_xy[g_h, g_w, anchor, 1]) / grid_height
                        w = t_wh[g_h, g_w, anchor, 0] * \
                            self.anchors[anchor][0] / image_width
                        h = t_wh[g_h, g_w, anchor, 1] * \
                            self.anchors[anchor][1] / image_height
                        x1 = (center_x - w / 2) * image_width
                        y1 = (center_y - h / 2) * image_height
                        x2 = (center_x + w / 2) * image_width
                        y2 = (center_y + h / 2) * image_height
                        boxes.append([x1, y1, x2, y2])
        return boxes, box_confidences, box_class_probs
