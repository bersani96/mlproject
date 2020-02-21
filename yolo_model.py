import tensorflow as tf
import sys
from absl import app, logging, flags
from absl.flags import FLAGS
import time
import cv2
import numpy as np
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

class YoloModel:
    def __init__(self):
        # Initialize detector
        flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
        flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                            'path to weights file')
        flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
        flags.DEFINE_integer('size', 416, 'resize images to')
        flags.DEFINE_string('image', './data/girl.png', 'path to input image')
        flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
        flags.DEFINE_string('output', './output.jpg', 'path to output image')
        flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

        app._run_init(['yolov3'], app.parse_flags_with_usage)

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def detect(self, imagePath):
        # Detect image
        FLAGS.image = imagePath

        if FLAGS.tiny:
            yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        else:
            yolo = YoloV3(classes=FLAGS.num_classes)
            
        yolo.load_weights(FLAGS.weights).expect_partial()
        logging.info('weights loaded')

        logging.info('classes loaded')

        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        return boxes, scores, classes, nums