import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import keras

from keras_yolov3.train import get_anchors, get_classes, data_generator_wrapper
from keras_yolov3.yolo3.model import yolo_body, yolo_loss, yolo_eval
from keras_yolov3.yolo3.utils import letterbox_image

def load_inference_model(model_path, args):
    model = keras.models.load_model(model_path)
    print("type:", type(model))
    return model

def load_training_model(num_classes, args):
    anchors_path = '../keras_yolov3/model_data/yolo_anchors.txt'
    anchors = get_anchors(anchors_path)
    num_anchors = len(anchors)

    input_shape = (416, 416)

    image_input = keras.layers.Input(shape=(None, None, 3))

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)

    h, w = input_shape

    y_true = [keras.layers.Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                                        num_anchors//3, num_classes + 5)) for l in range(3)]

    model_loss = keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                     arguments={'anchors': anchors, 'num_classes': num_classes,
                                                'ignore_thresh': 0.5})([*model_body.output, *y_true])

    return keras.models.Model([model_body.input, *y_true], model_loss)

def preprocess_image(image, args):
    new_image_size = (image.width - (image.width % 32),
                      image.height - (image.height % 32))
    boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    return image_data, image_data.shape

def create_generators(args):
    classes_path = '../keras_yolov3/model_data/coco_classes.txt'
    anchors_path = '../keras_yolov3/model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    annotation_path = '../keras_yolov3/model_data/train.txt'
    with open(annotation_path) as f:
        lines = f.readlines()

    num_train = len(lines)
    input_shape = (416, 416)
    batch_size = 32
    set_rate = 0.9
    train = data_generator_wrapper(lines[:int(num_train*set_rate)], batch_size, input_shape, anchors, num_classes)
    valid = data_generator_wrapper(lines[int(num_train*set_rate):], batch_size, input_shape, anchors, num_classes)
    return [train, valid]

def get_losses(args):
    return {'yolo_loss': lambda y_true, y_pred: y_pred}

def postprocess_output(output, shape, args):
    classes_path = '../keras_yolov3/model_data/coco_classes.txt'
    anchors_path = '../keras_yolov3/model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    num_anchors = len(anchors)

    h = output[0].shape[1]*32
    w = output[0].shape[2]*32

    input_layer = [keras.layers.Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                                             num_anchors // 3 * (num_classes + 5))) for l in range(3)]

    model_eval = keras.layers.Lambda(yolo_eval, name='yolo_eval',
                                     arguments={'anchors': anchors, 'num_classes': num_classes,
                                                'image_shape': [shape[:2]], 'max_boxes': 20,
                                                'score_threshold': 0.3, 'iou_threshold': 0.45})(input_layer)

    model = keras.models.Model(input_layer, model_eval)

    return model.predict_on_batch(output)

def show_result(img_path, y_pred_thresh, args):
    return 0
