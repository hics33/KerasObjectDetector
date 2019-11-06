import sys, os
sys.path.insert(0, os.path.abspath('..'))
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import yolk
from yolk.parser import parse_args


def main(args=None):
    args = []

    image = Image.open('000000008021.jpg')
    image, shape = yolk.detector.preprocessing_image(image, args)

    model_path = os.path.join('..', 'yolo.h5')
    model = yolk.detector.load_inference_model(model_path, args)

    model_output = model.predict_on_batch(np.expand_dims(image, axis=0))

    boxes, scores, labels = yolk.detector.postprocessing_output(model_output, shape)

    print(boxes)

if __name__ == '__main__':
    main()
