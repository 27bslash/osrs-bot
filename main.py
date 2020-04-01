# Imports
from mss import mss
import cv2
from PIL import Image, ImageGrab
from matplotlib import pyplot as plt
from io import StringIO
from collections import defaultdict
from distutils.version import StrictVersion
import pathlib
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
import numpy as np
import six.moves.urllib as urllib
import sys
import os
import time
import pyautogui
from random import randint, uniform
from pyclick import HumanClicker
from real_click import real_click
import math


MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = "D:/projects/osrs-bot/models/research/object_detection/inference_graph/frozen_inference_graph.pb"
PATH_TO_LABELS = "D:/projects/osrs-bot/models/research/object_detection/training/labelmap.pbtxt"


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
# Loading label map
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)


def run_inference_for_single_image(image, graph):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(
            tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                   real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                   real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


try:
    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            sct = mss()
            mon = {'top': 10, 'left': 10, 'width': 1920, 'height': 1080}
            mining = False
            current_ore = []
            class_array = []
            count = 0
            while True:
                sct.get_pixels(mon)
                img = Image.frombytes(
                    'RGB', (sct.width, sct.height), sct.image)
                start_time = time.time()
                image_np = np.array(img)
                # image_np = numpy_flip(image_np)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                # image_np = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = run_inference_for_single_image(
                    image_np, detection_graph)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)
                boxes = output_dict['detection_boxes']
                # get all boxes from an array
                max_boxes_to_draw = boxes.shape[0]
                # get scores to get a threshold
                scores = output_dict['detection_scores']
                # this is set as a default but feel free to adjust it to your needs
                min_score_thresh = 0.75
                # iterate over all objects found
                hc = HumanClicker()
                for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                    #
                    if scores is None or scores[i] > min_score_thresh:
                        # boxes[i] is the box which will be drawn
                        class_name = category_index[output_dict['detection_classes'][i]]['name']
                        xmax = boxes[i][3] * 1920
                        xmin = boxes[i][1] * 1920
                        ymax = boxes[i][2] * 1080
                        ymin = boxes[i][0] * 1080
                        x_center = (xmax + xmin) / 2
                        y_center = (ymax + ymin) / 2
                        temp = []
                        temp.append(class_name)
                        temp.append(x_center)
                        temp.append(y_center)
                        class_array.append(temp)

                length = math.floor(len(class_array)/3)
                for i in range(len(class_array)):
                    if class_array[i][0] == 'bag-iron-ore' and mining == False:
                        x_center = class_array[i][1]
                        y_center = class_array[i][2]
                        hc.move((math.floor(x_center) - randint(-5, 5),
                                 math.floor(y_center) - randint(-5, 5)), uniform(0.3, 0.5))
                        real_click('shift')
                    if class_array[i][0] == 'unmined-iron-ore' and mining == False:
                        x_center = class_array[i][1]
                        y_center = class_array[i][2]
                        current_ore.append(class_array[i][0])
                        current_ore.append(x_center)
                        current_ore.append(y_center)
                        hc.move((math.floor(x_center - randint(-40, 40)),
                                 math.floor(y_center - randint(-40, 40))), uniform(0.3, 0.5))
                        real_click(None)
                        mining = True
                        time.sleep(uniform(1, 1.5))
                        print(current_ore)

                for i in range(len(class_array)):
                    if mining == True and current_ore[1] > 1 and class_array[i][0] == 'mined-ore':
                        if round(class_array[i][1], -2) == round(current_ore[1], -2) or round(class_array[i][2], -2) == round(current_ore[2], -2):
                            mining = False
                            print(mining, 'hurrah')
                            current_ore = []
                            count = 0

                class_array = []
                cv2.imshow('window', cv2.cvtColor(
                    image_np, cv2.COLOR_BGR2RGB))
                pyautogui.keyUp('shift')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
except Exception as e:
    tb = sys.exc_info()[0]
    tb.print_exc()
    print(e, tb)
